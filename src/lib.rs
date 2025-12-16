use pyo3::prelude::pymodule;
pub mod qvd_structure;

#[pymodule]
mod qvd {
    use bitvec::prelude::*;
    use pyo3::{prelude::*, types::PyDict};
    use quick_xml::de::from_str;
    use crate::qvd_structure::{QvdFieldHeader, QvdTableHeader};
    use std::io::SeekFrom;
    use std::io::{self, Read};
    use std::path::Path;
    use std::str;
    use std::{collections::HashMap, fs::File};
    use std::{io::prelude::*};
    use parquet::{
        basic::{ConvertedType, Type as PhysicalType, Repetition},
        column::writer::ColumnWriter,
        data_type::ByteArray,
        file::{
            properties::WriterProperties,
            writer::{SerializedFileWriter}
        },
        schema::types::Type,
    };
    use parquet::file::writer::{SerializedColumnWriter, SerializedRowGroupWriter};
    use std::sync::Arc;

    #[derive(Clone, Debug)]
    enum SymbolValue {
        Str(String),
        Int(i32),
        Double(f64),
    }

    #[derive(Debug)]
    enum ColumnData {
        Utf8(Vec<Option<String>>),
        Int32(Vec<Option<i32>>),
        Int64(Vec<Option<i64>>),
        Double(Vec<Option<f64>>),
    }

    #[pyfunction]
    fn read_qvd(py: Python, file_name: String) -> PyResult<Py<PyDict>> {
        let xml: String = get_xml_data(&file_name).expect("Error reading file");
        let dict = PyDict::new(py);
        let binary_section_offset = xml.as_bytes().len();

        let qvd_structure: QvdTableHeader = from_str(&xml).unwrap();
        let mut symbol_map: HashMap<String, Vec<Option<String>>> = HashMap::new();

        if let Ok(f) = File::open(&file_name) {
            // Seek to the end of the XML section
            let buf = read_qvd_to_buf(f, binary_section_offset);
            let rows_start = qvd_structure.offset;
            let rows_end = buf.len();
            let rows_section = &buf[rows_start..rows_end];
            let record_byte_size = qvd_structure.record_byte_size;

            for field in qvd_structure.fields.headers {
                symbol_map.insert(
                    field.field_name.clone(),
                    get_symbols_as_strings(&buf, &field),
                );
                let symbol_indexes = get_row_indexes(&rows_section, &field, record_byte_size);
                let column_values =
                    match_symbols_with_indexes(&symbol_map[&field.field_name], &symbol_indexes);
                dict.set_item(field.field_name, column_values).unwrap();
            }
        }
        Ok(dict.into())
    }

    #[pyfunction]
    fn transform_qvd_to_parquet(_py: Python,
                                source_file_name: String,
                                target_file_name: String) -> PyResult<()> {
        _transform_qvd_to_parquet(source_file_name, target_file_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    fn _transform_qvd_to_parquet(source_file_name: String,
                                target_file_name: String) -> Result<(), String>{
        let xml: String = get_xml_data(&source_file_name).map_err(|e| e.to_string())?;
        let binary_section_offset = xml.as_bytes().len();
        let qvd_structure: QvdTableHeader = from_str(&xml).map_err(|e| e.to_string())?;

        let mut columns_data: Vec<(String, ColumnData)> = Vec::new();

        if let Ok(f) = File::open(&source_file_name) {
            let buf = read_qvd_to_buf(f, binary_section_offset);
            let rows_start = qvd_structure.offset;
            let rows_end = buf.len();
            let rows_section = &buf[rows_start..rows_end];
            let record_byte_size = qvd_structure.record_byte_size;

            for field in qvd_structure.fields.headers {
                // Read symbols as typed values
                let symbols_typed = get_symbols_as_values(&buf, &field);
                let symbol_indexes = get_row_indexes(&rows_section, &field, record_byte_size);
                let col_values_typed = match_values_with_indexes(&symbols_typed, &symbol_indexes);

                // Decide target Parquet type for the column
                let mut all_numeric_or_null = true;
                let mut has_double = false;
                let mut needs_i64 = false;
                let mut _only_int = true;

                for ov in &col_values_typed {
                    if let Some(v) = ov {
                        match v {
                            SymbolValue::Int(_) => { /* still int */ }
                            SymbolValue::Double(_) => { has_double = true; _only_int = false; }
                            SymbolValue::Str(s) => {
                                // Try parse integer first
                                if let Ok(iv) = s.parse::<i64>() {
                                    if iv < i64::from(i32::MIN) || iv > i64::from(i32::MAX) {
                                        needs_i64 = true;
                                    }
                                    // still integer
                                } else if let Ok(_) = s.parse::<f64>() {
                                    has_double = true;
                                    _only_int = false;
                                } else {
                                    all_numeric_or_null = false;
                                }
                            }
                        }
                    }
                }

                let column_data = if all_numeric_or_null {
                    if has_double {
                        // Promote to double
                        let mut values: Vec<Option<f64>> = Vec::with_capacity(col_values_typed.len());
                        for ov in col_values_typed {
                            match ov {
                                Some(SymbolValue::Int(i)) => values.push(Some(i as f64)),
                                Some(SymbolValue::Double(d)) => values.push(Some(d)),
                                Some(SymbolValue::Str(s)) => values.push(s.parse::<f64>().ok()),
                                None => values.push(None),
                            }
                        }
                        ColumnData::Double(values)
                    } else if needs_i64 {
                        let mut values: Vec<Option<i64>> = Vec::with_capacity(col_values_typed.len());
                        for ov in col_values_typed {
                            match ov {
                                Some(SymbolValue::Int(i)) => values.push(Some(i as i64)),
                                Some(SymbolValue::Str(s)) => values.push(s.parse::<i64>().ok()),
                                Some(SymbolValue::Double(d)) => values.push(Some(d as i64)), // unlikely path
                                None => values.push(None),
                            }
                        }
                        ColumnData::Int64(values)
                    } else {
                        // Int32
                        let mut values: Vec<Option<i32>> = Vec::with_capacity(col_values_typed.len());
                        for ov in col_values_typed {
                            match ov {
                                Some(SymbolValue::Int(i)) => values.push(Some(i)),
                                Some(SymbolValue::Str(s)) => values.push(s.parse::<i32>().ok()),
                                Some(SymbolValue::Double(d)) => values.push(Some(d as i32)), // unlikely path
                                None => values.push(None),
                            }
                        }
                        ColumnData::Int32(values)
                    }
                } else {
                    // Fallback to UTF8 strings
                    let mut values: Vec<Option<String>> = Vec::with_capacity(col_values_typed.len());
                    for ov in col_values_typed {
                        match ov {
                            Some(SymbolValue::Str(s)) => values.push(Some(s)),
                            Some(SymbolValue::Int(i)) => values.push(Some(i.to_string())),
                            Some(SymbolValue::Double(d)) => values.push(Some(d.to_string())),
                            None => values.push(None),
                        }
                    }
                    ColumnData::Utf8(values)
                };

                columns_data.push((field.field_name, column_data));
            }
        } else {
            return Err(format!("Could not open file: {}", source_file_name));
        }

        // Create Parquet Schema
        let mut fields: Vec<Arc<Type>> = Vec::new();
        for (name, col) in &columns_data {
            let field_type = match col {
                ColumnData::Utf8(_) => Type::primitive_type_builder(name, PhysicalType::BYTE_ARRAY)
                    .with_repetition(Repetition::OPTIONAL)
                    .with_converted_type(ConvertedType::UTF8)
                    .build()
                    .map_err(|e| e.to_string())?,
                ColumnData::Int32(_) => Type::primitive_type_builder(name, PhysicalType::INT32)
                    .with_repetition(Repetition::OPTIONAL)
                    .build()
                    .map_err(|e| e.to_string())?,
                ColumnData::Int64(_) => Type::primitive_type_builder(name, PhysicalType::INT64)
                    .with_repetition(Repetition::OPTIONAL)
                    .build()
                    .map_err(|e| e.to_string())?,
                ColumnData::Double(_) => Type::primitive_type_builder(name, PhysicalType::DOUBLE)
                    .with_repetition(Repetition::OPTIONAL)
                    .build()
                    .map_err(|e| e.to_string())?,
            };
            fields.push(Arc::new(field_type));
        }

        let schema = Type::group_type_builder("schema")
            .with_fields(fields)
            .build()
            .map_err(|e| e.to_string())?;
        let schema_ptr = Arc::new(schema);

        // Create Parquet Writer
        let file: File = File::create(&target_file_name).map_err(|e| e.to_string())?;
        let props = Arc::new(WriterProperties::builder().build());
        let mut writer: SerializedFileWriter<File> = SerializedFileWriter::new(file, schema_ptr, props).map_err(|e| e.to_string())?;

        // Write Data
        let mut row_group_writer: SerializedRowGroupWriter<_> = writer.next_row_group().map_err(|e| e.to_string())?;

        for (_, col_values) in columns_data {
            let ser_writer: Option<SerializedColumnWriter> = row_group_writer.next_column().map_err(|e| e.to_string())?;
            if let Some(mut col_writer) = ser_writer {
                match col_values {
                    ColumnData::Utf8(col_values) => {
                        if let ColumnWriter::ByteArrayColumnWriter(ref mut typed_writer) = col_writer.untyped() {
                            let mut values: Vec<ByteArray> = Vec::with_capacity(col_values.len());
                            let mut def_levels: Vec<i16> = Vec::with_capacity(col_values.len());
                            for val in col_values {
                                if let Some(s) = val {
                                    values.push(ByteArray::from(s.as_str()));
                                    def_levels.push(1);
                                } else {
                                    def_levels.push(0);
                                }
                            }
                            typed_writer.write_batch(&values, Some(&def_levels), None).map_err(|e| e.to_string())?;
                        } else {
                            return Err("Unexpected Parquet writer type for UTF8 column".into());
                        }
                    }
                    ColumnData::Int32(col_values) => {
                        if let ColumnWriter::Int32ColumnWriter(ref mut typed_writer) = col_writer.untyped() {
                            let mut values: Vec<i32> = Vec::new();
                            let mut def_levels: Vec<i16> = Vec::with_capacity(col_values.len());
                            for val in col_values {
                                if let Some(v) = val {
                                    values.push(v);
                                    def_levels.push(1);
                                } else {
                                    def_levels.push(0);
                                }
                            }
                            typed_writer.write_batch(&values, Some(&def_levels), None).map_err(|e| e.to_string())?;
                        } else {
                            return Err("Unexpected Parquet writer type for INT32 column".into());
                        }
                    }
                    ColumnData::Int64(col_values) => {
                        if let ColumnWriter::Int64ColumnWriter(ref mut typed_writer) = col_writer.untyped() {
                            let mut values: Vec<i64> = Vec::new();
                            let mut def_levels: Vec<i16> = Vec::with_capacity(col_values.len());
                            for val in col_values {
                                if let Some(v) = val {
                                    values.push(v);
                                    def_levels.push(1);
                                } else {
                                    def_levels.push(0);
                                }
                            }
                            typed_writer.write_batch(&values, Some(&def_levels), None).map_err(|e| e.to_string())?;
                        } else {
                            return Err("Unexpected Parquet writer type for INT64 column".into());
                        }
                    }
                    ColumnData::Double(col_values) => {
                        if let ColumnWriter::DoubleColumnWriter(ref mut typed_writer) = col_writer.untyped() {
                            let mut values: Vec<f64> = Vec::new();
                            let mut def_levels: Vec<i16> = Vec::with_capacity(col_values.len());
                            for val in col_values {
                                if let Some(v) = val {
                                    values.push(v);
                                    def_levels.push(1);
                                } else {
                                    def_levels.push(0);
                                }
                            }
                            typed_writer.write_batch(&values, Some(&def_levels), None).map_err(|e| e.to_string())?;
                        } else {
                            return Err("Unexpected Parquet writer type for DOUBLE column".into());
                        }
                    }
                }

                col_writer.close().expect("Unable to close column writer.");
            }
        }

        row_group_writer.close().expect("Unable to close row group writer.");
        writer.close().expect("Unable to close writer.");

        Ok(())
    }


    fn read_qvd_to_buf(mut f: File, binary_section_offset: usize) -> Vec<u8> {
        f.seek(SeekFrom::Start(binary_section_offset as u64))
            .unwrap();
        let mut buf: Vec<u8> = Vec::new();
        f.read_to_end(&mut buf).unwrap();
        buf
    }

    fn match_symbols_with_indexes(symbols: &[Option<String>], pointers: &[i64]) -> Vec<Option<String>> {
        let mut cols: Vec<Option<String>> = Vec::new();
        for pointer in pointers.iter() {
            if symbols.is_empty() || *pointer < 0 {
                cols.push(None);
            } else {
                cols.push(symbols[*pointer as usize].clone());
            }
        }
        cols
    }

    fn match_values_with_indexes(symbols: &[SymbolValue], pointers: &[i64]) -> Vec<Option<SymbolValue>> {
        let mut cols: Vec<Option<SymbolValue>> = Vec::new();
        for pointer in pointers.iter() {
            if symbols.is_empty() || *pointer < 0 {
                cols.push(None);
            } else {
                let idx = *pointer as usize;
                if idx < symbols.len() {
                    cols.push(Some(symbols[idx].clone()));
                } else {
                    cols.push(None);
                }
            }
        }
        cols
    }

    fn get_symbols_as_strings(buf: &[u8], field: &QvdFieldHeader) -> Vec<Option<String>> {
        let start = field.offset;
        let end = start + field.length;
        let mut string_start: usize = 0;
        let mut strings: Vec<Option<String>> = Vec::new();

        let mut i = start;
        while i < end {
            let byte = &buf[i];
            // Check first byte of symbol. This is not part of the symbol but tells us what type of data to read.
            match byte {
                0 => {
                    // Strings are null terminated
                    // Read bytes from start fo string (string_start) up to current byte.
                    let utf8_bytes = buf[string_start..i].to_vec().to_owned();
                    let value = String::from_utf8(utf8_bytes).unwrap_or_else(|_| {
                        panic!(
                        "Error parsing string value in field: {}, field offset: {}, byte offset: {}",
                        field.field_name, start, i
                    )
                    });
                    strings.push(Some(value));
                    i += 1;
                }
                1 => {
                    // 4 byte integer
                    let byte_array: [u8; 4] = [buf[i + 1], buf[i + 2], buf[i + 3], buf[i + 4]];
                    let numeric_value = i32::from_le_bytes(byte_array);
                    strings.push(Some(numeric_value.to_string()));
                    i += 5;
                }
                2 => {
                    // 4 byte double
                    let byte_array: [u8; 8] = [
                        buf[i + 1], buf[i + 2], buf[i + 3], buf[i + 4],
                        buf[i + 5], buf[i + 6], buf[i + 7], buf[i + 8]
                    ];
                    let numeric_value = f64::from_le_bytes(byte_array);
                    strings.push(Some(numeric_value.to_string()));
                    i += 9;
                }
                4 => {
                    // Beginning of a null terminated string type
                    // Mark where string value starts, excluding preceding byte 0x04
                    i += 1;
                    string_start = i;
                }
                5 => {
                    // 4 bytes of unknown followed by null terminated string
                    // Skip the 4 bytes before string
                    i += 5;
                    string_start = i;
                }
                6 => {
                    // 8 bytes of unknown followed by null terminated string
                    // Skip the 8 bytes before string
                    i += 9;
                    string_start = i;
                }
                _ => {
                    // Part of a string, do nothing until null terminator
                    i += 1;
                }
            }
        }
        strings
    }

    fn get_symbols_as_values(buf: &[u8], field: &QvdFieldHeader) -> Vec<SymbolValue> {
        let start = field.offset;
        let end = start + field.length;
        let mut string_start: usize = 0;
        let mut values: Vec<SymbolValue> = Vec::new();

        let mut i = start;
        while i < end {
            let byte = buf[i];
            match byte {
                0 => {
                    // String terminator: collect the string that started after 0x04/0x05/0x06 marker
                    let utf8_bytes = buf[string_start..i].to_vec();
                    let value = String::from_utf8(utf8_bytes).unwrap_or_else(|_| {
                        panic!(
                            "Error parsing string value in field: {}, field offset: {}, byte offset: {}",
                            field.field_name, start, i
                        )
                    });
                    values.push(SymbolValue::Str(value));
                    i += 1;
                }
                1 => {
                    // 4 byte integer (little endian)
                    if i + 4 >= buf.len() { break; }
                    let byte_array: [u8; 4] = [buf[i + 1], buf[i + 2], buf[i + 3], buf[i + 4]];
                    let numeric_value = i32::from_le_bytes(byte_array);
                    values.push(SymbolValue::Int(numeric_value));
                    i += 5;
                }
                2 => {
                    // 8 byte double (little endian)
                    if i + 8 >= buf.len() { break; }
                    let byte_array: [u8; 8] = [
                        buf[i + 1], buf[i + 2], buf[i + 3], buf[i + 4],
                        buf[i + 5], buf[i + 6], buf[i + 7], buf[i + 8]
                    ];
                    let numeric_value = f64::from_le_bytes(byte_array);
                    values.push(SymbolValue::Double(numeric_value));
                    i += 9;
                }
                4 => {
                    // Start of a null terminated string value
                    i += 1;
                    string_start = i;
                }
                5 => {
                    // 4 bytes of unknown followed by null terminated string
                    i += 5;
                    string_start = i;
                }
                6 => {
                    // 8 bytes of unknown followed by null terminated string
                    i += 9;
                    string_start = i;
                }
                _ => {
                    // Part of a string payload; advance until we hit null terminator
                    i += 1;
                }
            }
        }
        values
    }

    // Retrieve bit stuffed data. Each row has index to value from symbol map.
    fn get_row_indexes(buf: &[u8], field: &QvdFieldHeader, record_byte_size: usize) -> Vec<i64> {
        let mut cloned_buf = buf.to_owned();
        let chunks = cloned_buf.chunks_mut(record_byte_size);
        let mut indexes: Vec<i64> = Vec::new();
        for chunk in chunks {
            // Reverse the bytes in the record
            chunk.reverse();
            let bits = BitSlice::<_, Msb0>::from_slice(&chunk[..]);
            let start = bits.len() - field.bit_offset;
            let end = bits.len() - field.bit_offset - field.bit_width;
            let binary = bitslice_to_vec(&bits[end..start]);
            let index = binary_to_u32(binary);
            indexes.push((index as i32 + field.bias) as i64);
        }
        indexes
    }

    // Slow
    fn binary_to_u32(binary: Vec<u8>) -> u32 {
        let mut sum: u32 = 0;
        for bit in binary {
            sum <<= 1;
            sum += bit as u32;
        }
        sum
    }

    // Slow
    fn bitslice_to_vec(bitslice: &BitSlice<u8, Msb0>) -> Vec<u8> {
        let mut v: Vec<u8> = Vec::new();
        for bit in bitslice {
            let val = match *bit {
                true => 1,
                false => 0
            };
            v.push(val);
        }
        v
    }

    fn get_xml_data(file_name: &str) -> Result<String, io::Error> {
        match read_file(file_name) {
            Ok(mut reader) => {
                let mut buffer = Vec::new();
                // There is a line break, carriage return and a null terminator between the XMl and data
                // Find the null terminator
                reader
                    .read_until(0, &mut buffer)
                    .expect("Failed to read file");
                let xml_string =
                    str::from_utf8(&buffer[..]).expect("xml section contains invalid UTF-8 chars");
                Ok(xml_string.to_owned())
            }
            Err(e) => Err(e),
        }
    }

    fn read_file<P>(filename: P) -> io::Result<io::BufReader<File>>
    where
        P: AsRef<Path>,
    {
        let file = File::open(filename)?;
        Ok(io::BufReader::new(file))
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use parquet::file::reader::{SerializedFileReader, FileReader};
        use parquet::basic::Type as PhyType;

        #[test]
        fn test_double() {
            let buf: Vec<u8> = vec![
                0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x7a, 0x40, 0x02, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x50, 0x7a, 0x40,
            ];
            let field = QvdFieldHeader {
                length: buf.len(),
                offset: 0,
                field_name: String::new(),
                bias: 0,
                bit_offset: 0,
                bit_width: 0,
            };
            let res = get_symbols_as_strings(&buf, &field);
            let expected: Vec<Option<String>> = vec![Some(420.0.to_string()), Some(421.0.to_string())];
            assert_eq!(expected, res);
        }

        #[test]
        fn test_int() {
            let buf: Vec<u8> = vec![0x01, 0x0A, 0x00, 0x00, 0x00, 0x01, 0x14, 0x00, 0x00, 0x00];
            let field = QvdFieldHeader {
                length: buf.len(),
                offset: 0,
                field_name: String::new(),
                bias: 0,
                bit_offset: 0,
                bit_width: 0,
            };
            let res = get_symbols_as_strings(&buf, &field);
            let expected = vec![Some(10.0.to_string()), Some(20.0.to_string())];
            assert_eq!(expected, res);
        }

        #[test]
        #[rustfmt::skip]
        fn test_mixed_numbers() {
            let buf: Vec<u8> = vec![
                0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x7a, 0x40,
                0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x50, 0x7a, 0x40,
                0x01, 0x01, 0x00, 0x00, 0x00,
                0x01, 0x02, 0x00, 0x00, 0x00,
                0x05, 0x00, 0x00, 0x00, 0x00, 0x37, 0x30, 0x30, 0x30, 0x00,
                0x06, 0x00,0x00,0x00, 0x00,0x00,0x00,0x00,0x00, 0x38, 0x36, 0x35, 0x2e, 0x32, 0x00
            ];
            let field = QvdFieldHeader {
                length: buf.len(),
                offset: 0,
                field_name: String::new(),
                bias: 0,
                bit_offset: 0,
                bit_width: 0,
            };
            let res = get_symbols_as_strings(&buf, &field);
            let expected: Vec<Option<String>> = vec![
                Some(420.to_string()),
                Some(421.to_string()),
                Some(1.to_string()),
                Some(2.to_string()),
                Some(7000.to_string()),
                Some(865.2.to_string())
            ];
            assert_eq!(expected, res);
        }

        #[test]
        fn test_string() {
            let buf: Vec<u8> = vec![
                4, 101, 120, 97, 109, 112, 108, 101, 32, 116, 101, 120, 116, 0, 4, 114, 117, 115, 116,
                0,
            ];
            let field = QvdFieldHeader {
                length: buf.len(),
                offset: 0,
                field_name: String::new(),
                bias: 0,
                bit_offset: 0,
                bit_width: 0,
            };
            let res = get_symbols_as_strings(&buf, &field);
            let expected = vec![Some("example text".into()), Some("rust".into())];
            assert_eq!(expected, res);
        }

        #[test]
        #[rustfmt::skip]
        fn test_utf8_string() {
            let buf: Vec<u8> = vec![
                0x04, 0xE4, 0xB9, 0x9F, 0xE6, 0x9C, 0x89, 0xE4, 0xB8, 0xAD, 0xE6, 0x96, 0x87, 0xE7,
                0xAE, 0x80, 0xE4, 0xBD, 0x93, 0xE5, 0xAD, 0x97, 0x00,
                0x04, 0xF0, 0x9F, 0x90, 0x8D, 0xF0, 0x9F, 0xA6, 0x80, 0x00,
            ];

            let field = QvdFieldHeader {
                length: buf.len(),
                offset: 0,
                field_name: String::new(),
                bias: 0,
                bit_offset: 0,
                bit_width: 0,
            };
            let res = get_symbols_as_strings(&buf, &field);
            let expected = vec![Some("也有中文简体字".into()), Some("🐍🦀".into())];
            assert_eq!(expected, res);
        }

        #[test]
        fn test_mixed_string() {
            let buf: Vec<u8> = vec![
                4, 101, 120, 97, 109, 112, 108, 101, 32, 116, 101, 120, 116, 0, 4, 114, 117, 115, 116,
                0, 5, 42, 65, 80, 1, 49, 50, 51, 52, 0, 6, 1, 1, 1, 1, 1, 1, 1, 1, 100, 111, 117, 98,
                108, 101, 0,
            ];
            let field = QvdFieldHeader {
                length: buf.len(),
                offset: 0,
                field_name: String::new(),
                bias: 0,
                bit_offset: 0,
                bit_width: 0,
            };
            let res = get_symbols_as_strings(&buf, &field);
            let expected = vec![
                Some("example text".into()),
                Some("rust".into()),
                Some("1234".into()),
                Some("double".into()),
            ];
            assert_eq!(expected, res);
        }

        #[test]
        fn test_bitslice_to_vec() {
            let mut x: Vec<u8> = vec![
                0x00, 0x00, 0x00, 0x11, 0x01, 0x22, 0x02, 0x33, 0x13, 0x34, 0x14, 0x35,
            ];
            let bits = BitSlice::<_, Msb0>::from_slice(&mut x[..]);
            let target = &bits[27..32];
            let binary_vec = bitslice_to_vec(&target);

            let mut sum: u32 = 0;
            for bit in binary_vec {
                sum <<= 1;
                sum += bit as u32;
            }
            assert_eq!(17, sum);
        }

        #[test]
        fn test_get_row_indexes() {
            let buf: Vec<u8> = vec![
                0x00, 0x14, 0x00, 0x11, 0x01, 0x22, 0x02, 0x33, 0x13, 0x34, 0x24, 0x35,
            ];
            let field = QvdFieldHeader {
                field_name: String::from("name"),
                offset: 0,
                length: 0,
                bit_offset: 10,
                bit_width: 3,
                bias: 0,
            };
            let record_byte_size = buf.len();
            let res = get_row_indexes(&buf, &field, record_byte_size);
            let expected: Vec<i64> = vec![5];
            assert_eq!(expected, res);
        }

        // New tests for typed symbol extraction and matching

        #[test]
        fn test_get_symbols_as_values_double() {
            let buf: Vec<u8> = vec![
                0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x7a, 0x40,
                0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x50, 0x7a, 0x40,
            ];
            let field = QvdFieldHeader {
                length: buf.len(),
                offset: 0,
                field_name: String::new(),
                bias: 0,
                bit_offset: 0,
                bit_width: 0,
            };
            let res = get_symbols_as_values(&buf, &field);
            assert_eq!(res.len(), 2);
            match &res[0] { SymbolValue::Double(v) => assert_eq!(*v, 420.0), _ => panic!("expected double") }
            match &res[1] { SymbolValue::Double(v) => assert_eq!(*v, 421.0), _ => panic!("expected double") }
        }

        #[test]
        fn test_get_symbols_as_values_int() {
            let buf: Vec<u8> = vec![
                0x01, 0x0A, 0x00, 0x00, 0x00,
                0x01, 0x14, 0x00, 0x00, 0x00,
            ];
            let field = QvdFieldHeader {
                length: buf.len(),
                offset: 0,
                field_name: String::new(),
                bias: 0,
                bit_offset: 0,
                bit_width: 0,
            };
            let res = get_symbols_as_values(&buf, &field);
            assert_eq!(res.len(), 2);
            match &res[0] { SymbolValue::Int(v) => assert_eq!(*v, 10), _ => panic!("expected int") }
            match &res[1] { SymbolValue::Int(v) => assert_eq!(*v, 20), _ => panic!("expected int") }
        }

        #[test]
        #[rustfmt::skip]
        fn test_get_symbols_as_values_mixed() {
            let buf: Vec<u8> = vec![
                0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x7a, 0x40, // 420.0
                0x01, 0x2A, 0x00, 0x00, 0x00,                         // 42
                0x04, b'7', b'0', b'0', b'0', 0x00,                  // "7000"
            ];
            let field = QvdFieldHeader {
                length: buf.len(),
                offset: 0,
                field_name: String::new(),
                bias: 0,
                bit_offset: 0,
                bit_width: 0,
            };
            let res = get_symbols_as_values(&buf, &field);
            assert_eq!(res.len(), 3);
            match &res[0] { SymbolValue::Double(v) => assert_eq!(*v, 420.0), _ => panic!("expected double") }
            match &res[1] { SymbolValue::Int(v) => assert_eq!(*v, 42), _ => panic!("expected int") }
            match &res[2] { SymbolValue::Str(s) => assert_eq!(s, "7000"), _ => panic!("expected string") }
        }

        #[test]
        fn test_match_values_with_indexes_typed() {
            let symbols = vec![
                SymbolValue::Int(7),
                SymbolValue::Double(3.14),
                SymbolValue::Str("x".to_string()),
            ];
            let pointers: Vec<i64> = vec![0, -1, 2, 99];
            let res = match_values_with_indexes(&symbols, &pointers);
            assert_eq!(res.len(), 4);
            match &res[0] { Some(SymbolValue::Int(v)) => assert_eq!(*v, 7), _ => panic!("expected Some(Int(7))") }
            assert!(res[1].is_none());
            match &res[2] { Some(SymbolValue::Str(s)) => assert_eq!(s, "x"), _ => panic!("expected Some(Str)\n") }
            assert!(res[3].is_none());
        }

        #[test]
        fn test_transform_qvd_to_parquet_writes_typed_schema() {
            // Build synthetic QVD file with 2 fields and 2 records.
            // Field a: INT32 symbols [10, 20]
            // Field b: DOUBLE symbols [1.5, 2.5]
            // Rows section encodes pointers 0 and 1 for both fields using bit_width=1 at bit_offset=0

            // Build binary symbol sections
            let mut sym_a: Vec<u8> = Vec::new();
            sym_a.push(0x01); sym_a.extend_from_slice(&10i32.to_le_bytes());
            sym_a.push(0x01); sym_a.extend_from_slice(&20i32.to_le_bytes());
            let mut sym_b: Vec<u8> = Vec::new();
            sym_b.push(0x02); sym_b.extend_from_slice(&1.5f64.to_le_bytes());
            sym_b.push(0x02); sym_b.extend_from_slice(&2.5f64.to_le_bytes());

            let off_a = 0usize;
            let len_a = sym_a.len();
            let off_b = len_a;
            let len_b = sym_b.len();
            let rows_off = off_b + len_b;

            let rows: Vec<u8> = vec![0x00, 0x01]; // two records, LSB gives 0 then 1
            let record_byte_size = 1usize;

            // Construct XML header
            let xml = format!(
                concat!(
                    "<QvdTableHeader>",
                    "<TableName>T</TableName>",
                    "<CreatorDoc>D</CreatorDoc>",
                    "<Fields>",
                    "  <QvdFieldHeader>",
                    "    <FieldName>a</FieldName>",
                    "    <Offset>{off_a}</Offset>",
                    "    <Length>{len_a}</Length>",
                    "    <BitOffset>0</BitOffset>",
                    "    <BitWidth>1</BitWidth>",
                    "    <Bias>0</Bias>",
                    "  </QvdFieldHeader>",
                    "  <QvdFieldHeader>",
                    "    <FieldName>b</FieldName>",
                    "    <Offset>{off_b}</Offset>",
                    "    <Length>{len_b}</Length>",
                    "    <BitOffset>0</BitOffset>",
                    "    <BitWidth>1</BitWidth>",
                    "    <Bias>0</Bias>",
                    "  </QvdFieldHeader>",
                    "</Fields>",
                    "<NoOfRecords>2</NoOfRecords>",
                    "<RecordByteSize>{record_byte_size}</RecordByteSize>",
                    "<Offset>{rows_off}</Offset>",
                    "<Length>{total_len}</Length>",
                    "</QvdTableHeader>"
                ),
                off_a = off_a,
                len_a = len_a,
                off_b = off_b,
                len_b = len_b,
                record_byte_size = record_byte_size,
                rows_off = rows_off,
                total_len = len_a + len_b + rows.len(),
            );

            // Write to temporary source file
            let mut src_path = std::env::temp_dir();
            src_path.push(format!("qvd_test_{}.qvd", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
            let mut dst_path = std::env::temp_dir();
            dst_path.push(format!("qvd_test_{}.parquet", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));

            let mut f = std::fs::File::create(&src_path).unwrap();
            f.write_all(xml.as_bytes()).unwrap();
            f.write_all(&[0u8]).unwrap(); // null terminator
            f.write_all(&sym_a).unwrap();
            f.write_all(&sym_b).unwrap();
            f.write_all(&rows).unwrap();
            drop(f);

            // Run transform
            let src = src_path.to_string_lossy().to_string();
            let dst = dst_path.to_string_lossy().to_string();


            _transform_qvd_to_parquet(src.clone(), dst.clone()).expect("Transformation should succeed");

            // Read parquet and assert schema physical types
            let pf = std::fs::File::open(&dst_path).unwrap();
            let reader = SerializedFileReader::new(pf).unwrap();
            let schema_descr = reader.metadata().file_metadata().schema_descr();
            assert_eq!(schema_descr.num_columns(), 2);
            assert_eq!(schema_descr.column(0).physical_type(), PhyType::INT32);
            assert_eq!(schema_descr.column(1).physical_type(), PhyType::DOUBLE);

            // Cleanup
            let _ = std::fs::remove_file(src_path);
            let _ = std::fs::remove_file(dst_path);
        }
    }
}
