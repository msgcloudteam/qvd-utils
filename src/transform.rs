use std::fs::File;
use std::io;
use std::io::{BufRead, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use bitvec::order::Msb0;
use bitvec::prelude::BitSlice;
use parquet::basic::{ConvertedType, Repetition, Type as PhysicalType};
use parquet::column::writer::ColumnWriter;
use parquet::data_type::ByteArray;
use parquet::file::properties::WriterProperties;
use parquet::file::writer::{SerializedColumnWriter, SerializedFileWriter, SerializedRowGroupWriter};
use parquet::schema::types::Type;
use quick_xml::de::from_str;
use crate::qvd_structure::{QvdFieldHeader, QvdTableHeader};

#[derive(Clone, Debug)]
enum SymbolValue {
    Str(usize, usize),
    Int(i32),
    Double(f64),
}

impl SymbolValue {

    fn get_byte_array(&self, buf: &[u8]) -> ByteArray {
        match self {
            SymbolValue::Str(start, end) => {
                let utf8_bytes = &buf[*start..*end];
                utf8_bytes.into()
            }
            SymbolValue::Int(i) => { i.to_string().as_bytes().into() }
            SymbolValue::Double(d) => { d.to_string().as_bytes().into() }
        }
    }

    fn get_int(&self, buf: &[u8]) -> i32 {
        match self {
            SymbolValue::Str(start, end) => {
                let utf8_bytes = String::from_utf8_lossy(&buf[*start..*end]).into_owned();
                utf8_bytes.parse::<i32>().unwrap_or(0)
            }
            SymbolValue::Int(i) => { *i }
            SymbolValue::Double(d) => { d.round() as i32 }
        }
    }

    fn get_double(&self, buf: &[u8]) -> f64 {
        match self {
            SymbolValue::Str(start, end) => {
                let utf8_bytes = String::from_utf8_lossy(&buf[*start..*end]).into_owned();
                utf8_bytes.parse::<f64>().unwrap_or(0.0)
            }
            SymbolValue::Int(i) => { *i as f64 }
            SymbolValue::Double(d) => { *d }
        }
    }
}

pub fn qvd_to_parquet(source_file_name: String,
                                target_file_name: String) -> Result<(), String>{
    let xml: String = get_xml_data(&source_file_name).map_err(|e| e.to_string())?;
    let binary_section_offset = xml.as_bytes().len();
    let qvd_structure: QvdTableHeader = from_str(&xml).map_err(|e| e.to_string())?;

    let mut columns_data: Vec<(String, FieldValueIterator)> = Vec::new();

    let f = File::open(&source_file_name).expect("Unable to open file");
    let buf = read_qvd_to_buf(f, binary_section_offset);
    let record_byte_size = qvd_structure.record_byte_size;
    let rows_start = qvd_structure.offset;

    for field in qvd_structure.fields.headers {
        let symbols_typed = get_symbols_as_values(&buf, &field);
        let field_name = field.field_name.clone();
        let iterator = FieldValueIterator::new(field, rows_start, record_byte_size, symbols_typed);

        columns_data.push((field_name, iterator));
    }
    
    // Create Parquet Schema
    let mut fields: Vec<Arc<Type>> = Vec::new();
    for (name, col) in &columns_data {
        let field_type = match col.probe_value() {
            SymbolValue::Str(_, _) => Type::primitive_type_builder(name, PhysicalType::BYTE_ARRAY)
                .with_repetition(Repetition::OPTIONAL)
                .with_converted_type(ConvertedType::UTF8)
                .build()
                .map_err(|e| e.to_string())?,
            SymbolValue::Int(_) => Type::primitive_type_builder(name, PhysicalType::INT32)
                .with_repetition(Repetition::OPTIONAL)
                .build()
                .map_err(|e| e.to_string())?,
            SymbolValue::Double(_) => Type::primitive_type_builder(name, PhysicalType::DOUBLE)
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

    for (_, mut col_values) in columns_data {
        let ser_writer: Option<SerializedColumnWriter> = row_group_writer.next_column().map_err(|e| e.to_string())?;
        if let Some(mut col_writer) = ser_writer {
            match col_values.probe_value() {
                SymbolValue::Str(_, _) =>  {
                    if let ColumnWriter::ByteArrayColumnWriter(typed_writer) = col_writer.untyped() {
                        while col_values.has_next(&buf) {
                            let (def_levels, values) = col_values.take_byte_array(&buf, 1000);
                            typed_writer.write_batch(&values, Some(&def_levels), None).map_err(|e| e.to_string())?;
                        }
                    } else {
                        return Err("Unexpected Parquet writer type for UTF8 column".into());
                    }
                }
                SymbolValue::Int(_) =>  {
                    if let ColumnWriter::Int32ColumnWriter(typed_writer) = col_writer.untyped() {
                        while col_values.has_next(&buf) {
                            let (def_levels, values) = col_values.take_int(&buf, 1000);
                            typed_writer.write_batch(&values, Some(&def_levels), None).map_err(|e| e.to_string())?;
                        }
                    } else {
                        return Err("Unexpected Parquet writer type for INT32 column".into());
                    }
                }
                SymbolValue::Double(_)=> {
                    if let ColumnWriter::DoubleColumnWriter(typed_writer) = col_writer.untyped() {
                        while col_values.has_next(&buf) {
                            let (def_levels, values) = col_values.take_double(&buf, 1000);
                            typed_writer.write_batch(&values, Some(&def_levels), None).map_err(|e| e.to_string())?;
                        }
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


struct FieldValueIterator {
    i : usize,
    field: QvdFieldHeader,
    record_byte_size: usize,
    values: Vec<SymbolValue>
}

impl FieldValueIterator {

    fn new(field: QvdFieldHeader,
           rows_start: usize,
           record_byte_size: usize,
           values: Vec<SymbolValue>) -> Self {
        FieldValueIterator {
            i: rows_start,
            field,
            record_byte_size,
            values
        }
    }

    fn probe_value(&self) -> SymbolValue {
        match self.values.get(0) {
            Some(value) => value.clone(),
            None => SymbolValue::Int(0)
        }
    }

    /// Check with has_next() before calling this method
    fn next_index(&mut self, buf: &[u8]) -> i64 {
        let mut chunk = buf[self.i..(self.i + self.record_byte_size)].to_vec();

        // Reverse the bytes in the record
        chunk.reverse();

        let bits = BitSlice::<_, Msb0>::from_slice(&chunk[..]);
        let start = bits.len() - self.field.bit_offset;
        let end = bits.len() - self.field.bit_offset - self.field.bit_width;

        let binary = bitslice_to_vec(&bits[end..start]);
        let index = (binary_to_u32(binary) as i32 + self.field.bias) as i64;

        self.i += self.record_byte_size;

        index
    }

    fn take_indices(&mut self, buf: &[u8], num: usize) -> Vec<i64> {
        let mut indices: Vec<i64> = Vec::with_capacity(num);
        while self.has_next(buf) && indices.len() < num {
            indices.push(self.next_index(buf));
        }
        indices
    }

    fn take_byte_array(&mut self, buf: &[u8], num: usize) -> (Vec<i16>, Vec<ByteArray>) {
        let indices = self.take_indices(buf, num);
        let mut values: Vec<ByteArray> = Vec::with_capacity(num);
        let mut def_levels: Vec<i16> = Vec::with_capacity(num);

        for index in indices {
            match self.values.get(index as usize) {
                Some(s) => {
                    values.push(s.get_byte_array(buf));
                    def_levels.push(1);
                },
                None => {
                    def_levels.push(0);
                }
            }
        }
        (def_levels, values)
    }

    fn take_int(&mut self, buf: &[u8], num: usize) -> (Vec<i16>, Vec<i32>)  {
        let indices = self.take_indices(buf, num);
        let mut values: Vec<i32> = Vec::with_capacity(num);
        let mut def_levels: Vec<i16> = Vec::with_capacity(num);

        for index in indices {
            match self.values.get(index as usize) {
                Some(s) => {
                    values.push(s.get_int(buf));
                    def_levels.push(1);
                },
                None => {
                    def_levels.push(0);
                }
            }
        }
        (def_levels, values)
    }

    fn take_double(&mut self, buf: &[u8], num: usize) -> (Vec<i16>, Vec<f64>)  {
        let indices = self.take_indices(buf, num);
        let mut values: Vec<f64> = Vec::with_capacity(num);
        let mut def_levels: Vec<i16> = Vec::with_capacity(num);

        for index in indices {
            match self.values.get(index as usize) {
                Some(s) => {
                    values.push(s.get_double(buf));
                    def_levels.push(1);
                },
                None => {
                    def_levels.push(0);
                }
            }
        }
        (def_levels, values)
    }

    fn has_next(&self, buf: &[u8]) -> bool {
        self.i + self.record_byte_size <= buf.len()
    }

}

fn get_symbols_as_values(buf: &[u8], field: &QvdFieldHeader) -> Vec<SymbolValue> {
    let start = field.offset;
    let end = start + field.length;
    let mut string_start: usize = start;
    let mut values: Vec<SymbolValue> = Vec::new();

    let mut i = start;
    while i < end {
        let byte = buf[i];
        match byte {
            0 => {
                // String terminator: collect the string that started after 0x04/0x05/0x06 marker
                
                values.push(SymbolValue::Str(string_start, i));
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


fn read_qvd_to_buf(mut f: File, binary_section_offset: usize) -> Vec<u8> {
    f.seek(SeekFrom::Start(binary_section_offset as u64))
        .unwrap();
    let mut buf: Vec<u8> = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    buf
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
