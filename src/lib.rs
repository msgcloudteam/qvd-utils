pub mod qvd_structure;
mod transform;

#[pyo3::pymodule]
mod qvd {
    use pyo3::prelude::*;
    use crate::transform::qvd_to_parquet;

    #[pyfunction]
    fn transform_qvd_to_parquet(_py: Python,
                                source_file_name: String,
                                target_file_name: String) -> PyResult<()> {
        qvd_to_parquet(source_file_name, target_file_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }
    
}
