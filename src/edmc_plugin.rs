use pyo3::{types::PyModule, PyResult, Python};

// #[pyfunction]
// fn load()

pub fn init(py: Python<'_>, _m: &PyModule) -> PyResult<()> {
    let builtins = PyModule::import(py, "builtins")?;
    let _total: i32 =
        builtins.getattr("asdf")?.call1((vec![1, 2, 3],))?.extract()?;
    Ok(())
}
