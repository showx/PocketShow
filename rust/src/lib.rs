pub mod duml;
pub mod follow;
pub mod frame;
pub mod gimbal;
pub mod session;

#[cfg(feature = "extension-module")]
mod python_api;

#[cfg(feature = "extension-module")]
#[pyo3::pymodule]
fn _core(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    python_api::register(m)
}
