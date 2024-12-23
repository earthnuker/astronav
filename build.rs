fn main() -> shadow_rs::SdResult<()> {
    shadow_rs::new()?;
    #[cfg(feature = "pyo3")]
    {
        pyo3_build_config::add_extension_module_link_args();
        pyo3_build_config::use_pyo3_cfgs();
    }
    Ok(())
}
