fn main() {
    // macOS 扩展不链接 libpython，符号在加载进解释器时再解析。
    pyo3_build_config::add_extension_module_link_args();
}
