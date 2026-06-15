use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let vendor = manifest_dir.join("vendor");

    println!(
        "cargo:rerun-if-changed={}",
        vendor.join("gnubg_bridge.c").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        vendor.join("gnubg.weights").display()
    );
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");

    let mut build = cc::Build::new();
    build
        .file(vendor.join("gnubg_bridge.c"))
        .include(&vendor)
        .define("HAVE_CONFIG_H", None)
        .define("WITHOUT_GTK", None)
        .define("_FILE_OFFSET_BITS", "64")
        .flag_if_supported("-O3")
        .flag_if_supported("-march=x86-64-v3")
        .flag_if_supported("-mavx2")
        .flag_if_supported("-mfma")
        .warnings(false);

    build.compile("gnubg_bridge");
}
