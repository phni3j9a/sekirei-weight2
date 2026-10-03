//! Compile against the actual dedicated white-view release USI core rlib.
//! Execution is a separate Root-controlled proof; this builder only compiles.
//! INPUT must be a full-byte validated 03 transform, not a retagged 01 file.
use sekirei_core::nnue;
use std::{env, fs, path::Path};

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    })
}

fn main() {
    let args: Vec<_> = env::args().collect();
    assert_eq!(args.len(), 4, "usage: serialization-probe INPUT NEW_OUTPUT EXPECTED_FNV16");
    assert!(args[3].len() == 16 && args[3].bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
    let expected = u64::from_str_radix(&args[3], 16).expect("externally bound FNV16");
    let input = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    assert!(!output.exists() && output.parent().unwrap().is_dir());
    let original = fs::read(input).expect("read separately validated 03 input");
    assert_eq!(&original[..8], b"SEKIRW03", "dedicated native ABI");
    assert_eq!(fnv(&original), expected, "external input FNV differs");
    let loaded = nnue::read_weights(input).expect("actual white-view core read_weights/tie guard");
    nnue::save_weights(&loaded, output).expect("actual white-view core save_weights/tie guard");
    let canonical = fs::read(output).expect("read canonical bytes");
    assert_eq!(original, canonical, "actual loaded/saved bytes differ");
    println!("{{\"input_bytes\":{},\"canonical_bytes\":{},\"all_bytes_equal\":true,\"native_magic\":\"SEKIRW03\",\"input_fnv1a\":\"{:016x}\",\"canonical_fnv1a\":\"{:016x}\"}}", original.len(), canonical.len(), fnv(&original), fnv(&canonical));
}
