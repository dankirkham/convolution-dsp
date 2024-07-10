use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::Result;

use num_complex::Complex32;

pub fn read_numpy_file_f32(path: &str) -> Result<Vec<f32>> {
    let f = File::open(path)?;
    let mut reader = BufReader::new(f);

    let mut v = Vec::new();
    let mut buf = [0; 4];
    while reader.read_exact(&mut buf).is_ok() {
        let d = f32::from_le_bytes(buf);
        v.push(d);
    }
    Ok(v)
}

pub fn read_numpy_file_c32(path: &str) -> Result<Vec<Complex32>> {
    let f = File::open(path)?;
    let mut reader = BufReader::new(f);

    let mut v = Vec::new();
    let mut re_buf = [0; 4];
    let mut im_buf = [0; 4];
    while reader.read_exact(&mut re_buf).is_ok() {
        reader.read_exact(&mut im_buf)?;
        let re = f32::from_le_bytes(re_buf);
        let im = f32::from_le_bytes(im_buf);
        v.push(Complex32::new(re, im));
    }
    Ok(v)
}
