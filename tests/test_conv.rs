use std::time::Instant;

use rand::Rng;
use num_complex::Complex32;

use convolution_dsp::{ConvMode, Conv1dPlanner};
use convolution_dsp::file::*;

#[test]
fn test_dirac() {
    let filter = vec![0., 0., 0., 1., 0., 0., 0.];

    let planner = Conv1dPlanner::new();
    let mut conv = planner.plan_conv1d(&filter, ConvMode::Full);

    let signal = vec![
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.,
    ];
    let signal: Vec<_> = signal
        .into_iter()
        .map(|re| Complex32::new(re, 0.))
        .collect();

    let actual = conv.process(signal);

    assert_eq!(actual.len(), 35);

    for i in 0..35 {
        if i != 17 {
            assert!(actual[i].re.abs() < 0.001);
            assert!(actual[i].im.abs() < 0.001);
        } else {
            assert!(actual[i].re.abs() - 1. < 0.001);
            assert!(actual[i].im.abs() < 0.001);
        }
    }
}

fn conv_with_sizes(filter_len: usize, signal_len: usize) {
    let filter: Vec<_> = (0..filter_len).into_iter().map(|_| 1.).collect();

    let planner = Conv1dPlanner::new();
    let mut conv = planner.plan_conv1d(&filter, ConvMode::Full);

    let signal: Vec<_> = (0..signal_len).into_iter().map(|_| Complex32::ONE).collect();

    let now = Instant::now();
    let actual = conv.process(signal);
    println!("Convolution with {} kernel and {} signal took {} ms", filter_len, signal_len, now.elapsed().as_millis());

    assert_eq!(actual.len(), signal_len + filter_len - 1);
}
#[test]
fn test_input_sizes() {
    let mut rng = rand::thread_rng();

    for _ in 0..10000 {
        let filter_len = rng.gen_range(2..256);
        let signal_len = rng.gen_range(2..256);
        conv_with_sizes(filter_len, signal_len);
    }
}

#[test]
fn test_long() {
    let filter = read_numpy_file_f32("data/kernel.bin").unwrap();

    let planner = Conv1dPlanner::new();
    let mut conv = planner.plan_conv1d(&filter, ConvMode::Full);

    let signal = read_numpy_file_c32("data/signal.bin").unwrap();

    let expected = read_numpy_file_c32("data/output.bin").unwrap();
    let now = Instant::now();
    let actual = conv.process(signal);
    println!("Convolution took {} ms", now.elapsed().as_millis());

    assert_eq!(actual.len(), expected.len());
    for i in 0..expected.len() {
        assert!(actual[i].re.abs() - expected[i].re.abs() < 0.001);
        assert!(actual[i].im.abs() - expected[i].im.abs() < 0.001);
    }
}

#[test]
fn test_len_even() {
    let filter = read_numpy_file_f32("data/kernel_even.bin").unwrap();

    let planner = Conv1dPlanner::new();
    let mut conv = planner.plan_conv1d(&filter, ConvMode::Same);

    let signal = read_numpy_file_c32("data/signal_even.bin").unwrap();

    let expected = read_numpy_file_c32("data/output_even.bin").unwrap();
    let now = Instant::now();
    let actual = conv.process(signal);
    println!("Convolution took {} ms", now.elapsed().as_millis());

    assert_eq!(actual.len(), expected.len());
    for i in 0..expected.len() {
        assert!(actual[i].re.abs() - expected[i].re.abs() < 0.001);
        assert!(actual[i].im.abs() - expected[i].im.abs() < 0.001);
    }
}

#[test]
fn test_len_odd() {
    let filter = read_numpy_file_f32("data/kernel_odd.bin").unwrap();

    let planner = Conv1dPlanner::new();
    let mut conv = planner.plan_conv1d(&filter, ConvMode::Same);

    let signal = read_numpy_file_c32("data/signal_odd.bin").unwrap();

    let expected = read_numpy_file_c32("data/output_odd.bin").unwrap();
    let now = Instant::now();
    let actual = conv.process(signal);
    println!("Convolution took {} ms", now.elapsed().as_millis());

    assert_eq!(actual.len(), expected.len());
    for i in 0..expected.len() {
        assert!(actual[i].re.abs() - expected[i].re.abs() < 0.001);
        assert!(actual[i].im.abs() - expected[i].im.abs() < 0.001);
    }
}
