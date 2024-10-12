use std::time::Instant;

use num_complex::{Complex, Complex32};
use num_traits::identities::{ConstOne, ConstZero};
use rand::Rng;

use convolution_dsp::file::*;
use convolution_dsp::{Conv1dPlanner, ConvMode};

macro_rules! test_dirac_complex_impl {
    ( $type:ident ) => {{
        let mut filter = vec![$type::ZERO; 7];
        filter[3] = $type::ONE;

        let planner = Conv1dPlanner::new();
        let mut conv = planner.plan_conv1d(&filter, ConvMode::Full);

        let mut signal = vec![$type::ZERO; 29];
        signal[14] = $type::ONE;
        let signal: Vec<_> = signal
            .into_iter()
            .map(|re| Complex::<$type>::new(re, $type::ZERO))
            .collect();

        conv.process(signal)
    }};
}

macro_rules! test_dirac_complex_float {
    ( $name:ident, $type:ident ) => {
        #[test]
        fn $name() {
            let actual = test_dirac_complex_impl!($type);
            assert_eq!(actual.len(), 35);

            for i in 0..35 {
                if i != 17 {
                    assert!(actual[i].re.abs() < 0.001);
                    assert!(actual[i].im.abs() < 0.001);
                } else {
                    assert!((actual[i].re - 1.).abs() < 0.001);
                    assert!(actual[i].im.abs() < 0.001);
                }
            }
        }
    };
}
test_dirac_complex_float!(test_dirac_complex_f32, f32);
test_dirac_complex_float!(test_dirac_complex_f64, f64);

// macro_rules! test_dirac_complex_int {
//     ( $name:ident, $type:ident ) => {
//         #[test]
//         fn $name() {
//             let actual = test_dirac_complex_impl!($type);
//             assert_eq!(actual.len(), 35);
//             dbg!(&actual);
// 
//             for i in 0..35 {
//                 if i != 17 {
//                     assert_eq!(actual[i].re, $type::ZERO);
//                     assert_eq!(actual[i].im, $type::ZERO);
//                 } else {
//                     assert_eq!(actual[i].re, $type::ONE);
//                     assert_eq!(actual[i].im, $type::ZERO);
//                 }
//             }
//         }
//     };
// }
// test_dirac_complex_int!(test_dirac_complex_i8, i8);
// test_dirac_complex_int!(test_dirac_complex_i16, i16);
// test_dirac_complex_int!(test_dirac_complex_i32, i32);
// test_dirac_complex_int!(test_dirac_complex_i64, i64);

fn conv_with_sizes(filter_len: usize, signal_len: usize) {
    let filter: Vec<_> = (0..filter_len).into_iter().map(|_| 1.).collect();

    let planner = Conv1dPlanner::new();
    let mut conv = planner.plan_conv1d(&filter, ConvMode::Full);

    let signal: Vec<_> = (0..signal_len)
        .into_iter()
        .map(|_| Complex32::ONE)
        .collect();

    let now = Instant::now();
    let actual = conv.process(signal);
    println!(
        "Convolution with {} kernel and {} signal took {} ms",
        filter_len,
        signal_len,
        now.elapsed().as_millis()
    );

    assert_eq!(actual.len(), signal_len + filter_len - 1);
}
#[test]
fn test_input_sizes() {
    let mut rng = rand::thread_rng();

    for _ in 0..1000 {
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
