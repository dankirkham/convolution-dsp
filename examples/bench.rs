use std::time::Instant;

use num_complex::Complex32;
use rand::prelude::*;

use convolution_dsp::{Conv1dPlanner, ConvMode};

const KERNEL_SIZE: usize = 1024;
const SIGNAL_SIZE: usize = 1024 * 64;
const ITERATIONS: usize = 100;

fn main() {
    let mut rng = rand::thread_rng();
    let filter: Vec<f32> = (0..KERNEL_SIZE)
        .map(|_| rng.gen_range(-1_f32..1.))
        .collect();

    let planner = Conv1dPlanner::new();
    let mut conv = planner.plan_conv1d(&filter, ConvMode::Full);

    let now = Instant::now();
    for _ in 0..ITERATIONS {
        let signal: Vec<Complex32> = (0..SIGNAL_SIZE)
            .map(|_| Complex32::new(rng.gen_range(-1_f32..1.), rng.gen_range(-1_f32..1.)))
            .collect();
        conv.process(signal.clone());
    }
    println!(
        "Iterations: {}, Kernel: {}, Signal: {}, Time: {} ms",
        ITERATIONS,
        KERNEL_SIZE,
        SIGNAL_SIZE,
        now.elapsed().as_millis()
    );
}
