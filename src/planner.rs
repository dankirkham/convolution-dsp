use num_complex::Complex;
use rustfft::FftPlanner;

use crate::conv::Conv1d;
use crate::{ConvMode, ConvNum};

pub struct Conv1dPlanner;

impl Conv1dPlanner {
    pub fn new() -> Self {
        Self
    }

    pub fn plan_conv1d<T: ConvNum>(&self, kernel: &[T], mode: ConvMode) -> Conv1d<T> {
        let kernel_len = kernel.len();
        assert!(kernel_len > 1);

        // FFT size must be reasonably large to avoid circular convolution
        let fft_size = if kernel_len & (kernel_len - 1) != 0 {
            usize::pow(2, kernel_len.ilog2() + 2)
        } else {
            kernel_len * 2
        };

        let mut fft_planner = FftPlanner::new();
        let fft = fft_planner.plan_fft_forward(fft_size);
        let ifft = fft_planner.plan_fft_inverse(fft_size);

        let mut kernel: Vec<_> = kernel
            .iter()
            .map(|re| Complex::<T>::new(*re, T::ZERO))
            .collect();
        kernel.extend(vec![Complex::<T>::ZERO; fft_size - kernel.len()]);
        fft.process(&mut kernel);

        let fft_length = match T::from(fft.len()) {
            Some(len) => len,
            None => panic!("Failed to convert usize to T."),
        };

        Conv1d::new(kernel, kernel_len, fft, ifft, mode, fft_length)
    }
}
