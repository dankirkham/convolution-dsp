use std::sync::Arc;

use num_complex::Complex;
use rustfft::Fft;

use crate::{ConvMode, ConvNum};

pub struct Conv1d<T: ConvNum> {
    kernel: Vec<Complex<T>>,
    kernel_len: usize,
    fft: Arc<dyn Fft<T>>,
    ifft: Arc<dyn Fft<T>>,
    mode: ConvMode,
    fft_length: T,
}

impl<T: ConvNum> Conv1d<T> {
    pub fn new(
        kernel: Vec<Complex<T>>,
        kernel_len: usize,
        fft: Arc<dyn Fft<T>>,
        ifft: Arc<dyn Fft<T>>,
        mode: ConvMode,
        fft_length: T,
    ) -> Self {
        Self {
            kernel,
            kernel_len,
            fft,
            ifft,
            mode,
            fft_length,
        }
    }

    pub fn process(&mut self, mut input: Vec<Complex<T>>) -> Vec<Complex<T>> {
        let input_len = input.len();
        let segment_len = self.fft.len() - self.kernel_len - 1;
        let segments = ((input_len as f32) / (segment_len as f32)).ceil() as usize;

        // When an N sample signal is convolved with an M sample filter kernel, the output signal
        // is N + M + 1 samples long.
        input.extend_from_slice(&vec![Complex::<T>::ZERO; self.kernel_len - 1]);

        let mut output = vec![Complex::<T>::ZERO; input_len + self.kernel_len - 1];

        let needed_len = segments * segment_len;
        if input_len < needed_len {
            input.extend_from_slice(&vec![Complex::<T>::ZERO; needed_len - input_len]);
        }

        let mut segment = Vec::with_capacity(self.fft.len());
        for i in 0..segments {
            let offset = i * segment_len;
            segment.extend_from_slice(&input[offset..(offset + segment_len)]);
            segment.extend(vec![Complex::<T>::ZERO; self.fft.len() - segment_len]);

            // FFT the segment
            self.fft.process(&mut segment);

            // Piecewise multiply with kernel.
            for (j, value) in segment.iter_mut().enumerate() {
                *value = *value * self.kernel[j];
            }

            // IFFT back to time domain
            self.ifft.process(&mut segment);

            // Normalize and accumulate to output
            for j in 0..segment.len() {
                if offset + j < output.len() {
                    output[offset + j] = output[offset + j] + (segment[j] / self.fft_length);
                } else {
                    break;
                }
            }

            segment.clear();
        }

        match self.mode {
            ConvMode::Full => output,
            ConvMode::Same => {
                let target_len = input_len.max(self.kernel_len);
                let left = (output.len() - target_len) / 2;
                let right = left + target_len;

                output[left..right].to_vec()
            }
        }
    }
}
