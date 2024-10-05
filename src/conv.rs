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

    pub fn process(&mut self, input: Vec<Complex<T>>) -> Vec<Complex<T>> {
        let segment_len = self.fft.len() - self.kernel_len - 1;
        let segments = ((input.len() as f32) / (segment_len as f32)).ceil() as usize;

        let mut output = vec![Complex::<T>::ZERO; input.len() + self.kernel_len - 1];

        let mut segment = Vec::with_capacity(self.fft.len());
        for i in 0..segments {
            let offset = i * segment_len;
            let end = offset + segment_len;
            if end > input.len() {
                segment.extend_from_slice(&input[offset..input.len()]);
                segment.extend(vec![Complex::<T>::ZERO; end - input.len()]);
            } else {
                segment.extend_from_slice(&input[offset..(offset + segment_len)]);
            }
            segment.extend(vec![Complex::<T>::ZERO; self.fft.len() - segment_len]);
            assert_eq!(segment.len(), self.fft.len());

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
                let target_len = input.len().max(self.kernel_len);
                let left = (output.len() - target_len) / 2;
                let right = left + target_len;

                output[left..right].to_vec()
            }
        }
    }
}
