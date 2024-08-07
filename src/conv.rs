use std::sync::Arc;

use num_complex::Complex32;
use rustfft::Fft;

use crate::ConvMode;

pub struct Conv1d {
    kernel: Vec<Complex32>,
    kernel_len: usize,
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    mode: ConvMode,
}

impl Conv1d {
    pub fn new(
        kernel: Vec<Complex32>,
        kernel_len: usize,
        fft: Arc<dyn Fft<f32>>,
        ifft: Arc<dyn Fft<f32>>,
        mode: ConvMode,
    ) -> Self {
        Self {
            kernel,
            kernel_len,
            fft,
            ifft,
            mode,
        }
    }

    pub fn process(&mut self, mut input: Vec<Complex32>) -> Vec<Complex32> {
        let input_len = input.len();
        let segment_len = self.fft.len() - self.kernel_len - 1;
        let segments = ((input_len as f32) / (segment_len as f32)).ceil() as usize;

        // When an N sample signal is convolved with an M sample filter kernel, the output signal
        // is N + M + 1 samples long.
        input.extend_from_slice(&vec![Complex32::ZERO; self.kernel_len - 1]);

        let mut output = vec![Complex32::ZERO; input_len + self.kernel_len - 1];

        let needed_len = segments * segment_len;
        if input_len < needed_len {
            input.extend_from_slice(&vec![Complex32::ZERO; needed_len - input_len]);
        }

        let mut segment = Vec::with_capacity(self.fft.len());
        for i in 0..segments {
            let offset = i * segment_len;
            segment.extend_from_slice(&input[offset..(offset + segment_len)]);
            segment.extend(vec![Complex32::ZERO; self.fft.len() - segment_len]);

            // FFT the segment
            self.fft.process(&mut segment);

            // Piecewise multiply with kernel.
            for j in 0..segment.len() {
                segment[j] *= self.kernel[j];
            }

            // IFFT back to time domain
            self.ifft.process(&mut segment);

            // Normalize and accumulate to output
            for j in 0..segment.len() {
                if offset + j < output.len() {
                    output[offset + j] += segment[j] / self.fft.len() as f32;
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
