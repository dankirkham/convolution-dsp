# convolution-dsp

1-dimensional convolution library for Rust intended for use in DSP
applications. Uses the overlap-add FFT method.

## Planned features
- Input signal
    - ☑ Complex32 signals
    - ☐ f32 signals
- Filter kernels
    - ☑ f32 filter kernels
    - ☐ Complex32 filter kernels
- ☐ Use realfft when signal and kernel are both f32
- ☐ f64 support
- ☐ Fallbacks to non-fft convolution when it is faster
- ☐ Full or same length output mode, similar to numpy/scipy
- ☐ Threading
- ☐ Minimize memory allocations
- ☐ Faster than numpy/scipy
- ☐ Re-export num\_complex?

## References
Stephen W. Smith, Ph.D., [The Scientist and Engineer's Guide to Digital Signal Processing](https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch18.pdf), Chapter 18.
