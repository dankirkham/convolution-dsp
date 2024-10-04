use num_traits::cast::NumCast;
use num_traits::ConstZero;
use rustfft::FftNum;

pub trait ConvNum: FftNum + ConstZero + NumCast {}

impl<T> ConvNum for T where T: FftNum + ConstZero + NumCast {}
