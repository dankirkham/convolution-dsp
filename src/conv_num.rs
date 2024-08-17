use num_traits::cast::NumCast;
use num_traits::ConstZero;
use rustfft::FftNum;

pub trait ConvNum: FftNum + ConstZero + NumCast {}

impl ConvNum for f32 {}
impl ConvNum for f64 {}

// impl ConvNum for i8 {}
// impl ConvNum for i16 {}
// impl ConvNum for i32 {}
// impl ConvNum for i64 {}
