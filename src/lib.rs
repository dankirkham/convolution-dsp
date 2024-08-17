//!
//! ```
//! use convolution_dsp::{ConvMode, Conv1dPlanner};
//! use num_complex::Complex32;
//!
//! let filter = vec![0., 0., 0., 1., 0., 0., 0.];
//!
//! let planner = Conv1dPlanner::new();
//! let mut conv = planner.plan_conv1d(&filter, ConvMode::Same);
//!
//! let signal = vec![Complex32::ONE; 1_000_000];
//!
//! let output = conv.process(signal);
//!
//! assert_eq!(output.len(), 1_000_000);
//!
//! ```

mod conv;
mod conv_num;
mod mode;
mod planner;

#[doc(hidden)]
pub mod file;

pub use conv::Conv1d;
pub use conv_num::ConvNum;
pub use mode::ConvMode;
pub use planner::Conv1dPlanner;
