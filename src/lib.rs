//!
//! ```
//! use convolution_dsp::Conv1dPlanner;
//! use num_complex::Complex32;
//!
//! let filter = vec![0., 0., 0., 1., 0., 0., 0.];
//!
//! let planner = Conv1dPlanner::new();
//! let mut conv = planner.plan_conv1d(&filter);
//!
//! let signal = vec![Complex32::ONE; 1_000_000];
//!
//! let output = conv.process(signal);
//!
//! assert_eq!(output.len(), 1_000_006);
//!
//! ```

mod conv;
mod planner;

#[doc(hidden)]
pub mod file;

pub use conv::Conv1d;
pub use planner::Conv1dPlanner;
