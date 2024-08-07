#[derive(Debug)]
pub enum ConvMode {
    /// Returns full length convolution. `filter.len() + signal.len() - 1`
    Full,
    /// Return data is the same size as the largest operand. `filter.len().max(signal.len())`
    Same,
}
