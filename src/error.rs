use crate::dimensions::Dimensions;

#[derive(Debug, thiserror::Error)]
pub enum NdArrayError {
    #[error("Broadcasting error occurred: {0}")]
    BroadcastingError(BroadcastError),
    #[error("Array length: {0} and dimensions: {1:?} don't align.")]
    MisalignedArrayDimensions(usize, Dimensions),
}

#[derive(Debug, thiserror::Error)]
pub enum BroadcastError {
    #[error("Incompatible array shapes: {0:?} and {1:?}")]
    IncompatibleShapes(Dimensions, Dimensions),
    #[error("Unsupport data type for broadcasting: {0:?}")]
    UnsupportedArrowDataType(arrow::datatypes::DataType),
    #[error("Cannot find a broadcastable shape for the following dimensions: {0:?}")]
    NoBroadcastableShape(Vec<Dimensions>),
}
