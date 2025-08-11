use std::ops::Deref;

use arrow::array::ArrayRef;

use crate::{dimensions::Dimensions, error::NdArrayError};

pub mod batch;
pub mod broadcast;
pub mod dimensions;
pub mod error;

pub struct NdArrowArray {
    arrow_array: ArrayRef,
    dimensions: Dimensions,
}

impl NdArrowArray {
    pub fn new(arrow_array: ArrayRef, dimensions: Dimensions) -> Result<Self, NdArrayError> {
        // Validate the dimensions against the array shape
        if arrow_array.len() != dimensions.total_flat_size() {
            return Err(NdArrayError::MisalignedArrayDimensions(
                arrow_array.len(),
                dimensions,
            ));
        }

        Ok(NdArrowArray {
            arrow_array,
            dimensions,
        })
    }

    pub fn as_arrow_array(&self) -> &ArrayRef {
        &self.arrow_array
    }

    pub fn dimensions(&self) -> &Dimensions {
        &self.dimensions
    }
}

impl Deref for NdArrowArray {
    type Target = ArrayRef;

    fn deref(&self) -> &Self::Target {
        &self.arrow_array
    }
}
