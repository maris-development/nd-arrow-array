use std::{ops::Deref, sync::Arc};

#[cfg(feature = "arrow-56")]
pub use arrow_56 as arrow;
#[cfg(feature = "arrow-57")]
pub use arrow_57 as arrow;
#[cfg(feature = "arrow-58")]
pub use arrow_58 as arrow;

// Exactly one arrow version must be selected.
#[cfg(not(any(feature = "arrow-56", feature = "arrow-57", feature = "arrow-58")))]
compile_error!("Enable exactly one arrow feature: `arrow-56`, `arrow-57`, or `arrow-58`.");
#[cfg(any(
    all(feature = "arrow-56", feature = "arrow-57"),
    all(feature = "arrow-56", feature = "arrow-58"),
    all(feature = "arrow-57", feature = "arrow-58"),
))]
compile_error!("The arrow version features are mutually exclusive; enable only one.");

use crate::arrow::{
    array::{ArrayRef, NullArray, new_null_array},
    datatypes::DataType,
};

use crate::{dimensions::Dimensions, error::NdArrayError};

pub mod batch;
pub mod broadcast;
pub mod dimensions;
pub mod error;

#[derive(Debug, Clone)]
pub struct NdArrowArray {
    arrow_array: ArrayRef,
    dimensions: Dimensions,
}

impl NdArrowArray {
    pub fn new_null_scalar(data_type: Option<DataType>) -> Self {
        match data_type {
            Some(data_type) => {
                let array = new_null_array(&data_type, 1);
                Self::new(array, Dimensions::Scalar).unwrap()
            }
            None => Self::new(Arc::new(NullArray::new(1)), Dimensions::Scalar).unwrap(),
        }
    }

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
