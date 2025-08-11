use std::sync::Arc;

use arrow::array::{Array, ArrowPrimitiveType, Int8Array, PrimitiveArray};

use crate::{NdArrowArray, dimensions::Dimensions, error::BroadcastError};

pub fn broadcast_arrays(arrays: &[NdArrowArray]) -> Result<Vec<NdArrowArray>, BroadcastError> {
    let dimensions = arrays
        .iter()
        .map(|a| a.dimensions().clone())
        .collect::<Vec<_>>();
    let broadcast_dimension = find_broadcast_dimension_shape(&dimensions)
        .ok_or(BroadcastError::NoBroadcastableShape(dimensions))?;

    let mut broadcasted_arrays = vec![];
    for array in arrays {
        broadcasted_arrays.push(broadcast_array(array, &broadcast_dimension)?);
    }

    Ok(broadcasted_arrays)
}

pub fn find_broadcast_dimension_shape<D: AsRef<Dimensions>>(
    dimensions: &[D],
) -> Option<Dimensions> {
    //Find the biggest dimensions array
    let dimensions: Vec<&Dimensions> = dimensions
        .iter()
        .map(|dim| dim.as_ref())
        .collect::<Vec<_>>();

    let mut max_dimension_array: Option<&Dimensions> = None;
    for dim in dimensions {
        if max_dimension_array.is_none()
            || dim.num_dims() > max_dimension_array.as_ref().unwrap().num_dims()
        {
            max_dimension_array = Some(dim);
        }
    }

    max_dimension_array.cloned()
}

pub fn broadcast_array(
    array: &NdArrowArray,
    target_dimensions: &Dimensions,
) -> Result<NdArrowArray, BroadcastError> {
    let dimensions = array.dimensions();
    let (repeat_slice, repeat_element) =
        broadcast_reshape_args(dimensions, target_dimensions).ok_or(
            BroadcastError::IncompatibleShapes(dimensions.clone(), target_dimensions.clone()),
        )?;

    let broadcasted_array =
        broadcast_array_impl(&array.as_arrow_array(), repeat_element, repeat_slice)?;

    let nd_array = NdArrowArray::new(broadcasted_array, target_dimensions.clone()).unwrap();

    Ok(nd_array)
}

fn find_subslice<T: PartialEq>(haystack: &[T], needle: &[T]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn broadcast_reshape_args(
    dimensions: &Dimensions,
    target_dimensions: &Dimensions,
) -> Option<(usize, usize)> {
    if dimensions == target_dimensions {
        return Some((1, 1));
    }

    match dimensions {
        Dimensions::Scalar => {
            // Just scale up to the target dimensions
            Some((1, target_dimensions.total_flat_size()))
        }
        Dimensions::MultiDimensional(dimensions) => {
            // Needs reshaping of the array
            match target_dimensions {
                Dimensions::Scalar => None,
                Dimensions::MultiDimensional(target_dimensions) => {
                    if dimensions.len() > target_dimensions.len() {
                        return None;
                    }

                    match find_subslice(target_dimensions, dimensions) {
                        Some(start_loc) => {
                            let repeat_slice_count = target_dimensions[..start_loc]
                                .iter()
                                .map(|d| d.size)
                                .product::<usize>();
                            let repeat_element_counter = target_dimensions
                                [start_loc + dimensions.len()..]
                                .iter()
                                .map(|d| d.size)
                                .product::<usize>();

                            Some((repeat_slice_count, repeat_element_counter))
                        }
                        None => None,
                    }
                }
            }
        }
    }
}

fn broadcast_array_impl(
    array: &dyn Array,
    repeat_element_count: usize,
    repeat_slice_count: usize,
) -> Result<Arc<dyn Array>, BroadcastError> {
    match array.data_type() {
        arrow::datatypes::DataType::Int8 => {
            let array = array.as_any().downcast_ref::<Int8Array>().unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Int16 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::Int16Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Int32 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Int64 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::UInt8 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::UInt8Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::UInt16 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::UInt16Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::UInt32 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::UInt32Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::UInt64 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Float32 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::Float32Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Float64 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap();
            let result =
                broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Boolean => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
                .unwrap();
            let result = broadcast_boolean_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Null => {
            let new_len = array.len() * repeat_slice_count * repeat_element_count;
            let result = arrow::array::NullArray::new(new_len);
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Timestamp(unit, _) => match unit {
            arrow::datatypes::TimeUnit::Second => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow::array::TimestampSecondArray>()
                    .unwrap();
                let result =
                    broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
                Ok(Arc::new(result))
            }
            arrow::datatypes::TimeUnit::Millisecond => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow::array::TimestampMillisecondArray>()
                    .unwrap();
                let result =
                    broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
                Ok(Arc::new(result))
            }
            arrow::datatypes::TimeUnit::Microsecond => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
                    .unwrap();
                let result =
                    broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
                Ok(Arc::new(result))
            }
            arrow::datatypes::TimeUnit::Nanosecond => {
                let array = array
                    .as_any()
                    .downcast_ref::<arrow::array::TimestampNanosecondArray>()
                    .unwrap();
                let result =
                    broadcast_primitive_array(array, repeat_element_count, repeat_slice_count)?;
                Ok(Arc::new(result))
            }
        },
        arrow::datatypes::DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            let result = broadcast_string_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        arrow::datatypes::DataType::Binary => {
            let array = array
                .as_any()
                .downcast_ref::<arrow::array::BinaryArray>()
                .unwrap();
            let result = broadcast_byte_array(array, repeat_element_count, repeat_slice_count)?;
            Ok(Arc::new(result))
        }
        dtype => Err(BroadcastError::UnsupportedArrowDataType(dtype.clone())),
    }
}

fn broadcast_primitive_array<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    repeat_element_count: usize,
    repeat_slice_count: usize,
) -> Result<PrimitiveArray<T>, BroadcastError> {
    Ok(PrimitiveArray::<T>::from_iter(
        std::iter::repeat_with(|| array.iter()) // (1) clone slice iterator
            .take(repeat_slice_count) // keep only `times` copies
            .flatten() // Iterator<Item = i32>
            .flat_map(
                move |item|                      // (2) repeat each element
                std::iter::repeat_n(item, repeat_element_count), //      `n` times
            ),
    ))
}

fn broadcast_boolean_array(
    array: &arrow::array::BooleanArray,
    repeat_element_count: usize,
    repeat_slice_count: usize,
) -> Result<arrow::array::BooleanArray, BroadcastError> {
    Ok(arrow::array::BooleanArray::from_iter(
        std::iter::repeat_with(|| array.iter()) // (1) clone slice iterator
            .take(repeat_slice_count) // keep only `times` copies
            .flatten() // Iterator<Item = i32>
            .flat_map(
                move |item|                      // (2) repeat each element
                std::iter::repeat_n(item, repeat_element_count), //      `n` times
            ),
    ))
}

fn broadcast_string_array(
    array: &arrow::array::StringArray,
    repeat_element_count: usize,
    repeat_slice_count: usize,
) -> Result<arrow::array::StringArray, BroadcastError> {
    Ok(arrow::array::StringArray::from_iter(
        std::iter::repeat_with(|| array.iter()) // (1) clone slice iterator
            .take(repeat_slice_count) // keep only `times` copies
            .flatten() // Iterator<Item = i32>
            .flat_map(
                move |item|                      // (2) repeat each element
                std::iter::repeat_n(item, repeat_element_count), //      `n` times
            ),
    ))
}

fn broadcast_byte_array(
    array: &arrow::array::BinaryArray,
    repeat_element_count: usize,
    repeat_slice_count: usize,
) -> Result<arrow::array::BinaryArray, BroadcastError> {
    Ok(arrow::array::BinaryArray::from_iter(
        std::iter::repeat_with(|| array.iter()) // (1) clone slice iterator
            .take(repeat_slice_count) // keep only `times` copies
            .flatten() // Iterator<Item = i32>
            .flat_map(
                move |item|                      // (2) repeat each element
                std::iter::repeat_n(item, repeat_element_count), //      `n` times
            ),
    ))
}
