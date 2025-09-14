use std::sync::Arc;

use arrow::{
    array::{ArrayRef, RecordBatch},
    datatypes::{Field, Schema, SchemaRef},
};

use crate::{NdArrowArray, broadcast::broadcast_arrays, error::NdArrayError};

#[derive(Debug, Clone)]
pub struct NdRecordBatch {
    schema: SchemaRef,
    arrays: Vec<NdArrowArray>,
}

impl NdRecordBatch {
    pub fn new(fields: Vec<Field>, arrays: Vec<NdArrowArray>) -> Result<Self, NdArrayError> {
        assert_eq!(
            fields.len(),
            arrays.len(),
            "Number of fields must match number of arrays"
        );

        Ok(Self {
            schema: Schema::new(fields).into(),
            arrays,
        })
    }

    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    pub fn arrays(&self) -> &[NdArrowArray] {
        &self.arrays
    }

    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }

    pub fn to_arrow_record_batch(&self) -> Result<RecordBatch, NdArrayError> {
        // Broadcast the arrays
        let broadcast_arrays =
            broadcast_arrays(&self.arrays).map_err(NdArrayError::BroadcastingError)?;

        let arrow_arrays: Vec<ArrayRef> = broadcast_arrays
            .into_iter()
            .map(|a| a.as_arrow_array().clone())
            .collect();

        if arrow_arrays.is_empty() {
            return Ok(RecordBatch::new_empty(Arc::new(Schema::empty())));
        }

        // Create new record batch
        Ok(RecordBatch::try_new(self.schema(), arrow_arrays).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name() {
        let nd_batch = NdRecordBatch::new(vec![], vec![]).unwrap();

        let flat = nd_batch.to_arrow_record_batch().unwrap();

        println!("{:?}", flat);
    }
}
