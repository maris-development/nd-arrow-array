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

    pub fn to_arrow_record_batch(&self) -> Result<RecordBatch, NdArrayError> {
        // Broadcast the arrays
        let broadcast_arrays =
            broadcast_arrays(&self.arrays).map_err(NdArrayError::BroadcastingError)?;

        let arrow_arrays: Vec<ArrayRef> = broadcast_arrays
            .into_iter()
            .map(|a| a.as_arrow_array().clone())
            .collect();

        // Create new record batch
        Ok(RecordBatch::try_new(self.schema(), arrow_arrays)?)
    }
}
