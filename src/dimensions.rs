#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dimensions {
    Scalar,
    MultiDimensional(Vec<Dimension>),
}

impl AsRef<Dimensions> for Dimensions {
    fn as_ref(&self) -> &Dimensions {
        self
    }
}

impl Dimensions {
    pub fn new(dimensions: Vec<Dimension>) -> Self {
        if dimensions.is_empty() {
            Dimensions::Scalar
        } else {
            Dimensions::MultiDimensional(dimensions)
        }
    }

    pub fn num_dims(&self) -> usize {
        match self {
            Dimensions::Scalar => 0,
            Dimensions::MultiDimensional(dims) => dims.len(),
        }
    }

    pub fn total_flat_size(&self) -> usize {
        match self {
            Dimensions::Scalar => 1,
            Dimensions::MultiDimensional(dims) => dims.iter().map(|dim| dim.size).product(),
        }
    }

    pub fn new_scalar() -> Self {
        Dimensions::Scalar
    }

    pub fn is_scalar(&self) -> bool {
        matches!(self, Dimensions::Scalar)
    }

    pub fn is_multi_dimensional(&self) -> bool {
        matches!(self, Dimensions::MultiDimensional(_))
    }

    pub fn as_multi_dimensional(&self) -> Option<&Vec<Dimension>> {
        if let Dimensions::MultiDimensional(dims) = self {
            Some(dims)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dimension {
    pub name: String,
    pub size: usize,
}

impl From<(&str, usize)> for Dimension {
    fn from(tuple: (&str, usize)) -> Self {
        Dimension {
            name: tuple.0.to_string(),
            size: tuple.1,
        }
    }
}
