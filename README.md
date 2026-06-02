# nd-arrow-array

## Selecting an arrow version

This crate exposes [`arrow`](https://crates.io/crates/arrow) types in its public API, so the arrow
major version is selected at compile time via mutually-exclusive Cargo features. Supported versions
are **56**, **57**, and **58** (the default).

Pick the version that matches the rest of your dependency tree:

```toml
# Default (arrow 58):
nd-arrow-array = "2"

# Or select a specific arrow major:
nd-arrow-array = { version = "2", default-features = false, features = ["arrow-57"] }
```

Exactly one `arrow-*` feature must be enabled; enabling none or more than one is a compile error.

To guarantee the arrow types you pass in match the version the crate was built against, construct
them through the re-exported module rather than depending on `arrow` separately:

```rust
use nd_arrow_array::arrow; // resolves to the selected arrow version

let array = arrow::array::Int32Array::from(vec![1, 2, 3]);
```
