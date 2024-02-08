# complex-stuff

`complex-stuff` is a collection of utilities to make calculations with complex numbers easy.

## Examples

```rust
use complex_stuff::{Complex, ComplexCartesian, ComplexPolar};
use std::f64::consts::PI;

// Some of the most important operations:
fn main() {
    // Declare complex numbers with their cartesian or polar form.
    let c1 = Complex::new_cartesian(5.0, 3.0);
    let c2 = Complex::new_polar(7.0, PI / 4.0);

    // Basic arethmetic operations should work as expected.
    let sum = c1 + c2;
    let diff = c1 - c2;

    let prod = c1 * c2;

    // All operations which can result in undefined or infinite values
    // return a `Option` and need to be handled.

    let quot = (c1 / c2)?;

    let sqr = c1.pow(Complex::new_real(2.0))?;
    let sqrt = c1.root(Complex::new_real(2.0))?;

    let log10 = c1.log(Complex::new_real(10.0))?;

    let sin = c1.sin()?;
    let cos = c1.cos()?;
}
```

License: MIT
