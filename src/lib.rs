//! `complex-stuff` is a collection of utilities to make calculations with complex numbers easy.
//!
//! # Examples
//!
//! ```
//! use complex_stuff::{Complex, ComplexCartesian, ComplexPolar};
//! use std::f64::consts::PI;
//!
//! // Some of the most important operations:
//! fn main() {
//!     // Declare complex numbers with their cartesian or polar form.
//!     let c1 = Complex::new_cartesian(5.0, 3.0);
//!     let c2 = Complex::new_polar(7.0, PI / 4.0);
//!
//!     // Basic arethmetic operations should work as expected.
//!     let sum = c1 + c2;
//!     let diff = c1 - c2;
//!
//!     let prod = c1 * c2;
//!
//!     // All operations which can result in undefined or infinite values
//!     // return a `Option` and need to be handled.
//!
//!     let quot = (c1 / c2)?;
//!
//!     let sqr = c1.pow(Complex::new_real(2.0))?;
//!     let sqrt = c1.root(Complex::new_real(2.0))?;
//!
//!     let log10 = c1.log(Complex::new_real(10.0))?;
//!     
//!     let sin = c1.sin()?;
//!     let cos = c1.cos()?;
//! }
//! ```

use std::{
    f64::consts::E,
    ops::{Add, Div, Mul, Neg, Sub},
};

#[derive(Debug, Clone, Copy)]
/// Descibes a complex number in cartesion form _`re` + `im`i_.
pub struct ComplexCartesian {
    pub re: f64,
    pub im: f64,
}

impl std::fmt::Display for ComplexCartesian {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} + {}i", self.re, self.im)
    }
}

impl ComplexCartesian {
    /// Converts a complex number from polar to cartesion form.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64::consts::PI;
    /// use complex_stuff::{ComplexCartesian, ComplexPolar};
    ///
    /// let polar = ComplexPolar { mag: 5.0, ang: PI / 2.0 };
    /// let cartesian = ComplexCartesian::from_polar(&polar);
    /// ```
    pub fn from_polar(polar: &ComplexPolar) -> Self {
        let re = polar.mag * polar.ang.cos();
        let im = polar.mag * polar.ang.sin();
        return Self { re, im };
    }
}

#[derive(Debug, Clone, Copy)]
/// Describes a complex number in polar form _`mag`e^(`ang`i)_
pub struct ComplexPolar {
    pub mag: f64,
    pub ang: f64,
}

impl std::fmt::Display for ComplexPolar {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}e^({}i)", self.mag, self.ang)
    }
}

impl ComplexPolar {
    /// Converts a complex number from cartesian to polar form.
    ///
    /// # Examples
    ///
    /// ```
    /// use complex_stuff::{ComplexCartesian, ComplexPolar};
    ///
    /// let cartesian = ComplexCartesian { re: 5.0, im: 3.0 };
    /// let polar = ComplexPolar::from_cartesian(&cartesian);
    /// ```
    pub fn from_cartesian(cartesian: &ComplexCartesian) -> Self {
        let mag = (cartesian.re * cartesian.re + cartesian.im * cartesian.im).sqrt();
        let ang = (cartesian.re / mag).acos();
        return Self { mag, ang };
    }
}

#[derive(Debug, Clone, Copy)]
/// Describes a complex number in both cartesian and polar form.
pub struct Complex {
    cartesian: ComplexCartesian,
    polar: ComplexPolar,
}

impl std::fmt::Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Cartesian: {}\nPolar: {}", self.cartesian, self.polar)
    }
}

impl Complex {
    /// Creates a complex number from its cartesian parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_cartesian(5.0, 3.0);
    /// ```
    pub fn new_cartesian(re: f64, im: f64) -> Self {
        let cartesian = ComplexCartesian { re, im };
        let polar = ComplexPolar::from_cartesian(&cartesian);

        Self { cartesian, polar }
    }

    /// Creates a complex number from its polar parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64::consts::PI;
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_polar(5.0, PI / 2.0);
    /// ```
    pub fn new_polar(mag: f64, ang: f64) -> Self {
        let polar = ComplexPolar { mag, ang };
        let cartesian = ComplexCartesian::from_polar(&polar);

        Self { cartesian, polar }
    }

    /// Creates a complex number just from its real cartesian part leaving the imaginary part 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_real(5.0);
    ///
    /// assert_eq!(5.0, complex.re());
    /// assert_eq!(0.0, complex.im());
    /// ```
    pub fn new_real(re: f64) -> Self {
        Self::new_cartesian(re, 0.0)
    }

    /// Creates a complex number just from its imaginary cartesian part leaving the real part 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_imaginary(3.0);
    ///
    /// assert_eq!(0.0, complex.re());
    /// assert_eq!(3.0, complex.im());
    /// ```
    pub fn new_imaginary(im: f64) -> Self {
        Self::new_cartesian(0.0, im)
    }

    /// Creates a complex number representing the value 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::zero();
    ///
    /// assert_eq!(0.0, complex.re());
    /// assert_eq!(0.0, complex.im());
    /// ```
    pub fn zero() -> Self {
        Self::new_cartesian(0.0, 0.0)
    }

    /// Creates a complex number representing the value 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::one();
    ///
    /// assert_eq!(1.0, complex.re());
    /// assert_eq!(0.0, complex.im());
    /// ```
    pub fn one() -> Self {
        Self::new_cartesian(1.0, 0.0)
    }

    /// Creates a complex number representing the value i.
    ///
    /// # Examples
    ///
    /// ```
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::i();
    ///
    /// assert_eq!(0.0, complex.re());
    /// assert_eq!(1.0, complex.im());
    /// ```
    pub fn i() -> Self {
        Self::new_cartesian(0.0, 1.0)
    }

    /// Creates a complex number representing the value e.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64::consts::E;
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::e();
    ///
    /// assert_eq!(E, complex.re());
    /// assert_eq!(0.0, complex.im());
    /// ```
    pub fn e() -> Self {
        Self::new_real(E)
    }
}

impl Complex {
    /// Returns the complex number in cartesian form.
    ///
    /// # Examples
    ///
    /// ```
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_cartesian(5.0, 3.0);
    ///
    /// let cartesian = complex.cartesian();
    ///
    /// assert_eq!(5.0, cartesian.re);
    /// assert_eq!(3.0, cartesian.im);
    /// ```
    pub fn cartesian(&self) -> ComplexCartesian {
        self.cartesian
    }

    /// Returns the complex number in polar form.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64::consts::PI;
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_polar(5.0, PI / 2.0);
    ///
    /// let polar = complex.polar();
    ///
    /// assert_eq!(5.0, polar.mag);
    /// assert_eq!(PI / 2.0, polar.ang);
    /// ```
    pub fn polar(&self) -> ComplexPolar {
        self.polar
    }

    /// Returns just the real part of the complex numbers cartesian form.
    ///
    /// # Examples
    ///
    /// ```
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_cartesian(5.0, 3.0);
    ///
    /// let real = complex.re();
    ///
    /// assert_eq!(5.0, real);
    /// ```
    pub fn re(&self) -> f64 {
        self.cartesian.re
    }

    /// # Examples
    ///
    /// ```
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_cartesian(5.0, 3.0);
    ///
    /// let imaginary = complex.im();
    ///
    /// assert_eq!(3.0, imaginary);
    /// ```
    /// Returns just the imaginary part of the complex numbers cartesian form.
    pub fn im(&self) -> f64 {
        self.cartesian.im
    }

    /// Returns just the magnitude of the complex numbers polar form.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64::consts::PI;
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_polar(5.0, PI / 2.0);
    ///
    /// let magnitude = complex.mag();
    ///
    /// assert_eq!(5.0, magnitude);
    /// ```
    pub fn mag(&self) -> f64 {
        self.polar.mag
    }

    /// Returns just the angle of the complex numbers polar form.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64::consts::PI;
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_polar(5.0, PI / 2.0);
    ///
    /// let angle = complex.ang();
    ///
    /// assert_eq!(PI / 2.0, angle);
    /// ```
    pub fn ang(&self) -> f64 {
        self.polar.ang
    }
}

impl Complex {
    /// Returns the negation of the complex number.
    /// Same as using the unary negation operator `-`.
    ///
    /// # Examples
    ///
    /// ```
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_cartesian(5.0, 3.0);
    ///
    /// let opposite = complex.opposite();
    ///
    /// assert_eq!(-5.0, opposite.re());
    /// assert_eq!(-3.0, opposite.im());
    /// ```
    pub fn opposite(self) -> Self {
        let re = -self.cartesian.re;
        let im = -self.cartesian.im;

        Self::new_cartesian(re, im)
    }

    /// Returns the reciprocal of the complex number.
    /// # Examples
    ///
    /// ```
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_cartesian(5.0, 0.0);
    ///
    /// let reciprocal = complex.reciprocal();
    /// ```
    pub fn reciprocal(self) -> Option<Self> {
        let sqr_mag = self.cartesian.re * self.cartesian.re + self.cartesian.im * self.cartesian.im;

        if sqr_mag == 0.0 {
            return None;
        }

        let re = self.cartesian.re / sqr_mag;
        let im = -self.cartesian.im / sqr_mag;

        Some(Self::new_cartesian(re, im))
    }
}

impl Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self {
        self.opposite()
    }
}

impl Add for Complex {
    type Output = Self;

    /// Performs the `+` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use complex_stuff::Complex;
    ///
    /// let c1 = Complex::new_cartesian(5.0, 3.0);
    /// let c2 = Complex::new_cartesian(1.0, 2.0);
    ///
    /// let sum = c1 + c2;
    ///
    /// assert_eq!(6.0, sum.re());
    /// assert_eq!(5.0, sum.im());
    /// ```
    fn add(self, other: Self) -> Self {
        let re = self.cartesian.re + other.cartesian.re;
        let im = self.cartesian.im + other.cartesian.im;

        Self::new_cartesian(re, im)
    }
}

impl Sub for Complex {
    type Output = Self;

    /// Performs the `-` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use complex_stuff::Complex;
    ///
    /// let c1 = Complex::new_cartesian(5.0, 3.0);
    /// let c2 = Complex::new_cartesian(1.0, 2.0);
    ///
    /// let diff = c1 - c2;
    ///
    /// assert_eq!(4.0, diff.re());
    /// assert_eq!(1.0, diff.im());
    /// ```
    fn sub(self, other: Self) -> Self {
        let re = self.cartesian.re - other.cartesian.re;
        let im = self.cartesian.im - other.cartesian.im;

        Self::new_cartesian(re, im)
    }
}

impl Mul for Complex {
    type Output = Self;

    /// Performs the `*` operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use complex_stuff::Complex;
    ///
    /// let c1 = Complex::new_polar(5.0, 2.0);
    /// let c2 = Complex::new_polar(2.0, 1.0);
    ///
    /// let product = c1 * c2;
    ///
    /// assert_eq!(10.0, product.mag());
    /// assert_eq!(3.0, product.ang());
    /// ```
    fn mul(self, other: Self) -> Self {
        let mag = self.polar.mag * other.polar.mag;
        let ang = self.polar.ang + other.polar.ang;

        Self::new_polar(mag, ang)
    }
}

impl Div for Complex {
    type Output = Option<Self>;

    /// Performs the `/` operation.
    ///
    /// Returns `None` when the result is undefined or infinite.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use complex_stuff::Complex;
    ///
    /// let c1 = Complex::new_polar(5.0, 2.0);
    /// let c2 = Complex::new_polar(2.0, 1.0);
    ///
    /// let quotient = (c1 / c2)?;
    ///
    /// assert_eq!(2.5, product.mag());
    /// assert_eq!(1.0, product.ang());
    /// ```
    fn div(self, other: Self) -> Option<Self> {
        if other.polar.mag == 0.0 {
            return None;
        }

        let mag = self.polar.mag / other.polar.mag;
        let ang = self.polar.ang - other.polar.ang;

        Some(Self::new_polar(mag, ang))
    }
}

impl Complex {
    /// Returns the natural logarithm of the complex number.
    ///
    /// Returns `None` when the result is undefined or infinite.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_cartesian(5.0, 3.0);
    ///
    /// let result = complex.ln()?;
    /// ```
    pub fn ln(self) -> Option<Self> {
        if self.polar.mag == 0.0 {
            return None;
        }

        let re = self.polar.mag.ln();
        let im = self.polar.ang;

        Some(Self::new_cartesian(re, im))
    }

    /// Returns the logarithm to any other base of the complex number.
    ///
    /// Returns `None` when the result is undefined or infinite.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_cartesian(5.0, 3.0);
    /// let base = Complex::new_cartesian(7.0, 10.0);
    ///
    /// let result = complex.log(base)?;
    /// ```
    pub fn log(self, other: Self) -> Option<Self> {
        if other.polar.mag == 1.0 || self.polar.mag == 0.0 {
            return None;
        }

        if other.polar.mag == 0.0 {
            return Some(Self::zero());
        }

        let mag_s = self.polar.mag;
        let ang_s = self.polar.ang;

        let mag_o = other.polar.mag;
        let ang_o = other.polar.ang;

        let divisor = mag_o.ln().powi(2) + ang_o.powi(2);

        let re = (mag_s.ln() * mag_o.ln() + ang_s * ang_o) / divisor;
        let im = (ang_s * mag_o.ln() - ang_o * mag_s.ln()) / divisor;

        Some(Self::new_cartesian(re, im))
    }
}

impl Complex {
    /// Raises the complex number to any other complex number and returns the result.
    ///
    /// Returns `None` when the result is undefined or infinite.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_cartesian(5.0, 3.0);
    /// let exponent = Complex::new_cartesian(2.0, 1.0);
    ///
    /// let result = complex.pow(exponent)?;
    /// ```
    pub fn pow(self, other: Self) -> Option<Self> {
        if self.polar.mag == 0.0 && other.polar.mag == 0.0 {
            return None;
        }

        if other.polar.mag == 0.0 {
            return Some(Self::one());
        }

        if self.polar.mag == 0.0 {
            return Some(Self::zero());
        }

        let b_mag = self.polar.mag;
        let b_ang = self.polar.ang;

        let e_re = other.cartesian.re;
        let e_im = other.cartesian.im;

        let mag = (e_re * b_mag.ln() - e_im * b_ang).exp();
        let ang = e_re * b_ang + e_im * b_mag.ln();

        Some(Self::new_polar(mag, ang))
    }

    /// Returns the nth root of the complex number with n being any other complex number.
    ///
    /// Returns `None` when the result is undefined or infinite.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_cartesian(5.0, 3.0);
    /// let n = Complex::new_cartesian(2.0, 1.0);
    ///
    /// let result = complex.root(n)?;
    /// ```
    pub fn root(self, other: Self) -> Option<Self> {
        if other.polar.mag == 0.0 {
            return None;
        }

        if self.polar.mag == 0.0 {
            return Some(Self::zero());
        }

        let b_mag = self.polar.mag;
        let b_ang = self.polar.ang;

        let r_re = other.cartesian.re;
        let r_im = other.cartesian.im;

        let r_sqr_mag = r_re * r_re + r_im * r_im;

        let mag = ((r_re * b_mag.ln() + r_im * b_ang) / r_sqr_mag).exp();
        let ang = (r_re * b_ang - r_im * b_mag.ln()) / r_sqr_mag;

        Some(Self::new_polar(mag, ang))
    }
}

impl Complex {
    /// Returns the sine of the complex number.
    ///
    /// Returns `None` when the result is undefined or infinite.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_cartesian(5.0, 3.0);
    ///
    /// let result = complex.sin()?;
    /// ```
    pub fn sin(self) -> Option<Self> {
        let re = self.cartesian.re.sin() * self.cartesian.im.cosh();
        let im = self.cartesian.re.cos() * self.cartesian.im.sinh();

        Some(Self::new_cartesian(re, im))
    }

    /// Returns the cosine of the complex number.
    ///
    /// Returns `None` when the result is undefined or infinite.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use complex_stuff::Complex;
    ///
    /// let complex = Complex::new_cartesian(5.0, 3.0);
    ///
    /// let result = complex.cos()?;
    /// ```
    pub fn cos(self) -> Option<Self> {
        let re = self.cartesian.re.cos() * self.cartesian.im.cosh();
        let im = -(self.cartesian.re.sin() * self.cartesian.im.sinh());

        Some(Self::new_cartesian(re, im))
    }
}
