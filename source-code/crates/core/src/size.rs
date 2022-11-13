use core::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Size {
    pub width: f64,
    pub height: f64,
}

pub type AspectRatio = f64;

impl Size {
    pub const ZERO: Size = Self::new(0.0, 0.0);

    #[inline]
    pub const fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }

    pub fn aspect_ratio(&self) -> AspectRatio {
        self.width / self.height
    }
}

impl<P: Into<f64>> From<(P, P)> for Size {
    #[inline]
    fn from((width, height): (P, P)) -> Self {
        Self::new(width.into(), height.into())
    }
}

impl<P: Into<f64>> From<[P; 2]> for Size {
    #[inline]
    fn from([width, height]: [P; 2]) -> Self {
        Self::new(width.into(), height.into())
    }
}

impl From<Size> for (f64, f64) {
    #[inline]
    fn from(Size { width, height }: Size) -> Self {
        (width, height)
    }
}

impl From<Size> for [f64; 2] {
    #[inline]
    fn from(Size { width, height }: Size) -> Self {
        [width, height]
    }
}

// trait Pixel: Debug + Copy + Into<f64> {}
