//! Contains a selection of distance metrics that can be chosen from to measure the distance between
//! two points stored inside the tree.  Assumes types that have NaN, and saturate to +/- inf.

// #[cfg(any(target_arch = "x86_64"))]
// use std::arch::x86_64::*;
use crate::float::kdtree::Axis;
use crate::traits::DistanceMetric;


/// Returns the Manhattan / "taxi cab" distance between two points.
///
/// Faster than squared Euclidean, and just as effective if not more so in higher-dimensional spaces
///
/// re-exported as `kiddo::Manhattan` for convenience
///
/// # Examples
///
/// ```rust
/// use kiddo::traits::DistanceMetric;
/// use kiddo::Manhattan;
///
/// assert_eq!(0f32, Manhattan::dist(&[0f32, 0f32], &[0f32, 0f32], &[1f32, 1f32]));
/// assert_eq!(1f32, Manhattan::dist(&[0f32, 0f32], &[1f32, 0f32], &[1f32, 1f32]));
/// assert_eq!(2f32, Manhattan::dist(&[0f32, 0f32], &[1f32, 1f32], &[1f32, 1f32]));
/// ```
pub struct Manhattan {}

impl<A: Axis, const K: usize> DistanceMetric<A, K> for Manhattan {
    // The default implementation is a Manhattan distance metric
    #[inline]
    fn accumulate(acc: A, dist: A) -> A {
	acc.saturating_add(dist)
    }

    #[inline]
    fn dist(a: &[A; K], b: &[A; K], scale: &[A; K]) -> A {
	(0..K)
	    .map(|i| <Self as DistanceMetric<A, K>>::dist1( a[i], b[i], scale[i]))
	    .fold::<A, _>(A::zero(), <Self as DistanceMetric<A, K>>::accumulate)
    }

    #[inline]
    fn dist1(a: A, b: A, scale: A) -> A {
	a.saturating_dist(b).saturating_mul(scale)
    }
}

/// Returns the squared euclidean distance between two points.
///
/// Faster than Euclidean distance due to not needing a square root, but still
/// preserves the same distance ordering as with Euclidean distance.
///
/// re-exported as `kiddo::SquaredEuclidean` for convenience
///
/// # Examples
///
/// ```rust
/// use kiddo::traits::DistanceMetric;
/// use kiddo::SquaredEuclidean;
///
/// assert_eq!(0f32, SquaredEuclidean::dist(&[0f32, 0f32], &[0f32, 0f32], &[1f32, 1f32]));
/// assert_eq!(1f32, SquaredEuclidean::dist(&[0f32, 0f32], &[1f32, 0f32], &[1f32, 1f32]));
/// assert_eq!(2f32, SquaredEuclidean::dist(&[0f32, 0f32], &[1f32, 1f32], &[1f32, 1f32]));
/// ```
pub struct SquaredEuclidean {}

impl<A: Axis, const K: usize> DistanceMetric<A, K> for SquaredEuclidean {
    #[inline]
    fn accumulate(acc: A, dist: A) -> A {
	acc.saturating_add(dist.saturating_mul(dist))
    }

    #[inline]
    fn dist(a: &[A; K], b: &[A; K], scale: &[A; K]) -> A {
	(0..K)
	    .map(|i| <Self as DistanceMetric<A, K>>::dist1( a[i], b[i], scale[i]))
	    .fold::<A, _>(A::zero(), <Self as DistanceMetric<A, K>>::accumulate)
    }

    #[inline]
    fn dist1(a: A, b: A, scale: A) -> A {
	a.saturating_dist(b).saturating_mul(scale)
    }
}

/// Returns the maximal axis distance between two points.
///
/// Implements a rectangular region with different ranges on each axis.
///
/// re-exported as `kiddo::Rectangular` for convenience
///
/// # Examples
///
/// ```rust
/// use kiddo::traits::DistanceMetric;
/// use kiddo::Rectangular;
///
/// assert_eq!(0f32, Rectangular::dist(&[0f32, 0f32], &[0f32, 0f32], &[1f32, 1f32]));
/// assert_eq!(1f32, Rectangular::dist(&[0f32, 0f32], &[1f32, 0f32], &[1f32, 1f32]));
/// assert_eq!(1f32, Rectangular::dist(&[0f32, 0f32], &[1f32, 1f32], &[1f32, 1f32]));
/// ```
pub struct Rectangular {}

impl<A: Axis, const K: usize> DistanceMetric<A, K> for Rectangular {
    /// Implements a simple max calculation; assumes the default dist1 is an absolute distance metric
    #[inline]
    fn accumulate(acc: A, dist: A) -> A {
	if dist > acc {
	    dist
	} else {
	    acc  // NaN doesn't affect accumulation (and should be avoided by dist1 anyway)
	}
    }

    #[inline]
    fn dist(a: &[A; K], b: &[A; K], scale: &[A; K]) -> A {
	(0..K)
	    .map(|i| <Self as DistanceMetric<A, K>>::dist1( a[i], b[i], scale[i]))
	    .fold::<A, _>(A::zero(), <Self as DistanceMetric<A, K>>::accumulate)
    }

    #[inline]
    fn dist1(a: A, b: A, scale: A) -> A {
	a.saturating_dist(b).saturating_mul(scale)
    }
}
