//! Defines different distance metrics, in simplest case it defines the
//! euclidean distance which is no more than the square root of the sum of the
//! squares of the distances in each dimension.  Assumes fixed::... types that
//! do not have +/- inf, and may easily overflow/saturate.

// #[cfg(any(target_arch = "x86_64"))]
// use std::arch::x86_64::*;

use crate::fixed::kdtree::Axis;
use crate::traits::DistanceMetric;

/// Returns the squared euclidean distance between two points. When you only
/// need to compare distances, rather than having the exact distance between
/// the points, this metric is beneficial because it avoids the expensive square
/// root computation.
///
/// # Examples
///
/// ```rust
/// use fixed::types::extra::U0;
/// use fixed::FixedU16;
/// use kiddo::traits::DistanceMetric;
/// use kiddo::fixed::distance::Manhattan;
/// type Fxd = FixedU16<U0>;
///
/// let ZERO = Fxd::from_num(0);
/// let ONE = Fxd::from_num(1);
/// let TWO = Fxd::from_num(2);
///
/// assert_eq!(ZERO, Manhattan::dist(&[ZERO, ZERO], &[ZERO, ZERO]));
/// assert_eq!(ONE, Manhattan::dist(&[ZERO, ZERO], &[ONE, ZERO]));
/// assert_eq!(TWO, Manhattan::dist(&[ZERO, ZERO], &[ONE, ONE]));
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
        let mut result = A::zero();
        for i in 0..K {
            // Use explicit Manhattan::dist1 to avoid type inference issues
            let dist = <Self as DistanceMetric<A, K>>::dist1(a[i], b[i], scale[i]);
            // Use explicit Manhattan::accumulate to avoid type inference issues
            result = <Self as DistanceMetric<A, K>>::accumulate(result, dist);
        }
        result
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
/// # Examples
///
/// ```rust
/// use fixed::types::extra::U0;
/// use fixed::FixedU16;
/// use kiddo::traits::DistanceMetric;
/// use kiddo::fixed::distance::SquaredEuclidean;
/// type Fxd = FixedU16<U0>;
///
/// let ZERO = Fxd::from_num(0);
/// let ONE = Fxd::from_num(1);
/// let TWO = Fxd::from_num(2);
/// let EIGHT = Fxd::from_num(8);
///
/// assert_eq!(SquaredEuclidean::dist(&[ZERO, ZERO], &[ZERO, ZERO]), ZERO);
/// assert_eq!(SquaredEuclidean::dist(&[ZERO, ZERO], &[ONE, ZERO]), ONE);
/// assert_eq!(SquaredEuclidean::dist(&[ZERO, ZERO], &[TWO, TWO]), EIGHT);
/// ```
pub struct SquaredEuclidean {}

impl<A: Axis, const K: usize> DistanceMetric<A, K> for SquaredEuclidean {
    #[inline]
    fn accumulate(acc: A, dist: A) -> A {
	acc.saturating_add(dist.saturating_mul(dist))  // dist^2
    }

    #[inline]
    fn dist(a: &[A; K], b: &[A; K], scale: &[A; K]) -> A {
        let mut result = A::zero();
        for i in 0..K {
            // Use explicit Manhattan::dist1 to avoid type inference issues
            let dist = <Self as DistanceMetric<A, K>>::dist1(a[i], b[i], scale[i]);
            // Use explicit Manhattan::accumulate to avoid type inference issues
            result = <Self as DistanceMetric<A, K>>::accumulate(result, dist);
        }
        result
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
/// # Examples
///
/// ```rust
/// use fixed::types::extra::U0;
/// use fixed::FixedU16;
/// use kiddo::traits::DistanceMetric;
/// use kiddo::fixed::distance::SquaredEuclidean;
/// type Fxd = FixedU16<U0>;
///
/// let ZERO = Fxd::from_num(0);
/// let ONE = Fxd::from_num(1);
/// let TWO = Fxd::from_num(2);
/// let EIGHT = Fxd::from_num(8);
///
/// assert_eq!(Rectangular::dist(&[ZERO, ZERO], &[ZERO, ZERO]), &[ONE, ONE], ZERO);
/// assert_eq!(Rectangular::dist(&[ZERO, ZERO], &[ONE, ZERO]), &[ONE, ONE], ONE);
/// assert_eq!(Rectangular::dist(&[ZERO, ZERO], &[TWO, TWO]), &[ONE, ONE], TWO);
/// ```
pub struct Rectangular {}

impl<A: Axis, const K: usize> DistanceMetric<A, K> for Rectangular {
    #[inline]
    fn accumulate(acc: A, dist: A) -> A {
	if dist > acc {
	    dist
	} else {
	    acc
	}
    }

    #[inline]
    fn dist(a: &[A; K], b: &[A; K], scale: &[A; K]) -> A {
        let mut result = A::zero();
        for i in 0..K {
            // Use explicit Manhattan::dist1 to avoid type inference issues
            let dist = <Self as DistanceMetric<A, K>>::dist1(a[i], b[i], scale[i]);
            // Use explicit Manhattan::accumulate to avoid type inference issues
            result = <Self as DistanceMetric<A, K>>::accumulate(result, dist);
        }
        result
    }

    #[inline]
    fn dist1(a: A, b: A, scale: A) -> A {
	a.saturating_dist(b).saturating_mul(scale)
    }
}
