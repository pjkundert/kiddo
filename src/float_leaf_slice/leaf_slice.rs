use crate::float::result_collection::ResultCollection;
use az::Cast;
use std::collections::BinaryHeap;
use std::slice::ChunksExact;

use crate::traits::DistanceMetric;
use crate::{float::kdtree::Axis, neighbour::{BestNeighbour, NearestNeighbour}, traits::Content};

const CHUNK_SIZE: usize = 32;

/*#[cfg(all(
    feature = "simd",
    target_feature = "avx2",
    any(target_arch = "x86", target_arch = "x86_64")
))]
use super::f64_avx2::get_best_from_dists_f64_avx2;*/
//use super::{f32_avx2::get_best_from_dists_f32_avx2};

// #[cfg(all(
//     feature = "simd",
//     target_feature = "avx512f",
//     any(target_arch = "x86", target_arch = "x86_64")
// ))]
// use super::f64_avx512::get_best_from_dists_f64_avx512;

#[doc(hidden)]
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct LeafFixedSlice<'a, A: Axis, T: Content, const K: usize, const C: usize> {
    pub content_points: [&'a [A; C]; K],
    pub content_items: &'a [T; C],
}

impl<A, T, const K: usize, const C: usize> LeafFixedSlice<'_, A, T, K, C>
where
    A: Axis + LeafSliceFloatChunk<T, K>,
    T: Content,
    usize: Cast<T>,
{
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn nearest_one<D>(&self, query: &[A; K], scale: &[A; K], best_dist: &mut A, best_item: &mut T)
    where
        D: DistanceMetric<A, K>,
    {
        // Calculate distances for all points in this chunk
        let mut acc = [A::zero(); C];
        
        // For each dimension
        (0..K).step_by(1).for_each(|dim| {
            
            // For each point in the chunk
            (0..C).step_by(1).for_each(|idx| {
                
                // Accumulate distance in this dimension
                acc[idx] = D::accumulate(acc[idx], D::dist1(
		    
                    self.content_points[dim][idx], query[dim], scale[dim]
		));
            
            });
        
        });

        
        // Update the best distance and item if we found a better one
        A::update_nearest_dist(acc, self.content_items, best_dist, best_item);
    
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub(crate) struct LeafSlice<'a, A: Axis, T: Content, const K: usize> {
    pub content_points: [&'a [A]; K],
    pub content_items: &'a [T],
}

impl<A: Axis, T: Content, const K: usize> LeafSlice<'_, A, T, K> {
    #[allow(dead_code)]
    #[inline]
    fn len(&self) -> usize {
        self.content_items.len()
    }
}

pub(crate) struct LeafFixedSliceIterator<'a, A: Axis, T: Content, const K: usize, const C: usize> {
    points_iterators: [ChunksExact<'a, A>; K],
    items_iterator: ChunksExact<'a, T>,
}

impl<'a, A: Axis, T: Content, const K: usize, const C: usize> Iterator
    for LeafFixedSliceIterator<'a, A, T, K, C>
{
    type Item = ([&'a [A; C]; K], &'a [T; C]);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.items_iterator.len() == 0 {
            None
        } else {
            let points_chunk: [&[A; C]; K] = array_init::array_init(|i| {
                self.points_iterators[i].next().unwrap().try_into().unwrap()
            });
            let item_chunk: &[T; C] = self.items_iterator.next().unwrap().try_into().unwrap();
            Some((points_chunk, item_chunk))
        }
    }
}

impl<'a, A: Axis, T: Content, const K: usize, const C: usize>
    LeafFixedSliceIterator<'a, A, T, K, C>
{
    #[inline]
    fn remainder(&self) -> ([&'a [A]; K], &'a [T]) {
        (
            array_init::array_init(|i| self.points_iterators[i].remainder()),
            self.items_iterator.remainder(),
        )
    }
}

pub trait LeafSliceFloatChunk<T, const K: usize>
where
    T: Content,
{
    fn results_for_chunk<D, const C: usize>(points: [&[Self; C]; K], items: &[T; C], query: &[Self; K], scale: &[Self; K], include: F, &mut results: R)
    where
        Self: Sized,
        D: DistanceMetric<Self, K>,
	F: Fn(A, T, &mut R),
	N: NeighbourEntry<A, T>,
	R: ResultCollection<N, A, T, K>;
    fn results_for_remainder<D, const C: usize>(points: [&[Self]; K], items: &[T], query: &[Self; K], scale: &[Self; K], include: F, &mut results: R)
    where
        Self: Sized,
        D: DistanceMetric<Self, K>,
	F: Fn(A, T, &mut R),
	N: NeighbourEntry<A, T>,
	R: ResultCollection<N, A, T, K>;
    
}

impl<A, T, const K: usize> LeafSlice<'_, A, T, K>
where
    A: Axis + LeafSliceFloatChunk<T, K>,
    T: Content,
    usize: Cast<T>,
{
    #[inline]
    pub(crate) fn new<'a>(
        content_points: [&'a [A]; K],
        content_items: &'a [T],
    ) -> LeafSlice<'a, A, T, K> {
        let size = content_items.len();
        for arr in content_points {
            debug_assert_eq!(arr.len(), size);
        }

        LeafSlice {
            content_items,
            content_points,
        }
    }

    #[inline]
    fn as_full_chunks<const C: usize>(&self) -> LeafFixedSliceIterator<A, T, K, C> {
        let points_iterators = self.content_points.map(|i| i.chunks_exact(C));
        let items_iterator = self.content_items.chunks_exact(C);

        LeafFixedSliceIterator {
            items_iterator,
            points_iterators,
        }
    }

    #[inline]
    pub(crate) fn nearest_one<D>(&self, query: &[A; K], scale: &[A; K], best_dist: &mut A, best_item: &mut T, best_point: &mut [A; K])
    where
        D: DistanceMetric<A, K>,
    {
        let chunks_iter = self.as_full_chunks::<CHUNK_SIZE>();
        let (remain_points, remain_items) = chunks_iter.remainder();

	let mut nearest: Option<NearestNeighbourPoint<A, T, K>> = None;
	fn one(distance: A, _item: T, &mut _results: R) -> bool {
	    match results {
		Some(nnp) => distance < nnp.distance(),
		None => true,
	    }
	};

        for (chunks_points, chunks_items) in chunks_iter {
	    A::results_for_chunk::<D, CHUNK_SIZE>(chunks_points, chunks_items, query, scale, one, &mut nearest);
        }
	A::results_for_remainder::<D, CHUNK_SIZE>(remain_points, remain_items, query, scale, one, &mut nearest);

	match nearest {
	    Some(nnp) => {
		best_dist = nnp.distance();
		best_item = nnp.item();
		best_point.copy_from_slice(nnp.point);
	    },
	    None => {},
	}
    }

    #[inline]
    pub(crate) fn nearest_n_within<D, R>(&self, query: &[A; K], scale: &[A; K], radius: A, results: &mut R)
    where
        D: DistanceMetric<A, K>,
        R: ResultCollection<NearestNeighbour<A, T>, A, T, K>,
    {
        let chunks_iter = self.as_full_chunks::<CHUNK_SIZE>();
        let (remaind_points, remaind_items) = chunks_iter.remainder();
        
	fn within(distance: A, _item: T, &mut _results: R) -> bool {
	    distance < radius // TODO: should be <= ?
	};

        for (chunks_points, chunks_item) in chunk_iter {
	    A::results_for_chunk::<D, CHUNK_SIZE>(chunks_points, chunks_items, query, scale, within, &mut results);
	}
	A::results_for_remainder::<D, CHUNK_SIZE>(remain_points, remain_items, query, scale, within, &mut results);
    }

    // Any sorted ResultCollection container w/ peek() and pop() should work.
    #[inline]
    pub(crate) fn best_n_within<D>(&self, query: &[A; K], scale: &[A; K], radius: A, max_qty: usize, results: &mut R ) where
        D: DistanceMetric<A, K>,
	R: ResultCollection<BestNeighbour<A, T>, A, T, K>  // eg. BinaryHeap<BestNeighbourPoint<A, T, K>>
    {
        let chunks_iter = self.as_full_chunks::<CHUNK_SIZE>();
        let (remain_points, remain_items) = chunks_iter.remainder();

	// Adjusts the results if it's full but this item is better.
        fn n_within(distance: A, item: T, &mut results: R) -> bool {
            if distance <= radius {
		if results.len() < max_qty {
		    return true;
		}
                if item < results.peek().unwrap().0.item {
		    // Remove the worst (greatest) item, if ours is better (less)
		    results.pop();
		    return true;
		}
	    };
	    false
	}

	for (chunks_points, chunks_item) in chunks_iter {
	    A::results_for_chunk::<D, CHUNK_SIZE>(chunks_points, chunks_items, query, scale, n_within, &mut results);
	}
	A::results_for_remainder::<D, CHUNK_SIZE>(remain_points, remain_items, query, scale, n_within, &mut results);
    }
}


impl<T: Content, const K: usize> LeafSliceFloatChunk<T, K> for f64
where
    T: Content,
    usize: Cast<T>,
{
    // Scan a chunk in SIMD- and cache-friendly fashion.  Then, evaluate the resultant dists
    // according to the provided evaluation function, and add {Nearest,Best}NeighbourPoint records
    // to the ResultCollection.  
    #[inline]
    fn results_for_chunk<D, F, const C: usize>(points: [&[Self; C]; K], items: &[T; C], query: &[Self; K], scale: &[Self; K], include: F, &mut results: R)
    where
	Self: Sized,
        D: DistanceMetric<Self, K>,
	F: Fn(A, T, &mut R) -> bool,
	N: NeighbourEntry<A, T>,  // {Nearest,Best}Neighbour
	R: ResultCollection<N, A, T, K>
    {
        // AVX512: 4 loops of 32 iterations, each 4x unrolled, 5 instructions per pre-unrolled iteration
        let mut acc = [0f64; C];
        (0..K).step_by(1).for_each(|dim| {
            (0..C).step_by(1).for_each(|idx| {
		acc[idx] = D::accumulate(acc[idx], D::dist1(points[dim][idx], query[dim], scale[dim]));
            });
        });

	// Iterate each computed distance, evaluating each for inclusion in the results.
	(0..C).step_by(1).for_each(|idx| {
	    if include(acc[idx], items[idx], results) {
		let neighbour = N::new(acc[idx], items[idx]);
		let mut point = [A; K];
		for dim in (0..K) {
		    point[dim] = points[dim][idx];
		}
		let neighbour_point = NeighbourPoint::<N, A, T, K>::new( neighbour, point );
		results.add( neighbour_point );
	    }
	});
    }

    #[inline]
    fn results_for_remainder<D, F, const C: usize>(points: [&[Self]; K], items: &[T], query: &[Self; K], scale: &[Self; K], include: F, &mut results: ResultCollection<N, A, T, K>)
    where
	Self: Sized,
        D: DistanceMetric<Self, K>,
	F: Fn(A, T, &mut R) -> bool,
	N: NeighbourEntry<A, T>,  // {Nearest,Best}Neighbour
	R: ResultCollection<N, A, T, K>,
    {
        let mut acc = [0f64; C];  // Will be something less than C
        (0..K).step_by(1).for_each(|dim| {
            (0..points[dim].len()).step_by(1).for_each(|idx| {
		acc[idx] = D::accumulate(acc[idx], D::dist1(points[dim][idx], query[dim], scale[dim]));
            });
        });

	// Iterate each computed distance, evaluating each for inclusion in the results.
	(0..points[0].len()).step_by(1).for_each(|idx| {
	    if include(acc[idx], items[idx], results) {
		let neighbour = N::new(acc[idx], items[idx]);
		let mut point = [A; K];
		for dim in (0..K) {
		    point[dim] = points[dim][idx];
		}
		let neighbour_point = NeighbourPoint::<N, A, T, K>::new( neighbour, point );
		results.add( neighbour_point );
	    }
	});
    }
}

#[cfg(test)]
mod test {
    use crate::float_leaf_slice::leaf_slice::{LeafFixedSlice, LeafSliceFloat};
    use crate::{BestNeighbour, NearestNeighbour, SquaredEuclidean};
    use std::collections::BinaryHeap;

    #[test]
    fn leaf_fixed_slice_nearest_one_works() {
        let content_points = [
            [0.0f64, 3.0f64, 5.0f64, 0.0f64],
            [0.0f64, 4.0f64, 12.0f64, 0.0f64],
        ];

        let content_items = [1u32, 2u32, 3u32, 4u32];

        let slice = LeafFixedSlice {
            content_points: [&content_points[0], &content_points[1]],
            content_items: &content_items,
        };

        let mut best_dist = f64::INFINITY;
        let mut best_item = u32::MAX;

        slice.nearest_one::<SquaredEuclidean>(&[0.0f64, 0.0f64], &mut best_dist, &mut best_item);

        assert_eq!(best_dist, 0f64);
        assert_eq!(best_item, 1u32);
    }
}
