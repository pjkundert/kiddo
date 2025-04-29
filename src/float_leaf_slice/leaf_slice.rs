use crate::float::result_collection::ResultCollection;
use az::Cast;
use std::slice::ChunksExact;

use crate::traits::DistanceMetric;
use crate::{float::kdtree::Axis, neighbour::{NearestNeighbour, BestNeighbour, NeighbourEntry}, traits::Content};

const CHUNK_SIZE: usize = 32;

/// LeafFixedSlice for Leaf slice w/ fixed length C
#[doc(hidden)]
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct LeafFixedSlice<'a, A: Axis, T: Content, const K: usize, const C: usize> {
    pub content_points: [&'a [A; C]; K],
    pub content_items: &'a [T; C],
}

#[doc(hidden)]
#[derive(Debug)]
pub(crate) struct LeafSlice<'a, A: Axis, T: Content, const K: usize> {
    pub content_points: [&'a [A]; K],
    pub content_items: &'a [T],
}

/// LeafFixedSlice for Leaf slice w/ arbitrary length
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

/// Trait for processing points in chunks or as remainder
pub trait LeafSliceFloatChunk<T, const K: usize>
where
    T: Content,
    Self: Sized + Axis,
{
    /// Process a fixed-size chunk of points generically for any NeighbourEntry type
    fn results_for_chunk<D, F, R, N, const C: usize>(points: [&[Self; C]; K], items: &[T; C], query: &[Self; K], scale: Option<&[Self; K]>, include: F, results: &mut R) 
    where
        D: DistanceMetric<Self, K>,
        F: Fn(Self, T, &mut R) -> bool,
        N: NeighbourEntry<Self, T, K>,
        R: ResultCollection<N, Self, T, K>;
        
    /// Process points that don't fit into a full chunk generically for any NeighbourEntry type
    fn results_for_remainder<D, F, R, N, const C: usize>(points: [&[Self]; K], items: &[T], query: &[Self; K], scale: Option<&[Self; K]>, include: F, results: &mut R)
    where
        D: DistanceMetric<Self, K>,
        F: Fn(Self, T, &mut R) -> bool,
        N: NeighbourEntry<Self, T, K>,
        R: ResultCollection<N, Self, T, K>;
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
    pub(crate) fn nearest_one<D>(&self, query: &[A; K], scale: Option<&[A; K]>, best_dist: &mut A, best_item: &mut T, best_point: &mut [A; K])
    where
        D: DistanceMetric<A, K>,
    {
        let chunks_iter = self.as_full_chunks::<CHUNK_SIZE>();
        let (remain_points, remain_items) = chunks_iter.remainder();

        let mut nearest: Option<NearestNeighbour<A, T, K>> = None;
        let is_nearer = |distance: A, _item: T, results: &mut Option<NearestNeighbour<A, T, K>>| -> bool {
            match results {
                Some(n) => distance < n.0.distance,
                None => true,
            }
        };

        for (chunks_points, chunks_items) in chunks_iter {
            A::results_for_chunk::<D, _, _, NearestNeighbour<A, T, K>, CHUNK_SIZE>(chunks_points, chunks_items, query, scale, is_nearer, &mut nearest);
        }
        A::results_for_remainder::<D, _, _, NearestNeighbour<A, T, K>, CHUNK_SIZE>(remain_points, remain_items, query, scale, is_nearer, &mut nearest);

        match nearest {
            Some(nnp) => (*best_dist, *best_item, *best_point) = nnp.into(),
            None => {},
        }
    }

    /// Find all neighbors within a specified radius
    #[inline]
    pub(crate) fn nearest_n_within<D, R>(&self, query: &[A; K], radius: A, scale: Option<&[A; K]>, results: &mut R)
    where
        D: DistanceMetric<A, K>,
        R: ResultCollection<NearestNeighbour<A, T, K>, A, T, K>,
    {
        // Function to check if this distance is within radius
        let within = |distance: A, _item: T, _results: &mut R| -> bool {
            distance <= radius 
        };

        let chunks_iter = self.as_full_chunks::<CHUNK_SIZE>();
        let (remain_points, remain_items) = chunks_iter.remainder();

        for (chunks_points, chunks_items) in chunks_iter {
            A::results_for_chunk::<D, _, _, NearestNeighbour<A, T, K>, CHUNK_SIZE>(chunks_points, chunks_items, query, scale, within, results);
        }
        A::results_for_remainder::<D, _, _, NearestNeighbour<A, T, K>, CHUNK_SIZE>(remain_points, remain_items, query, scale, within, results);
    }

    /// Find the best N neighbors within a radius, keeping only the best items up to max_qty
    /// Any ordered ResultCollection container with peek() and pop() should work.
    #[inline]
    pub(crate) fn best_n_within<D, R>(&self, query: &[A; K], radius: A, max_qty: usize, scale: Option<&[A; K]>, results: &mut R)
    where
        D: DistanceMetric<A, K>,
        R: ResultCollection<BestNeighbour<A, T, K>, A, T, K>,
    {
        // Function to check if this distance is within radius *and* better than worst entry (if full)
        let n_within = |distance: A, item: T, results: &mut R| -> bool {
            if distance <= radius {
                if results.result_len() < max_qty {
                    return true;
                }
                if let Some(worst) = results.result_peek() {
                    if item < worst.0.item {
                        // Remove the worst (greatest) item, if ours is better (less)
                        results.result_pop();
                        return true;
                    }
                }
            }
            false
        };

        let chunks_iter = self.as_full_chunks::<CHUNK_SIZE>();
        let (remain_points, remain_items) = chunks_iter.remainder();

        for (chunks_points, chunks_items) in chunks_iter {
            A::results_for_chunk::<D, _, _, BestNeighbour<A, T, K>, CHUNK_SIZE>(chunks_points, chunks_items, query, scale, n_within, results);
        }
        A::results_for_remainder::<D, _, _, BestNeighbour<A, T, K>, CHUNK_SIZE>(remain_points, remain_items, query, scale, n_within, results);
    }
}


impl<A, T, const K: usize> LeafSliceFloatChunk<T, K> for A
where
    A: Axis,
    T: Content,
    usize: Cast<T>,
{
    /// Scan a chunk in SIMD- and cache-friendly fashion for NearestNeighbour results.
    /// 
    /// This method is optimized for fixed-size chunks and processes data in a way that's friendly for
    /// modern CPU caches and potential SIMD optimizations.
    #[inline]
    fn results_for_chunk<D, F, R, N, const C: usize>(points: [&[Self; C]; K], items: &[T; C], query: &[Self; K], scale: Option<&[Self; K]>, include: F, results: &mut R)
    where
        D: DistanceMetric<Self, K>,
        F: Fn(Self, T, &mut R) -> bool,
        N: NeighbourEntry<Self, T, K>,
        R: ResultCollection<N, Self, T, K>
    {
        // Calculate distances for all points in the chunk
        let mut acc = [Self::zero(); C];
        
        // For each dimension
        (0..K).step_by(1).for_each(|dim| {
            // For each point in the chunk
            (0..C).step_by(1).for_each(|idx| {
                // Accumulate distance in this dimension
                acc[idx] = D::accumulate(acc[idx], D::dist1(points[dim][idx], query[dim], scale.map(|s| s[dim])));
            });
        });

        // Iterate each computed distance, evaluating each for inclusion in the results
        (0..C).step_by(1).for_each(|idx| {
            if include(acc[idx], items[idx], results) {
                let mut point = [Self::zero(); K];
                for dim in 0..K {
                    point[dim] = points[dim][idx];
                }
                let neighbour = N::new(acc[idx], items[idx], point);
                results.add(neighbour);
            }
        });
    }

    /// Process points that don't fit into a full chunk for NearestNeighbour results.
    ///
    /// This handles either leftover data after chunking or data that can't be chunked at all.
    #[inline]
    fn results_for_remainder<D, F, R, N, const C: usize>(points: [&[Self]; K], items: &[T], query: &[Self; K], scale: Option<&[Self; K]>, include: F, results: &mut R)
    where
        D: DistanceMetric<Self, K>,
        F: Fn(Self, T, &mut R) -> bool,
        N: NeighbourEntry<Self, T, K>,
        R: ResultCollection<N, Self, T, K>
    {
        // Handle variable-sized or remainder data
        if points[0].is_empty() {
            return; // Nothing to process
        }

        let len = points[0].len();
        let mut acc = vec![Self::zero(); len];
        
        // For each dimension
        (0..K).step_by(1).for_each(|dim| {
            // For each point
            (0..len).step_by(1).for_each(|idx| {
                // Accumulate distance in this dimension
                acc[idx] = D::accumulate(acc[idx], D::dist1(points[dim][idx], query[dim], scale.map(|s| s[dim])));
            });
        });

        // Iterate each computed distance, evaluating each for inclusion in the results
        (0..len).step_by(1).for_each(|idx| {
            if include(acc[idx], items[idx], results) {
                let mut point = [Self::zero(); K];
                for dim in 0..K {
                    point[dim] = points[dim][idx];
                }
                let neighbour = N::new(acc[idx], items[idx], point);
                results.add(neighbour);
            }
        });
    }
}

#[cfg(test)]
mod test {
    use crate::float_leaf_slice::leaf_slice::LeafSlice;
    use crate::SquaredEuclidean;

    #[test]
    fn leaf_fixed_slice_nearest_one_works() {
        let content_points = [
            [  0.0f64,  3.0f64,  5.0f64,  0.0f64],
            [  0.0f64,  4.0f64, 12.0f64,  0.0f64],
            [ -1.0f64,  4.0f64, 12.0f64,  0.0f64],
            [  0.0f64, -3.0f64,  5.0f64,  1.0f64],
        ];

        let dim0: [f64; 4] = [content_points[0][0], content_points[1][0], content_points[2][0], content_points[3][0]];
        let dim1: [f64; 4] = [content_points[0][1], content_points[1][1], content_points[2][1], content_points[3][1]];
        let dim2: [f64; 4] = [content_points[0][2], content_points[1][2], content_points[2][2], content_points[3][2]];
        let dim3: [f64; 4] = [content_points[0][3], content_points[1][3], content_points[2][3], content_points[3][3]];

        let content_items = [1u32, 2u32, 3u32, 4u32];

        let slice = LeafSlice {
            content_points: [&dim0, &dim1, &dim2, &dim3],
            content_items: &content_items,
        };

        let mut best_dist = f64::INFINITY;
        let mut best_item = u32::MAX;

        // Create a dummy point to satisfy the API
        let mut best_point = [0.0f64; 4];
        slice.nearest_one::<SquaredEuclidean>(
	    &[0f64, 0f64, 0f64, 0f64],
	    None,
	    &mut best_dist,
	    &mut best_item,
	    &mut best_point
	);

        assert_eq!((best_dist, best_item, best_point), (34f64, 1u32, content_points[0]));
    }
}
