//! Iterator object returned by within_unsorted_iter
use crate::nearest_neighbour::{NearestNeighbour, NearestNeighbourPoint};
use generator::Generator;

/// Iterator object returned by within_unsorted_iter
pub struct WithinUnsortedIter<'a, A, T, const K: usize>(Generator<'a, (), NearestNeighbour<A, T>>);

impl<'a, A, T, const K: usize> WithinUnsortedIter<'a, A, T, K> {
    pub(crate) fn new(gen: Generator<'a, (), NearestNeighbourPoint<A, T, K>>) -> Self {
        WithinUnsortedIter(gen)
    }
}

impl<A, T, const K: usize> Iterator for WithinUnsortedIter<'_, A, T, K> {
    type Item = NearestNeighbour<A, T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next() {
	    Some(nnp) => Some(nnp.neighbour),
	    None => None,
	}
    }
}


pub struct WithinUnsortedPointIter<'a, A, T, const K: usize>(Generator<'a, (), NearestNeighbourPoint<A, T, K>>);

impl<'a, A, T, const K: usize> WithinUnsortedPointIter<'a, A, T, K> {
    pub(crate) fn new(gen: Generator<'a, (), NearestNeighbourPoint<A, T, K>>) -> Self {
        WithinUnsortedPointIter(gen)
    }
}

impl<A, T, const K: usize> Iterator for WithinUnsortedPointIter<'_, A, T, K> {
    type Item = NearestNeighbourPoint<A, T, K>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
