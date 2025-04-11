//! Iterator object returned by within_unsorted_iter
use crate::nearest_neighbour::{NearestNeighbour, NearestNeighbourPoint};
use crate::traits::Content;
use generator::Generator;

/// Iterator object returned by within_unsorted_iter
pub struct WithinUnsortedIter<'a, A: Copy + std::cmp::PartialOrd, T: Content, const K: usize>(Generator<'a, (), NearestNeighbourPoint<A, T, K>>);

impl<'a, A: Copy + std::cmp::PartialOrd, T: Content, const K: usize> WithinUnsortedIter<'a, A, T, K> {
    pub(crate) fn new(gen: Generator<'a, (), NearestNeighbourPoint<A, T, K>>) -> Self {
        WithinUnsortedIter::<A, T, K>(gen)
    }
}

impl<A: Copy + std::cmp::PartialOrd, T: Content, const K: usize> Iterator for WithinUnsortedIter<'_, A, T, K> {
    type Item = NearestNeighbour<A, T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next() {
            // Use explicit type to help the compiler with the inference
            Some(nnp) => Some(nnp.neighbour),
            None => None,
        }
    }
}


pub struct WithinUnsortedPointIter<'a, A: Copy + std::cmp::PartialOrd, T: Content, const K: usize>(Generator<'a, (), NearestNeighbourPoint<A, T, K>>);

impl<'a, A: Copy + std::cmp::PartialOrd, T: Content, const K: usize> WithinUnsortedPointIter<'a, A, T, K> {
    pub(crate) fn new(gen: Generator<'a, (), NearestNeighbourPoint<A, T, K>>) -> Self {
        WithinUnsortedPointIter::<A, T, K>(gen)
    }
}

impl<A: Copy + std::cmp::PartialOrd, T: Content, const K: usize> Iterator for WithinUnsortedPointIter<'_, A, T, K> {
    type Item = NearestNeighbourPoint<A, T, K>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
