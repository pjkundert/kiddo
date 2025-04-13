//! Iterator object returned by within_unsorted_iter
use crate::neighbour::NearestNeighbour;
use crate::traits::Content;
use generator::Generator;

/// Iterator object returned by within_unsorted_iter
pub struct WithinUnsortedIter<'a, A: Copy + std::cmp::PartialOrd, T: Content, const K: usize>(Generator<'a, (), NearestNeighbour<A, T, K>>);

impl<'a, A: Copy + std::cmp::PartialOrd, T: Content, const K: usize> WithinUnsortedIter<'a, A, T, K> {
    pub(crate) fn new(gen: Generator<'a, (), NearestNeighbour<A, T, K>>) -> Self {
        WithinUnsortedIter::<A, T, K>(gen)
    }
}

impl<A: Copy + std::cmp::PartialOrd, T: Content, const K: usize> Iterator for WithinUnsortedIter<'_, A, T, K> {
    type Item = NearestNeighbour<A, T, K>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
