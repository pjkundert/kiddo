//! Consistent interface for storing 0, 1 or more NearestNeighbourPoints.

use crate::float::kdtree::Axis;
use crate::nearest_neighbour::{NearestNeighbour, NearestNeighbourPoint};
use crate::traits::Content;
use sorted_vec::SortedVec;
use std::collections::BinaryHeap;

pub trait ResultCollection<A: Axis, T: Content, const K: usize> {
    fn new_with_capacity(capacity: usize) -> Self;
    fn add(&mut self, entry: NearestNeighbourPoint<A, T, K>);
    fn max_dist(&self) -> A;
    fn into_vec(self) -> Vec<NearestNeighbour<A, T>>;
    fn into_sorted_vec(self) -> Vec<NearestNeighbour<A, T>>;
    fn into_vec_points(self) -> Vec<NearestNeighbourPoint<A, T, K>>;
    fn into_sorted_vec_points(self) -> Vec<NearestNeighbourPoint<A, T, K>>;
}

impl<A: Axis, T: Content, const K: usize> ResultCollection<A, T, K> for BinaryHeap<NearestNeighbourPoint<A, T, K>> {
    fn new_with_capacity(capacity: usize) -> Self {
        BinaryHeap::with_capacity(capacity)
    }
    fn add(&mut self, entry: NearestNeighbourPoint<A, T, K>) {
        let k = self.capacity();
        if self.len() < k {
            self.push(entry);
        } else {
            let mut max_heap_value = self.peek_mut().unwrap();
            if entry < *max_heap_value {
                *max_heap_value = entry;
            }
        }
    }
    fn max_dist(&self) -> A {
        if self.len() < self.capacity() {
            A::infinity()
        } else {
            self.peek().map_or(A::infinity(), |n| n.neighbour.distance)
        }
    }
    fn into_vec(self) -> Vec<NearestNeighbour<A, T>> {
        BinaryHeap::into_vec(self).iter().map(|nnp| nnp.neighbour).collect()
    }
    fn into_sorted_vec(self) -> Vec<NearestNeighbour<A, T>> {
        BinaryHeap::into_sorted_vec(self).iter().map(|nnp| nnp.neighbour).collect()
    }
    fn into_vec_points(self) -> Vec<NearestNeighbourPoint<A, T, K>> {
        BinaryHeap::into_vec(self)
    }
    fn into_sorted_vec_points(self) -> Vec<NearestNeighbourPoint<A, T, K>> {
        BinaryHeap::into_sorted_vec(self)
    }
}

impl<A: Axis, T: Content, const K: usize> ResultCollection<A, T, K> for Vec<NearestNeighbourPoint<A, T, K>> {
    fn new_with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity)
    }
    fn add(&mut self, entry: NearestNeighbourPoint<A, T, K>) {
        self.push(entry)
    }
    fn max_dist(&self) -> A {
        A::infinity()
    }
    fn into_vec(self) -> Vec<NearestNeighbour<A, T>> {
        self.iter().map(|nnp| nnp.neighbour).collect()
    }
    fn into_sorted_vec(self) -> Vec<NearestNeighbour<A, T>> {
        self.iter().map(|nnp| nnp.neighbour).collect()
    }
    fn into_vec_points(self) -> Vec<NearestNeighbourPoint<A, T, K>> {
        self
    }
    fn into_sorted_vec_points(mut self) -> Vec<NearestNeighbourPoint<A, T, K>> {
        self.sort();
        self
    }
}

impl<A: Axis, T: Content, const K: usize> ResultCollection<A, T, K> for SortedVec<NearestNeighbourPoint<A, T, K>> {
    fn new_with_capacity(capacity: usize) -> Self {
        SortedVec::with_capacity(capacity)
    }
    fn add(&mut self, entry: NearestNeighbourPoint<A, T, K>) {
        let len = self.len();
        if len < self.capacity() {
            self.insert(entry);
        } else if entry < *self.last().unwrap() {
            self.pop();
            self.push(entry);
        }
    }
    fn max_dist(&self) -> A {
        if self.len() < self.capacity() {
            A::infinity()
        } else {
            self.last().map_or(A::infinity(), |n| n.neighbour.distance)
        }
    }
    fn into_vec(self) -> Vec<NearestNeighbour<A, T>> {
        self.iter().map(|nnp| nnp.neighbour).collect()
    }
    fn into_sorted_vec(self) -> Vec<NearestNeighbour<A, T>> {
        self.iter().map(|nnp| nnp.neighbour).collect()
    }
    fn into_vec_points(self) -> Vec<NearestNeighbourPoint<A, T, K>> {
        self.into_vec()
    }
    fn into_sorted_vec_points(self) -> Vec<NearestNeighbourPoint<A, T, K>> {
        self.into_vec()
    }
}

impl<A: Axis, T: Content, const K: usize> ResultCollection<A, T, K> for Option<NearestNeighbourPoint<A, T, K>> {
    fn new_with_capacity(capacity: usize) -> Self {
        assert_eq!(capacity, 1)
    }
    fn add(&mut self, entry: NearestNeighbourPoint<A, T, K>) {
	self = Some(entry);
    }
    fn max_dist(&self) -> A {
        match self {
	    Some(nnp) => nnp.neighbour.distance,
            None => A::infinity(),
        }
    }
    fn into_vec(self) -> Vec<NearestNeighbour<A, T>> {
	match self {
	    Some(nnp) => vec![nnp.neighbour],
	    None => vec![],
	}
    }
    fn into_sorted_vec(self) -> Vec<NearestNeighbour<A, T>> {
	match self {
	    Some(nnp) => vec![nnp.neighbour],
	    None => vec![],
	}
    }
    fn into_vec_points(self) -> Vec<NearestNeighbourPoint<A, T, K>> {
	match self {
	    Some(nnp) => vec![nnp],
	    None => vec![],
	}
    }
    fn into_sorted_vec_points(self) -> Vec<NearestNeighbourPoint<A, T, K>> {
	match self {
	    Some(nnp) => vec![nnp],
	    None => vec![],
	}
    }
}
