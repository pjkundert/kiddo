//! Consistent interface for storing 0, 1 or more neighbor entry points.
//!
//! Capable of storing and ordering {Best,Nearest}NeighbourPoint implementations

use crate::float::kdtree::Axis;
use crate::neighbour::{NeighbourEntry, NeighbourPoint};
use crate::traits::Content;
use sorted_vec::SortedVec;
use std::collections::BinaryHeap;

pub trait ResultCollection<N, A, T, const K: usize>
where
    N: NeighbourEntry<A, T>,
    T: Content,
    A: Axis,
{
    fn new_with_capacity(capacity: usize) -> Self;
    fn result_len(&self) -> usize;
    fn result_pop(&mut self) -> Option<NeighbourPoint<N, A, T, K>>;
    fn result_peek(&self) -> Option<&NeighbourPoint<N, A, T, K>>;
    fn add(&mut self, entry: NeighbourPoint<N, A, T, K>);
    fn max_dist(&self) -> A;
    fn into_vec(self) -> Vec<N>;
    fn into_sorted_vec(self) -> Vec<N>;
    fn into_vec_points(self) -> Vec<NeighbourPoint<N, A, T, K>>;
    fn into_sorted_vec_points(self) -> Vec<NeighbourPoint<N, A, T, K>>;
}


impl<N, A, T, const K: usize> ResultCollection<N, A, T, K> for BinaryHeap<NeighbourPoint<N, A, T, K>>
where
    N: NeighbourEntry<A, T>,
    T: Content,
    A: Axis,
    NeighbourPoint<N, A, T, K>: Ord,
{
    fn new_with_capacity(capacity: usize) -> Self {
        BinaryHeap::with_capacity(capacity)
    }
    
    fn result_len(&self) -> usize {
        self.len()
    }

    fn result_pop(&mut self) -> Option<NeighbourPoint<N, A, T, K>> {
        self.pop()
    }

    fn result_peek(&self) -> Option<&NeighbourPoint<N, A, T, K>> {
        self.peek()
    }

    fn add(&mut self, entry: NeighbourPoint<N, A, T, K>) {
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
            self.peek().map_or(A::infinity(), |n| n.neighbour.distance())
        }
    }
    
    fn into_vec(self) -> Vec<N> {
        BinaryHeap::into_vec(self).into_iter().map(|np| np.neighbour).collect()
    }
    
    fn into_sorted_vec(self) -> Vec<N> {
        BinaryHeap::into_sorted_vec(self).into_iter().map(|np| np.neighbour).collect()
    }
    
    fn into_vec_points(self) -> Vec<NeighbourPoint<N, A, T, K>> {
        BinaryHeap::into_vec(self)
    }
    
    fn into_sorted_vec_points(self) -> Vec<NeighbourPoint<N, A, T, K>> {
        BinaryHeap::into_sorted_vec(self)
    }
}


impl<N, A, T, const K: usize> ResultCollection<N, A, T, K> for Vec<NeighbourPoint<N, A, T, K>>
where
    N: NeighbourEntry<A, T>,
    T: Content,
    A: Axis,
    NeighbourPoint<N, A, T, K>: Ord,
{
    fn new_with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity)
    }
    
    fn result_len(&self) -> usize {
        self.len()
    }

    fn result_pop(&mut self) -> Option<NeighbourPoint<N, A, T, K>> {
        self.pop()
    }

    fn result_peek(&self) -> Option<&NeighbourPoint<N, A, T, K>> {
        // Vec doesn't have a peek method, so use last
        self.last()
    }
    
    fn add(&mut self, entry: NeighbourPoint<N, A, T, K>) {
        self.push(entry)
    }
    
    fn max_dist(&self) -> A {
        A::infinity()
    }
    
    fn into_vec(self) -> Vec<N> {
        self.into_iter().map(|np| np.neighbour).collect()
    }
    
    fn into_sorted_vec(self) -> Vec<N> {
        let mut sorted = self;
        sorted.sort();
        sorted.into_iter().map(|np| np.neighbour).collect()
    }
    
    fn into_vec_points(self) -> Vec<NeighbourPoint<N, A, T, K>> {
        self
    }
    
    fn into_sorted_vec_points(mut self) -> Vec<NeighbourPoint<N, A, T, K>> {
        self.sort();
        self
    }
}



impl<N, A, T, const K: usize> ResultCollection<N, A, T, K> for SortedVec<NeighbourPoint<N, A, T, K>>
where
    N: NeighbourEntry<A, T>,
    T: Content,
    A: Axis,
    NeighbourPoint<N, A, T, K>: Ord,
{
    fn new_with_capacity(capacity: usize) -> Self {
        SortedVec::with_capacity(capacity)
    }
    
    fn result_len(&self) -> usize {
        self.len()
    }

    fn result_pop(&mut self) -> Option<NeighbourPoint<N, A, T, K>> {
        self.pop()
    }
    
    fn result_peek(&self) -> Option<&NeighbourPoint<N, A, T, K>> {
        // Use SortedVec's inherent last method as peek
        self.last()
    }

    fn add(&mut self, entry: NeighbourPoint<N, A, T, K>) {
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
            self.last().map_or(A::infinity(), |n| n.neighbour.distance())
        }
    }
    
    fn into_vec(self) -> Vec<N> {
        self.into_vec().into_iter().map(|np| np.neighbour).collect()
    }
    
    fn into_sorted_vec(self) -> Vec<N> {
        self.into_vec().into_iter().map(|np| np.neighbour).collect()
    }
    
    fn into_vec_points(self) -> Vec<NeighbourPoint<N, A, T, K>> {
        self.into_vec()
    }
    
    fn into_sorted_vec_points(self) -> Vec<NeighbourPoint<N, A, T, K>> {
        self.into_vec()
    }
}

impl<N, A, T, const K: usize> ResultCollection<N, A, T, K> for Option<NeighbourPoint<N, A, T, K>>
where
    N: NeighbourEntry<A, T>,
    T: Content,
    A: Axis,
{
    fn new_with_capacity(capacity: usize) -> Self {
        assert_eq!(capacity, 1);
        None
    }

    fn result_len(&self) -> usize {
        match self {
            Some(_) => 1_usize,
            None => 0_usize,
        }
    }

    fn result_pop(&mut self) -> Option<NeighbourPoint<N, A, T, K>> {
        match self {
            Some(entry) => {
                let result = entry.clone();
                *self = None;
                Some(result)
            },
            None => None,
        }
    }

    fn result_peek(&self) -> Option<&NeighbourPoint<N, A, T, K>> {
        self.as_ref()
    }
    
    fn add(&mut self, entry: NeighbourPoint<N, A, T, K>) {
        *self = Some(entry);
    }
    
    fn max_dist(&self) -> A {
        match self {
            Some(np) => np.neighbour.distance(),
            None => A::infinity(),
        }
    }
    
    fn into_vec(self) -> Vec<N> {
        match self {
            Some(np) => vec![np.neighbour],
            None => vec![],
        }
    }
    
    fn into_sorted_vec(self) -> Vec<N> {
        match self {
            Some(np) => vec![np.neighbour],
            None => vec![],
        }
    }
    
    fn into_vec_points(self) -> Vec<NeighbourPoint<N, A, T, K>> {
        match self {
            Some(np) => vec![np],
            None => vec![],
        }
    }
    
    fn into_sorted_vec_points(self) -> Vec<NeighbourPoint<N, A, T, K>> {
        match self {
            Some(np) => vec![np],
            None => vec![],
        }
    }
}
