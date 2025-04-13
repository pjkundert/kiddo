//! Consistent interface for storing 0, 1 or more neighbor entry points.
//!
//! Capable of storing and ordering {Best,Nearest}Neighbour implementations

use crate::float::kdtree::Axis;
use crate::neighbour::NeighbourEntry;
use crate::traits::Content;
use sorted_vec::SortedVec;
use std::collections::BinaryHeap;

pub trait ResultCollection<N, A, T, const K: usize>
where
    N: NeighbourEntry<A, T, K>,
    T: Content,
    A: Axis,
{
    fn new_with_capacity(capacity: usize) -> Self;
    fn result_len(&self) -> usize;
    fn result_pop(&mut self) -> Option<N>;
    fn result_peek(&self) -> Option<&N>;
    fn add(&mut self, entry: N);
    fn max_dist(&self) -> A;
    fn into_vec(self) -> Vec<N>;
    fn into_sorted_vec(self) -> Vec<N>;
}


impl<N, A, T, const K: usize> ResultCollection<N, A, T, K> for BinaryHeap<N>
where
    N: NeighbourEntry<A, T, K> + Ord,
    T: Content,
    A: Axis,
{
    fn new_with_capacity(capacity: usize) -> Self {
        BinaryHeap::with_capacity(capacity)
    }
    
    fn result_len(&self) -> usize {
        self.len()
    }

    fn result_pop(&mut self) -> Option<N> {
        self.pop()
    }

    fn result_peek(&self) -> Option<&N> {
        self.peek()
    }

    fn add(&mut self, entry: N) {
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
            self.peek().map_or(A::infinity(), |n| n.distance())
        }
    }
    
    fn into_vec(self) -> Vec<N> {
        BinaryHeap::into_vec(self)
    }
    
    fn into_sorted_vec(self) -> Vec<N> {
        BinaryHeap::into_sorted_vec(self)
    }
}


impl<N, A, T, const K: usize> ResultCollection<N, A, T, K> for Vec<N>
where
    N: NeighbourEntry<A, T, K> + Ord,
    T: Content,
    A: Axis,
{
    fn new_with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity)
    }
    
    fn result_len(&self) -> usize {
        self.len()
    }

    fn result_pop(&mut self) -> Option<N> {
        self.pop()
    }

    fn result_peek(&self) -> Option<&N> {
        // Vec doesn't have a peek method, so use last
        self.last()
    }
    
    fn add(&mut self, entry: N) {
        self.push(entry)
    }
    
    fn max_dist(&self) -> A {
        A::infinity()
    }
    
    fn into_vec(self) -> Vec<N> {
        self
    }
    
    fn into_sorted_vec(self) -> Vec<N> {
        let mut sorted = self;
        sorted.sort();
        sorted
    }
}



impl<N, A, T, const K: usize> ResultCollection<N, A, T, K> for SortedVec<N>
where
    N: NeighbourEntry<A, T, K> + Ord,
    T: Content,
    A: Axis,
{
    fn new_with_capacity(capacity: usize) -> Self {
        SortedVec::with_capacity(capacity)
    }
    
    fn result_len(&self) -> usize {
        self.len()
    }

    fn result_pop(&mut self) -> Option<N> {
        self.pop()
    }
    
    fn result_peek(&self) -> Option<&N> {
        // Use SortedVec's inherent last method as peek
        self.last()
    }

    fn add(&mut self, entry: N) {
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
            self.last().map_or(A::infinity(), |n| n.distance())
        }
    }
    
    fn into_vec(self) -> Vec<N> {
        self.into_vec()
    }
    
    fn into_sorted_vec(self) -> Vec<N> {
        self.into_vec()
    }
}

impl<N, A, T, const K: usize> ResultCollection<N, A, T, K> for Option<N>
where
    N: NeighbourEntry<A, T, K>,
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

    fn result_pop(&mut self) -> Option<N> {
        match self {
            Some(entry) => {
                let result = entry.clone();
                *self = None;
                Some(result)
            },
            None => None,
        }
    }

    fn result_peek(&self) -> Option<&N> {
        self.as_ref()
    }
    
    fn add(&mut self, entry: N) {
        *self = Some(entry);
    }
    
    fn max_dist(&self) -> A {
        match self {
            Some(n) => n.distance(),
            None => A::infinity(),
        }
    }
    
    fn into_vec(self) -> Vec<N> {
        match self {
            Some(n) => vec![n],
            None => vec![],
        }
    }
    
    fn into_sorted_vec(self) -> Vec<N> {
        self.into_vec()
    }
}
