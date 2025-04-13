//! Neighbour implementations for search results and queries
//! 
//! This module provides the base types for representing nearest neighbors
//! in k-d tree searches, with support for different ordering strategies.

use crate::traits::Content;
use std::cmp::Ordering;

/// A trait defining common operations for neighbour entries
pub trait NeighbourEntry<A, T, const K: usize>: Copy + Clone + PartialEq
where
    A: PartialOrd + Copy,
    T: Content,
{
    /// Get the distance value
    fn distance(&self) -> A;
    
    /// Get the item value
    fn item(&self) -> T;
    
    /// Get the point coordinates
    fn point(&self) -> &[A; K];
}

/// Base neighbour struct with point data, distance, and item
/// 
/// This is the foundation for all neighbour types in the system,
/// storing the essential information about a neighbor point.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Neighbour<A, T, const K: usize>
where
    A: PartialOrd + Copy,
    T: Content,
{
    /// Distance from the query point
    pub distance: A,
    /// Item identifier or index
    pub item: T,
    /// Coordinates of the point
    pub point: [A; K],
}

impl<A, T, const K: usize> Neighbour<A, T, K>
where
    A: PartialOrd + Copy,
    T: Content,
{
    /// Create a new Neighbour with distance, item, and point coordinates
    pub fn new(distance: A, item: T, point: [A; K]) -> Self {
        Self { distance, item, point }
    }
}

/// Implement NeighbourEntry for the base Neighbour struct
impl<A, T, const K: usize> NeighbourEntry<A, T, K> for Neighbour<A, T, K>
where
    A: PartialOrd + Copy,
    T: Content,
{
    fn distance(&self) -> A {
        self.distance
    }
    
    fn item(&self) -> T {
        self.item
    }
    
    fn point(&self) -> &[A; K] {
        &self.point
    }
}

/// Wrapper for ordering by distance (ascending) - used for nearest neighbour queries
/// 
/// This wrapper orders neighbours by distance in ascending order, making it
/// suitable for nearest neighbor searches.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct NearestNeighbour<A, T, const K: usize>(pub Neighbour<A, T, K>)
where
    A: PartialOrd + Copy,
    T: Content;

impl<A, T, const K: usize> NearestNeighbour<A, T, K>
where
    A: PartialOrd + Copy,
    T: Content,
{
    /// Create a new NearestNeighbour
    pub fn new(distance: A, item: T, point: [A; K]) -> Self {
        Self(Neighbour::new(distance, item, point))
    }
}

impl<A, T, const K: usize> NeighbourEntry<A, T, K> for NearestNeighbour<A, T, K>
where
    A: PartialOrd + Copy,
    T: Content,
{
    fn distance(&self) -> A {
        self.0.distance
    }
    
    fn item(&self) -> T {
        self.0.item
    }
    
    fn point(&self) -> &[A; K] {
        &self.0.point
    }
}

impl<A, T, const K: usize> Ord for NearestNeighbour<A, T, K>
where
    A: PartialOrd + Copy,
    T: Content,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl<A, T, const K: usize> PartialOrd for NearestNeighbour<A, T, K>
where
    A: PartialOrd + Copy,
    T: Content,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.distance.partial_cmp(&other.0.distance)
    }
}

impl<A, T, const K: usize> Eq for NearestNeighbour<A, T, K>
where
    A: PartialOrd + Copy + PartialEq,
    T: Content + PartialEq,
{}

impl<A, T, const K: usize> From<NearestNeighbour<A, T, K>> for (A, T)
where
    A: PartialOrd + Copy,
    T: Content,
{
    fn from(neighbour: NearestNeighbour<A, T, K>) -> Self {
        (neighbour.0.distance, neighbour.0.item)
    }
}

impl<A, T, const K: usize> From<NearestNeighbour<A, T, K>> for (A, T, [A; K])
where
    A: PartialOrd + Copy,
    T: Content,
{
    fn from(neighbour: NearestNeighbour<A, T, K>) -> Self {
        (neighbour.0.distance, neighbour.0.item, neighbour.0.point)
    }
}

/// Wrapper for ordering by item (ascending) - used for best neighbour queries
/// 
/// This wrapper orders neighbours by item in ascending order,
/// suitable for best match searches.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct BestNeighbour<A, T, const K: usize>(pub Neighbour<A, T, K>)
where
    A: PartialOrd + Copy,
    T: Content;

impl<A, T, const K: usize> BestNeighbour<A, T, K>
where
    A: PartialOrd + Copy,
    T: Content,
{
    /// Create a new BestNeighbour
    pub fn new(distance: A, item: T, point: [A; K]) -> Self {
        Self(Neighbour::new(distance, item, point))
    }
}

impl<A, T, const K: usize> NeighbourEntry<A, T, K> for BestNeighbour<A, T, K>
where
    A: PartialOrd + Copy,
    T: Content,
{
    fn distance(&self) -> A {
        self.0.distance
    }
    
    fn item(&self) -> T {
        self.0.item
    }
    
    fn point(&self) -> &[A; K] {
        &self.0.point
    }
}

impl<A, T, const K: usize> Ord for BestNeighbour<A, T, K>
where
    A: PartialOrd + Copy,
    T: Content + Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.item.cmp(&other.0.item)
    }
}

impl<A, T, const K: usize> PartialOrd for BestNeighbour<A, T, K>
where
    A: PartialOrd + Copy,
    T: Content + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.item.partial_cmp(&other.0.item)
    }
}

impl<A, T, const K: usize> Eq for BestNeighbour<A, T, K>
where
    A: PartialOrd + Copy + PartialEq,
    T: Content + PartialEq,
{}

impl<A, T, const K: usize> From<BestNeighbour<A, T, K>> for (A, T)
where
    A: PartialOrd + Copy,
    T: Content,
{
    fn from(neighbour: BestNeighbour<A, T, K>) -> Self {
        (neighbour.0.distance, neighbour.0.item)
    }
}


impl<A, T, const K: usize> From<BestNeighbour<A, T, K>> for (A, T, [A; K])
where
    A: PartialOrd + Copy,
    T: Content,
{
    fn from(neighbour: BestNeighbour<A, T, K>) -> Self {
        (neighbour.0.distance, neighbour.0.item, neighbour.0.point)
    }
}

