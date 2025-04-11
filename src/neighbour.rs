use crate::traits::Content;
use std::cmp::Ordering;
use std::marker::PhantomData;

/// Common base struct for all neighbour entries
#[derive(Debug, Copy, Clone)]
pub struct Neighbour<A, T>
where
    A: PartialOrd + Copy,
    T: Content,
{
    /// the distance of the found item from the query point according to the supplied distance metric
    pub distance: A,
    /// the stored index of an item that was found in the query
    pub item: T,
}

impl<A: PartialOrd + Copy, T: Content> Neighbour<A, T> {
    /// Create a new Neighbour with distance and item
    pub fn new(distance: A, item: T) -> Self {
        Self { distance, item }
    }
}

/// A common trait for neighbour entries used in search results
pub trait NeighbourEntry<A, T>: Clone
where
    A: PartialOrd + Copy,
    T: Content,
{
    /// Create a new neighbour entry with distance and item
    fn new(distance: A, item: T) -> Self;

    /// Get the distance of this entry
    fn distance(&self) -> A;

    /// Get the item of this entry
    fn item(&self) -> T;

    /// Convert to a tuple
    fn into_tuple(self) -> (A, T);
    
    /// Compare by distance (ascending)
    fn cmp_by_distance(&self, other: &Self) -> Option<Ordering>;
    
    /// Compare by item (ascending)
    fn cmp_by_item(&self, other: &Self) -> Option<Ordering>;
}

/// Implement the NeighbourEntry trait for the base Neighbour struct
impl<A: PartialOrd + Copy, T: Content> NeighbourEntry<A, T> for Neighbour<A, T> {
    fn new(distance: A, item: T) -> Self {
        Neighbour { distance, item }
    }
    
    fn distance(&self) -> A {
        self.distance
    }
    
    fn item(&self) -> T {
        self.item
    }
    
    fn into_tuple(self) -> (A, T) {
        (self.distance, self.item)
    }
    
    fn cmp_by_distance(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
    
    fn cmp_by_item(&self, other: &Self) -> Option<Ordering> {
        self.item.partial_cmp(&other.item)
    }
}

/// Wrapper for ordering by distance (ascending) - used for nearest neighbour queries
#[derive(Debug, Copy, Clone)]
pub struct NearestNeighbour<A: PartialOrd + Copy, T: Content>(pub Neighbour<A, T>);

impl<A: Copy + PartialOrd, T: Content> NearestNeighbour<A, T> {
    /// Create a new NearestNeighbour
    pub fn new(distance: A, item: T) -> Self {
        Self(Neighbour::new(distance, item))
    }
}

impl<A: PartialOrd + Copy, T: Content> Ord for NearestNeighbour<A, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl<A: PartialOrd + Copy, T: Content> PartialEq for NearestNeighbour<A, T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.distance == other.0.distance && self.0.item == other.0.item
    }
}

impl<A: PartialOrd + Copy + PartialEq, T: Content> PartialOrd for NearestNeighbour<A, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.distance.partial_cmp(&other.0.distance)
    }
}

impl<A: Copy + PartialOrd + PartialEq, T: Content> Eq for NearestNeighbour<A, T> {}

// Implement NeighbourEntry for NearestNeighbour wrapper
impl<A: Copy + PartialOrd, T: Content> NeighbourEntry<A, T> for NearestNeighbour<A, T> {
    fn new(distance: A, item: T) -> Self {
        Self(Neighbour::new(distance, item))
    }
    
    fn distance(&self) -> A {
        self.0.distance
    }
    
    fn item(&self) -> T {
        self.0.item
    }
    
    fn into_tuple(self) -> (A, T) {
        self.0.into_tuple()
    }
    
    fn cmp_by_distance(&self, other: &Self) -> Option<Ordering> {
        self.0.cmp_by_distance(&other.0)
    }
    
    fn cmp_by_item(&self, other: &Self) -> Option<Ordering> {
        self.0.cmp_by_item(&other.0)
    }
}

/// Wrapper for ordering by item (ascending) - used for best neighbour queries
#[derive(Debug, Copy, Clone)]
pub struct BestNeighbour<A: Copy + PartialOrd, T: Content>(pub Neighbour<A, T>);

impl<A: Copy + PartialOrd, T: Content> BestNeighbour<A, T> {
    /// Create a new BestNeighbour
    pub fn new(distance: A, item: T) -> Self {
        Self(Neighbour::new(distance, item))
    }
}

impl<A: Copy + PartialOrd, T: Content> Ord for BestNeighbour<A, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl<A: Copy + PartialOrd, T: Content> PartialEq for BestNeighbour<A, T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.distance == other.0.distance && self.0.item == other.0.item
    }
}

impl<A: Copy + PartialOrd, T: Content> PartialOrd for BestNeighbour<A, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.item.partial_cmp(&other.0.item)
    }
}

impl<A: Copy + PartialOrd + PartialEq, T: Content> Eq for BestNeighbour<A, T> {}

// Implement NeighbourEntry for BestNeighbour wrapper
impl<A: Copy + PartialOrd, T: Content> NeighbourEntry<A, T> for BestNeighbour<A, T> {
    fn new(distance: A, item: T) -> Self {
        Self(Neighbour::new(distance, item))
    }
    
    fn distance(&self) -> A {
        self.0.distance
    }
    
    fn item(&self) -> T {
        self.0.item
    }
    
    fn into_tuple(self) -> (A, T) {
        self.0.into_tuple()
    }
    
    fn cmp_by_distance(&self, other: &Self) -> Option<Ordering> {
        self.0.cmp_by_distance(&other.0)
    }
    
    fn cmp_by_item(&self, other: &Self) -> Option<Ordering> {
        self.0.cmp_by_item(&other.0)
    }
}

// Implement From for NearestNeighbour and BestNeighbour to convert to tuple
impl<A: Copy + PartialOrd, T: Content> From<NearestNeighbour<A, T>> for (A, T) {
    fn from(neighbour: NearestNeighbour<A, T>) -> Self {
        neighbour.0.into_tuple()
    }
}

impl<A: Copy + PartialOrd, T: Content> From<BestNeighbour<A, T>> for (A, T) {
    fn from(neighbour: BestNeighbour<A, T>) -> Self {
        neighbour.0.into_tuple()
    }
}

/// Generic wrapper for a neighbour entry with its associated point
#[derive(Debug, Copy, Clone)]
pub struct NeighbourPoint<N, A, T, const K: usize>
where
    N: NeighbourEntry<A, T>,
    A: Copy + PartialOrd,
    T: Content,
{
    pub neighbour: N, // A type w/ trait N; Cannot simply use N because of unused trait bounds?
    pub point: [A; K],
    /// Marker for the unused type parameter T
    _phantom: PhantomData<T>,
}

impl<N, A, T, const K: usize> NeighbourPoint<N, A, T, K>
where
    N: NeighbourEntry<A, T>,
    A: Copy + PartialOrd,
    T: Content,
{
    /// Create a new NeighbourPoint with a neighbour entry and point
    pub fn new(neighbour: N, point: [A; K]) -> Self {
        Self { 
            neighbour, 
            point,
            _phantom: PhantomData,
        }
    }
    
    /// Get the distance of this neighbour point
    pub fn distance(&self) -> A {
        self.neighbour.distance()
    }
    
    /// Get the item of this neighbour point
    pub fn item(&self) -> T {
        self.neighbour.item()
    }
}

/// Type alias for NeighbourPoint with NearestNeighbour ordering
pub type NearestNeighbourPoint<A, T, const K: usize> = NeighbourPoint<NearestNeighbour<A, T>, A, T, K>;

impl<A: Copy + PartialOrd, T: Content, const K: usize> NearestNeighbourPoint<A, T, K> {
    /// Create a new NearestNeighbourPoint
    pub fn new_nearest(distance: A, item: T, point: [A; K]) -> Self {
        Self::new(NearestNeighbour::new(distance, item), point)
    }
}

/// Type alias for NeighbourPoint with BestNeighbour ordering
pub type BestNeighbourPoint<A, T, const K: usize> = NeighbourPoint<BestNeighbour<A, T>, A, T, K>;

impl<A: Copy + PartialOrd, T: Content, const K: usize> BestNeighbourPoint<A, T, K> {
    /// Create a new BestNeighbourPoint
    pub fn new_best(distance: A, item: T, point: [A; K]) -> Self {
        Self::new(BestNeighbour::new(distance, item), point)
    }
}

// Implement Ord/PartialOrd for NeighbourPoint with a NeighbourEntry w/ either item or distance comparison
impl<N, A, T, const K: usize> Ord for NeighbourPoint<N, A, T, K>
where
    N: NeighbourEntry<A, T> + Ord,
    A: Copy + PartialOrd,
    T: Content,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.neighbour.cmp(&other.neighbour)
    }
}

impl<N, A, T, const K: usize> PartialOrd for NeighbourPoint<N, A, T, K>
where
    N: NeighbourEntry<A, T> + PartialOrd,
    A: Copy + PartialOrd,
    T: Content,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.neighbour.partial_cmp(&other.neighbour)
    }
}

// Implement Eq for NeighbourPoint
impl<N, A, T, const K: usize> Eq for NeighbourPoint<N, A, T, K>
where
    N: NeighbourEntry<A, T> + Eq,
    A: Copy + PartialOrd + PartialEq,
    T: Content + PartialEq,
{}

// Implement PartialEq for NeighbourPoint
impl<N, A, T, const K: usize> PartialEq for NeighbourPoint<N, A, T, K>
where
    N: NeighbourEntry<A, T> + PartialEq,
    A: Copy + PartialOrd + PartialEq,
    T: Content + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.neighbour == other.neighbour
    }
}
