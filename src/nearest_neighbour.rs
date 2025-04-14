//! A result item returned by a query
//! Re-exports NearestNeighbour from neighbour.rs for backward compatibility

pub use crate::neighbour::{NearestNeighbour, Neighbour};
pub use crate::traits::Content;



#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn test_from_tuple() {
        let n = NearestNeighbour::new(1.0f32, 1usize);
        let nn: (f32, usize) = n.0.into_tuple();

        assert_eq!(nn.0, 1.0f32);
        assert_eq!(nn.1, 1usize);
    }

    #[test]
    fn test_nearest_neighbour_comparison() {
        let a = NearestNeighbour::new(1.0, 10);
        let b = NearestNeighbour::new(2.0, 5);
        
        // NearestNeighbour compares by distance (ascending)
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
    }

    #[test]
    fn test_nearest_neighbour_point_construction() {
        let point = [1.0, 2.0, 3.0];
        let np = NearestNeighbourPoint::new_nearest(0.5, 42, point);
        
        assert_eq!(np.point, point);
        assert_eq!(np.neighbor.0.distance, 0.5);
        assert_eq!(np.neighbor.0.item, 42);
    }
}
