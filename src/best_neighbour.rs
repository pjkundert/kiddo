//! A result item returned by a query
//! Re-exports BestNeighbour and BestNeighbourPoint from neighbour.rs

// Re-export the types from neighbour.rs
pub use crate::neighbour::{BestNeighbour, BestNeighbourPoint};
pub use crate::traits::Content;



#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn test_smoke() {
	let bn = BestNeighbour(Neighbour{ distance: 1_f64, item: 2_uszie });
        assert_eq!(bn.0.distance, 1.0f32);
        assert_eq!(nn.0.item, 1usize);

    }

    #[test]
    fn test_from_tuple() {
        let b = BestNeighbour::new(1.0f32, 1usize);
        let nn: (f32, usize) = b.0.into_tuple();

        assert_eq!(nn.0, 1.0f32);
        assert_eq!(nn.1, 1usize);
    }

    #[test]
    fn test_best_neighbour_comparison() {
        let a = BestNeighbour::new(1.0, 10);
        let b = BestNeighbour::new(2.0, 5);
        
        // BestNeighbour compares by item (ascending)
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Greater));
    }

    #[test]
    fn test_best_neighbour_point_construction() {
        let point = [1.0, 2.0, 3.0];
        let bp = BestNeighbourPoint::new_best(0.5, 42, point);
        
        assert_eq!(bp.point, point);
        assert_eq!(bp.neighbor.0.distance, 0.5);
        assert_eq!(bp.neighbor.0.item, 42);
    }
}

