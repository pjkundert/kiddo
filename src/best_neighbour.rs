//! A result item returned by a query
//! Re-exports BestNeighbour from neighbour.rs for backward compatibility


pub use crate::neighbour::BestNeighbour;
pub use crate::traits::Content;



#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn test_smoke() {
        let point = [1.0f64, 2.0, 3.0];
        let bn: BestNeighbour<f64, usize, 3> = BestNeighbour::new(1.0, 2, point);
        assert_eq!(bn.distance(), 1.0);
        assert_eq!(bn.item(), 2);
    }

    #[test]
    fn test_from_tuple() {
        let point = [1.0f32, 2.0, 3.0];
        let b: BestNeighbour<f32, usize, 3> = BestNeighbour::new(1.0f32, 1usize, point);
        let nn: (f32, usize) = b.into_tuple();

        assert_eq!(nn.0, 1.0f32);
        assert_eq!(nn.1, 1usize);
    }

    #[test]
    fn test_best_neighbour_comparison() {
        let point_a = [1.0f64, 2.0, 3.0];
        let point_b = [4.0f64, 5.0, 6.0];
        let a: BestNeighbour<f64, usize, 3> = BestNeighbour::new(1.0, 10, point_a);
        let b: BestNeighbour<f64, usize, 3> = BestNeighbour::new(2.0, 5, point_b);
        
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

