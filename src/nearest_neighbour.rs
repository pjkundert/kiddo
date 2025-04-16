//! A result item returned by a query
//! Re-exports NearestNeighbour from neighbour.rs for backward compatibility

pub use crate::neighbour::{NearestNeighbour, Neighbour};
pub use crate::traits::Content;



#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn test_smoke() {
        let point = [1.0f64, 2.0, 3.0];
        let nn: NearestNeighbour<f64, usize, 3> = NearestNeighbour::new(1.0, 2, point);
        assert_eq!(nn.0.distance, 1.0);
        assert_eq!(nn.0.item, 2);
        assert_eq!(nn.0.point, point);
    }

    #[test]
    fn test_from_tuple() {
	let point = [0f32, 1f32];
        let n = NearestNeighbour::new(1.0f32, 1usize, point);
        let nn0: (f32, usize) = n.0.into();
        let nn: (f32, usize) = n.into();
        let nnp: (f32, usize, [f32; 2_usize]) = n.into();

        assert_eq!(nn0.0, nn.0);
        assert_eq!(nn.0, 1.0f32);
        assert_eq!(nn.1, 1usize);
        assert_eq!(nnp.2, point);
    }

    #[test]
    fn test_nearest_neighbour_comparison() {
        let a = NearestNeighbour::new(1.0, 10, [0.0]);
        let b = NearestNeighbour::new(2.0, 5,  [1.0]);

        // NearestNeighbour compares by distance (ascending)
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
    }
}
