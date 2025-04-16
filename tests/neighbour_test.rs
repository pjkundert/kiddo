use kiddo::neighbour::{BestNeighbour, Neighbour};


#[test]
fn test_smoke() {
    let bn = BestNeighbour(Neighbour{ distance: 1_f64, item: 2_usize, point:[1_f64; 3] });
    assert_eq!(bn.0.distance, 1.0f64);
    assert_eq!(bn.0.item, 2usize);
    assert_eq!(bn.0.point, [1_f64; 3]);

}

#[test]
fn test_from_tuple() {
    let b = BestNeighbour::new(1.0f32, 1usize, [0_f32; 5]);
    let nn: (f32, usize) = b.into();

    assert_eq!(nn.0, 1.0f32);
    assert_eq!(nn.1, 1usize);
}

#[test]
fn test_best_neighbour_comparison() {
    let a = BestNeighbour::new(1.0, 10, [0.0; 2]);
    let b = BestNeighbour::new(2.0, 5, [0.0; 2]);

    // BestNeighbour compares by item (ascending)
    assert_eq!(a.partial_cmp(&b), Some(std::cmp::Ordering::Greater));
}

#[test]
fn test_best_neighbour_point_construction() {
    let point = [1.0, 2.0, 3.0];
    let bp = BestNeighbour::new(0.5, 42, point);

    assert_eq!(bp.0.point, point);
    assert_eq!(bp.0.distance, 0.5);
    assert_eq!(bp.0.item, 42);
}
