use kiddo::float::distance::SquaredEuclidean;
use kiddo::float::kdtree::{Axis, KdTree};
use kiddo::test_utils::{build_populated_tree_float};
use kiddo::traits::{Content, Index};
use rand::distributions::{Distribution, Standard};
use std::hint::black_box;
use std::time::{Duration, Instant};
use az::Cast;

const BUCKET_SIZE: usize = 32;
const TREE_SIZE: usize = 1_000_000;
const DIMENSION: usize = 3;
const RUNTIME_SECONDS: u64 = 10;

// Generate a single random point for querying
fn random_point<A: Axis>() -> [A; DIMENSION] 
where
    Standard: Distribution<[A; DIMENSION]>,
{
    rand::random()
}

// Function to perform a single nearest neighbor query
fn perform_query_float<
    A: Axis,
    T: Content + 'static,
    const K: usize,
    IDX: Index<T = IDX> + 'static,
>(
    kdtree: &KdTree<A, T, K, BUCKET_SIZE, IDX>,
    point: &[A; K],
) where
    usize: Cast<IDX>,
{
    black_box(kdtree.nearest_one::<SquaredEuclidean>(point));
}

fn main() {
    println!("Building a 3D kdtree with {} nodes...", TREE_SIZE);
    
    // Build a kdtree with 1,000,000 random nodes
    let kdtree = build_populated_tree_float::<f64, u32, DIMENSION, BUCKET_SIZE, u32>(TREE_SIZE, 0);
    
    println!("Tree built. Running nearest neighbor queries for {} seconds...", RUNTIME_SECONDS);
    
    // Set up timing
    let start_time = Instant::now();
    let end_time = start_time + Duration::from_secs(RUNTIME_SECONDS);
    
    // Counter for number of queries performed
    let mut query_count: usize = 0;
    
    // Run queries until time is up
    while Instant::now() < end_time {
        // Generate a random query point
        let query_point = random_point::<f64>();
        
        // Perform nearest neighbor query
        perform_query_float::<f64, u32, DIMENSION, u32>(&kdtree, &query_point);
        
        // Increment counter
        query_count += 1;
        
        // Every million queries, give an update
        if query_count % 1_000_000 == 0 {
            println!("Completed {} million queries", query_count / 1_000_000);
        }
    }
    
    // Calculate queries per second
    let elapsed = start_time.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let queries_per_sec = query_count as f64 / elapsed_secs;
    
    println!("Benchmark complete.");
    println!("Ran {} queries in {:.2} seconds", query_count, elapsed_secs);
    println!("Average throughput: {:.2} queries/second", queries_per_sec);
}
