#[doc(hidden)]
#[macro_export]
macro_rules! generate_within {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn within<D>(&self, query: &[A; K], dist: A) -> Vec<Neighbour<A, T, K>>
            where
                D: DistanceMetric<A, K>,
            {
		let unit = [A::one(); K];
                self.within_scaled::<D>(query, &unit, dist).iter().map(|nn| nn.0).collect()

            }

            #[inline]
            pub fn within_scaled<D>(&self, query: &[A; K], scale: &[A; K], dist: A) -> Vec<NearestNeighbour<A, T, K>>
            where
                D: DistanceMetric<A, K>,
            {
                let mut matching_items = self.within_unsorted_scaled::<D>(query, scale, dist);
                matching_items.sort();
                matching_items
            }
        }
    };
}
