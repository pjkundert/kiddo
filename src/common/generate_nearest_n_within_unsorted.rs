#[doc(hidden)]
#[macro_export]
macro_rules! generate_nearest_n_within_unsorted {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,

            #[inline]
            pub fn nearest_n_within<D>(&self, query: &[A; K], dist: A, max_items: std::num::NonZero<usize>, sorted: bool) -> Vec<NearestNeighbour<A, T>>
            where
                D: DistanceMetric<A, K>,
            {
		let unit = [A::one(); K];
                let result = if sorted || max_items < std::num::NonZero::new(usize::MAX).unwrap() {
                    if max_items <= std::num::NonZero::new(MAX_VEC_RESULT_SIZE).unwrap() {
                        self.nearest_n_within_stub::<D, SortedVec<NearestNeighbourPoint<A, T, K>>>(query, &unit, dist, max_items.get(), sorted)
                    } else {
                        self.nearest_n_within_stub::<D, BinaryHeap<NearestNeighbourPoint<A, T, K>>>(query, &unit, dist, max_items.get(), sorted)
                    }
                } else {
                    self.nearest_n_within_stub::<D, Vec<NearestNeighbourPoint<A, T, K>>>(query, &unit, dist, 0, sorted)
                };

                if sorted {
                    result.into_sorted_vec()
                } else {
                    result.into_vec()
                }
            }

            #[inline]
            pub fn nearest_n_within_points<D>(
		&self,
		query: &[A; K],
		scale: &[A; K],
		dist: A,
		max_items: std::num::NonZero<usize>,
		sorted: bool
	    ) -> Vec<NearestNeighbourPoint<A, T, K>>
            where
                D: DistanceMetric<A, K>,
            {
		let result = if sorted || max_items < std::num::NonZero::new(usize::MAX).unwrap() {
                    if max_items <= std::num::NonZero::new(MAX_VEC_RESULT_SIZE).unwrap() {
                        self.nearest_n_within_stub::<D, SortedVec<NearestNeighbourPoint<A, T, K>>>(query, scale, dist, max_items.get(), sorted)
                    } else {
                        self.nearest_n_within_stub::<D, BinaryHeap<NearestNeighbourPoint<A, T, K>>>(query, scale, dist, max_items.get(), sorted)
                    }
                } else {
                    self.nearest_n_within_stub::<D, Vec<NearestNeighbourPoint<A, T, K>>>(query, scale, dist, 0, sorted)
                };

                if sorted {
                    result.into_sorted_vec_points()
                } else {
                    result.into_vec_points()
                }
            }

            fn nearest_n_within_stub<D: DistanceMetric<A, K>, R>(
                &self,
		query: &[A; K],
		scale: &[A; K],
		dist: A,
		res_capacity: usize,
		sorted: bool
            ) -> R
	    where
		R: ResultCollection<NearestNeighbour<A, T>, A, T, K>
	    {
                let mut matching_items = R::new_with_capacity(res_capacity);
                let mut off = [A::zero(); K];

                unsafe {
                    self.nearest_n_within_unsorted_recurse::<D, R>(
                        query,
			scale,
                        dist,
                        self.root_index,
                        0,
                        &mut matching_items,
                        &mut off,
                        A::zero(),
                    );
                }

		matching_items
            }

            #[allow(clippy::too_many_arguments)]
            unsafe fn nearest_n_within_unsorted_recurse<D, R>(
                &self,
                query: &[A; K],
                scale: &[A; K],
                radius: A,
                curr_node_idx: IDX,
                split_dim: usize,
                matching_items: &mut R,
                off: &mut [A; K],
                rd: A,
            ) where
                D: DistanceMetric<A, K>,
		R: ResultCollection<NearestNeighbour<A, T>, A, T, K>,
            {
                if is_stem_index(curr_node_idx) {
                    let node = self.stems.get_unchecked(curr_node_idx.az::<usize>());

                    let mut rd = rd;
                    let old_off = off[split_dim];
                    let new_off = query[split_dim].saturating_dist(node.split_val);

                    let [closer_node_idx, further_node_idx] =
                        if *query.get_unchecked(split_dim) < node.split_val {
                            [node.left, node.right]
                        } else {
                            [node.right, node.left]
                        };
                    let next_split_dim = (split_dim + 1).rem(K);

                    self.nearest_n_within_unsorted_recurse::<D, R>(
                        query,
                        radius,
                        closer_node_idx,
                        next_split_dim,
                        matching_items,
                        off,
                        rd,
                    );

                    rd = Axis::rd_update(rd, D::dist1(new_off, old_off, scale[split_dim]));

                    if rd <= radius {
                        off[split_dim] = new_off;
                        self.nearest_n_within_unsorted_recurse::<D, R>(
                            query,
                            radius,
                            further_node_idx,
                            next_split_dim,
                            matching_items,
                            off,
                            rd,
                        );
                        off[split_dim] = old_off;
                    }
                } else {
                    let leaf_node = self
                        .leaves
                        .get_unchecked((curr_node_idx - IDX::leaf_offset()).az::<usize>());

                    leaf_node
                        .content_points
                        .iter()
                        .enumerate()
                        .take(leaf_node.size.az::<usize>())
                        .for_each(|(idx, entry)| {
                            let distance = D::dist(query, entry);

                            if distance < radius {
                                matching_items.add(NearestNeighbourPoint {
				    neighbour: NearestNeighbour {
					distance,
					item: *leaf_node.content_items.get_unchecked(idx.az::<usize>()),
                                    },
				    point: entry.to_owned(),
				})
                            }
                        });
                }
            }
        }
    };
}
