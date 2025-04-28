#[doc(hidden)]
#[macro_export]
macro_rules! generate_nearest_one {
    ($leafnode:ident, $comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn nearest_one<D>(&self, query: &[A; K]) -> Neighbour<A, T, K>
                where
                    D: DistanceMetric<A, K>,
            {
                let unit = [A::one(); K];
                self.nearest_one_scaled::<D>(query, &unit).0
            }

            #[inline]
            pub fn nearest_one_scaled<D>(&self, query: &[A; K], scale: &[A; K]) -> NearestNeighbour<A, T, K>
                where
                    D: DistanceMetric<A, K>,
            {
                let mut off = query.clone();  // The nearest possible point in the "further" dimension
		let mut stats: (usize, usize, usize) = (0, 0, 0);
                let result = unsafe {
                    self.nearest_one_recurse::<D>(
                        query,
                        scale,
                        self.root_index,
                        0,
                        NearestNeighbour::new(
                            A::max_value(),
                            T::default(),
                            [A::zero(); K]
                        ),
                        &mut off,
                        A::zero(),
			&mut stats,
                    )
                };
		println!("Nearest One scanned {} nodes, {} leaves and {}/{} points; nearest: {:?}",
			 stats.0, stats.1, stats.2, self.size, result.0);

		result
            }

            #[allow(clippy::too_many_arguments)]
            unsafe fn nearest_one_recurse<D>(
                &self,
                query: &[A; K],
                scale: &[A; K],
                curr_node_idx: IDX,
                split_dim: usize,
                mut nearest: NearestNeighbour<A, T, K>,
                off: &mut [A; K],
                rd: A,
		stats: &mut (usize, usize, usize),  // nodes, leaves, points
            ) -> NearestNeighbour<A, T, K>
                where
                    D: DistanceMetric<A, K>,
            {
                if is_stem_index(curr_node_idx) {
                    let node = &self.stems.get_unchecked(curr_node_idx.az::<usize>());

		    // Recurse into the "closer" node dimension; the nearest point could certainly
		    // be on this side.
		    //
		    // Computes the absolute difference between the last node.split_val for this
		    // axis, and the current .split_val.  Whichever "closer" side of the .split_val
		    // the query resides on will be recursed into.
		    //
		    // Then, we'll decide if we need to recurse into the "further" side, depending
		    // on the DistanceMetric between the query point and the *nearest* possible
		    // point on the "further" side.  We'll start with the exact query point as 'off'; as we spiral,
		    // if we're on the nearer side, the 'off' axis will be the exact query; if "further", the 'off' axis will
		    // be 
		    //
		    // For example; if we're spiraling down splitting the axes,
		    // and we're on the "closer" side, then the nearest coordinate 'off' in that
		    // axis is the exact query value; if we're querying the "further" side, the
		    // nearest coordinate 'off' in that axis is the .split_val.
		    //
		    //                             v
		    // |---------------------------|------------------------|
		    // ... (other axes) ...
		    //        v
		    // |------|--------------------|
		    //
		    //         <------------------>
		    //            added to 'rd'
		    //            (after D::dist1 computed on the range)

		    // Computes the absolute distance between the query and this axis .split_val The
		    // off array contains the current absolute distances.  These are un-scaled.
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

		    // On the "closer" side, the offset to the nearest possible point on this axis
		    // is the query point itself.
		    off[split_dim] = query[split_dim];
                    let nearest_neighbour = self.nearest_one_recurse::<D>(
                        query,
                        scale,
                        closer_node_idx,
                        next_split_dim,
                        nearest,
                        off,
                        rd,
			stats,
                    );
		    off[split_dim] = old_off;
		    stats.0 += 1;

                    if nearest_neighbour < nearest {
                        nearest = nearest_neighbour;
                    }

		    // Decide if we need to recurse into the "further" node of the dimension to find
		    // a potential nearer point, by checking if the nearest possible point in the
		    // "further" node could possibly be nearer to the query point.  We have computed
		    // the .distanct to the nearest point residing in the "closer" node.  How can we
		    // determine if it is possible for a nearer point to exist in the "further" node
		    // tree?
		    //
		    // - We know the distance between the query point and the "closer" side's
		    //   nearest point, accumulated in all dimensions.
		    // - We can compute the distance between the query point and the pivot .split_val
		    //   in this one dimension.
		    //
		    // If the distance to the closes possible dimension value in the further node is
		    // less than the distance to the current nearest neighbour, it *may* contain a
		    // nearer point!  Of course, we have to measure this distance using the official
		    // DistanceMetric, which may not be linear.  But, assuming that every
		    // dimensional value in every other dimension is the same (zero distance), there
		    // *could* exist a point at exactly the distance in this one dimension.
		    //
		    // What about 'rd'?  This accumulates a radius 'rd' as we spiral down through
		    // each dimension.  It is the accumulated distance between the query point and
		    // each dimension's .split_val pivot.  We know from this, that there cannot be a
		    // point in those dimensions nearer to the query than this accumulated offset.
		    // However, we try to accumulate this piecewise, by summing all the differences
		    // between the old and new .split_vals!  This assumes:
		    //
		    // 1) That each offset is uniformly more distant from the origin than the old, because
		    //    dist1 is (usually) an absolute value, and we're (usually) summing.
		    // 2) That the accumulation is linear and can accumulate piecewise (so, no "squared" calculations
		    //    allowed, because (a+b)^2 != a^2 + b^2.
		    //
		    // These are not sound assumptions, IMHO.
		    //
		    // This certainly won't work for Rectangular regions, which take the minimum of
		    // all dimensions as their radius.  I suspect it also won't work for eg. 3-D
		    // trees that are *more* than 3 layers deep, nor if old_off is not 0 (we could
		    // maintain this as a shortcut for a full D::dist for old_off == 0)
		    //
		    // To compute the "nearest possible" point in the "further" nodes that we
		    // haven't seen, we need to compute the accumulated distance from the query
		    // point to the full accumulated .split_val in every dimension, using the
		    // DistanceMetric.  Only if this is less than the current nearest.distance,
		    // could there possibly be a nearer point there.
		    
                    // rd = D::accumulate(rd, D::dist1(new_off, old_off, scale[split_dim]));
		    // // {
		    // // 	let mut new = off.clone();
		    // // 	new[split_dim] = new_off;
		    // // 	println!("rd w/ off[{}] == {:?} vs {:?}: {:?}, vs. dist: {:?}",
		    // // 		 split_dim, old_off, new_off, rd, D::dist(off, &new, scale));
		    // // }

                    // if rd <= nearest.0.distance {
                    //     off[split_dim] = new_off;
                    //     let result = self.nearest_one_recurse::<D>(
                    //         query,
                    //         scale,
                    //         further_node_idx,
                    //         next_split_dim,
                    //         nearest,
                    //         off,
                    //         rd,
		    // 	    stats,
                    //     );
		    // 	stats.0 += 1;
                    //     off[split_dim] = old_off;

                    //     if result < nearest {
                    //         nearest = result;
                    //     }
                    // }


		    // On the "further" side, the offset to the nearest possible point on this axis
		    // is the node.split_val.  Using the DistanceMetric, see if this nearest
		    // possible point in this "further" side is could possibly be nearer than the
		    // current nearest point.
                    off[split_dim] = node.split_val;

		    let further_distance = D::dist( query, off, scale );  // The total distance in all axes seems to eliminate too many
		    // let further_distance = D::accumulate( A::zero(), D::dist1( query[split_dim], node.split_val, scale[split_dim] ));
                    if further_distance <= nearest.0.distance {
                        let result = self.nearest_one_recurse::<D>(
                            query,
                            scale,
                            further_node_idx,
                            next_split_dim,
                            nearest,
                            off,
                            rd,
			    stats
			);
			stats.0 += 1;

                        if result < nearest {
                            nearest = result;
                        }
                    }
                    off[split_dim] = old_off;

                } else {
                    let leaf_node = self
                        .leaves
                        .get_unchecked((curr_node_idx - IDX::leaf_offset()).az::<usize>());

                    Self::search_content_for_nearest::<D>(
                        query,
                        scale,
                        &mut nearest,
                        leaf_node,
			stats,
                    );
                }

                nearest
            }

            #[inline]
            fn search_content_for_nearest<D>(
                query: &[A; K],
                scale: &[A; K],
                nearest: &mut NearestNeighbour<A, T, K>,
                leaf_node: &$leafnode<A, T, K, B, IDX>,
		stats: &mut (usize, usize, usize),  // nodes, leaves, points
            ) where
                D: DistanceMetric<A, K>,
            {
		let mut best: (A, usize, &[A; K]) = (nearest.0.distance, 0, &nearest.0.point);
		stats.1 += 1;
                leaf_node
                    .content_points
                    .iter()
                    .enumerate()
                    .take(leaf_node.size.az::<usize>())
                    .for_each(|(idx, entry)| {
                        let dist = D::dist(query, entry, scale);
                        if dist < best.0 {
			    best = (dist, idx, entry);
                        }
			stats.2 += 1;
                    });
                if best.0 < nearest.0.distance {
                    // Create a new NearestNeighbour with the better distance
                    let item = unsafe { *leaf_node.content_items.get_unchecked(best.1) };
                    *nearest = NearestNeighbour::new(best.0, item, best.2.clone());
                }
            }
        }
    };
}
