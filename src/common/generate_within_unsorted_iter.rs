#[doc(hidden)]
#[macro_export]
macro_rules! generate_within_unsorted_iter {
    ($comments:tt) => {
        doc_comment! {
            concat!$comments,
            #[inline]
            pub fn within_unsorted_iter<D>(
                &'a self,
                query: &'a [A; K],
                dist: A,
            ) -> impl Iterator<Item = Neighbour<A, T, K>> + 'a
            where
                D: DistanceMetric<A, K>,
            {
                // Create the generator directly with its own unit scale array
                let gen = Gn::new_scoped(move |gen_scope| {
                    // Create all necessary variables inside the generator scope
                    let unit = [A::one(); K];
                    let mut off = [A::zero(); K];
                    
                    unsafe {
                        self.within_unsorted_iter_recurse::<D>(
                            query,
                            &unit,
                            dist,
                            self.root_index,
                            0,
                            gen_scope,
                            &mut off,
                            A::zero(),
                        );
                    }

                    done!();
                });

                // Convert the generator to an iterator that yields Neighbour<A, T, K>
                WithinUnsortedIter::new(gen).map(|nn| nn.0)
            }

            #[inline]
            pub fn within_unsorted_iter_scaled<D>(
                &'a self,
                query: &'a [A; K],
                scale: &'a [A; K],
                dist: A,
            ) -> impl Iterator<Item = NearestNeighbour<A, T, K>> + 'a
            where
                D: DistanceMetric<A, K>,
            {
                // Create the generator directly with its own offset array
                let gen = Gn::new_scoped(move |gen_scope| {
                    // Create offset array inside generator scope
                    let mut off = [A::zero(); K];
                    
                    unsafe {
                        self.within_unsorted_iter_recurse::<D>(
                            query,
                            scale,
                            dist,
                            self.root_index,
                            0,
                            gen_scope,
                            &mut off,
                            A::zero(),
                        );
                    }

                    done!();
                });

                // Return the iterator directly as it already yields NearestNeighbour<A, T, K>
                WithinUnsortedIter::new(gen)
            }

            #[allow(clippy::too_many_arguments)]
            unsafe fn within_unsorted_iter_recurse<'scope, D>(
                &'a self,
                query: &[A; K],
                scale: &[A; K],
                radius: A,
                curr_node_idx: IDX,
                split_dim: usize,
                mut gen_scope: Scope<'scope, 'a, (), NearestNeighbour<A, T, K>>,
                off: &mut [A; K],
                rd: A,
            ) -> Scope<'scope, 'a, (), NearestNeighbour<A, T, K>>
            where
                D: DistanceMetric<A, K>,
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

                    gen_scope = self.within_unsorted_iter_recurse::<D>(
                        query,
                        scale,
                        radius,
                        closer_node_idx,
                        next_split_dim,
                        gen_scope,
                        off,
                        rd,
                    );

                    rd = D::accumulate(rd, D::dist1(new_off, old_off, scale[split_dim]));

                    if rd <= radius {
                        off[split_dim] = new_off;
                        gen_scope = self.within_unsorted_iter_recurse::<D>(
                            query,
                            scale,
                            radius,
                            further_node_idx,
                            next_split_dim,
                            gen_scope,
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
                            let distance = D::dist(query, entry, scale);

                            if distance < radius {
                                gen_scope.yield_with(NearestNeighbour::new(
                                    distance,
                                    *leaf_node.content_items.get_unchecked(idx.az::<usize>()),
                                    entry.to_owned()
                                ));
                            }
                        });
                }

                gen_scope
            }
        }
    };
}
