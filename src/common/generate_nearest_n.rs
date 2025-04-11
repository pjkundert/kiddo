#[doc(hidden)]
#[macro_export]
macro_rules! generate_nearest_n {
    ($comments:tt) => {
    doc_comment! {
    concat!$comments,
    #[inline]
    pub fn nearest_n<D>(&self, query: &[A; K], qty: usize) -> Vec<NearestNeighbour<A, T>>
    where
        D: DistanceMetric<A, K>,
    {
        let unit = [A::one(); K];
        self.nearest_n_points::<D>(query, &unit, qty).iter().map(|(nn,_p)| nn.to_owned()).collect()
    }

    #[inline]
    pub fn nearest_n_points<D>(&self, query: &[A; K], scale: &[A; K], qty: usize) -> Vec<(NearestNeighbour<A, T>, [A; K])>
    where
        D: DistanceMetric<A, K>,
    {
        let mut off = [A::zero(); K];
        let mut result: BinaryHeap<NearestNeighbourPoint<A, T, K>> = BinaryHeap::with_capacity(qty);

        unsafe {
            self.nearest_n_recurse::<D>(
                query,
		scale,
                self.root_index,
                0,
                &mut result,
                &mut off,
                A::zero(),
            )
        }

        result.into_sorted_vec().iter().map(|nnp| (nnp.neighbour,nnp.point)).collect()
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn nearest_n_recurse<D>(
        &self,
        query: &[A; K],
        scale: &[A; K],
        curr_node_idx: IDX,
        split_dim: usize,
        results: &mut BinaryHeap<NearestNeighbourPoint<A, T, K>>,
        off: &mut [A; K],
        rd: A,
    ) where
        D: DistanceMetric<A, K>,
    {
        if is_stem_index(curr_node_idx) {
            let node = &self.stems.get_unchecked(curr_node_idx.az::<usize>());

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

            self.nearest_n_recurse::<D>(
                query,
                scale,
                closer_node_idx,
                next_split_dim,
                results,
                off,
                rd,
            );

            rd = D::accumulate(rd, D::dist1(new_off, old_off, scale[split_dim]));

            if Self::dist_belongs_in_heap(rd, results) {
                off[split_dim] = new_off;
                self.nearest_n_recurse::<D>(
                    query,
		    scale,
                    further_node_idx,
                    next_split_dim,
                    results,
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
                .take(leaf_node.size.az::<usize>())
                .enumerate()
                .for_each(|(idx, entry)| {
                    let distance: A = D::dist(query, entry, scale);
                    if Self::dist_belongs_in_heap(distance, results) {
                        let item = unsafe { *leaf_node.content_items.get_unchecked(idx) };
                        let element = NearestNeighbourPoint::new_nearest(distance, item, *entry);
                        if results.len() < results.capacity() {
                            results.push(element)
                        } else {
                            let mut top = results.peek_mut().unwrap();
                            if element.neighbour.distance() < top.neighbour.distance() {
                                *top = element;
                            }
                        }
                    }
                });
        }
    }

    #[inline]
    fn dist_belongs_in_heap(dist: A, heap: &BinaryHeap<NearestNeighbourPoint<A, T, K>>) -> bool {
        heap.is_empty() || dist < heap.peek().unwrap().neighbour.distance || heap.len() < heap.capacity()
    }
}}}
