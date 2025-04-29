#[doc(hidden)]
#[macro_export]
macro_rules! generate_best_n_within {
    ($leafnode:ident, $comments:tt) => {
    doc_comment! {
    concat!$comments,
    #[inline]
    pub fn best_n_within<D>(
        &self,
        query: &[A; K],
        dist: A,
        max_qty: usize,
    ) -> impl Iterator<Item = Neighbour<A, T, K>>
    where
        D: DistanceMetric<A, K>,
    {
	self.best_n_within_scaled::<D>(query, dist, max_qty, None ).map(|nn| nn.0)
    }

    #[inline]
    pub fn best_n_within_scaled<D>(
        &self,
        query: &[A; K],
        dist: A,
        max_qty: usize,
	scale: Option<&[A; K]>,
    ) -> impl Iterator<Item = BestNeighbour<A, T, K>>
    where
        D: DistanceMetric<A, K>,
    {
        let mut off = [A::zero(); K];
        let mut best_items: BinaryHeap<BestNeighbour<A, T, K>> = BinaryHeap::new();

        unsafe {
            self.best_n_within_recurse::<D>(
                query,
                dist,
                max_qty,
                scale,
                self.root_index,
                0,
                &mut best_items,
                &mut off,
                A::zero(),
            );
        }

        best_items.into_iter()
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn best_n_within_recurse<D>(
        &self,
        query: &[A; K],
        radius: A,
        max_qty: usize,
	scale: Option<&[A; K]>,
        curr_node_idx: IDX,
        split_dim: usize,
        best_items: &mut BinaryHeap<BestNeighbour<A, T, K>>,
        off: &mut [A; K],
        rd: A,
    ) where
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

            self.best_n_within_recurse::<D>(
                query,
                radius,
                max_qty,
		scale,
                closer_node_idx,
                next_split_dim,
                best_items,
                off,
                rd,
            );

            rd = D::accumulate(rd, D::dist1(new_off, old_off, scale.map(|s| s[split_dim])));

            if rd <= radius {
                off[split_dim] = new_off;
                self.best_n_within_recurse::<D>(
                    query,
                    radius,
                    max_qty,
		    scale,
                    further_node_idx,
                    next_split_dim,
                    best_items,
                    off,
                    rd,
                );
                off[split_dim] = old_off;
            }
        } else {
            let leaf_node = self
                .leaves
                .get_unchecked((curr_node_idx - IDX::leaf_offset()).az::<usize>());

            Self::process_leaf_node::<D>(query, radius, max_qty, scale, best_items, leaf_node);
        }
    }

    #[inline]
    unsafe fn process_leaf_node<D>(
        query: &[A; K],
        radius: A,
        max_qty: usize,
        scale: Option<&[A; K]>,
        best_items: &mut BinaryHeap<BestNeighbour<A, T, K>>,
        leaf_node: &$leafnode<A, T, K, B, IDX>,
    ) where
        D: DistanceMetric<A, K>,
    {
        leaf_node
            .content_points
            .iter()
            .take(leaf_node.size.az::<usize>())
            .map(|entry| (D::dist(query, entry, scale), entry))
            .enumerate()
            .filter(|(_, (distance, _entry))| *distance <= radius)
            .for_each(|(idx, (distance, entry))| {
                Self::get_item_and_add_if_good(max_qty, best_items, leaf_node, idx, distance, entry)
            });
    }

    #[inline]
    unsafe fn get_item_and_add_if_good(
        max_qty: usize,
        best_items: &mut BinaryHeap<BestNeighbour<A, T, K>>,
        leaf_node: &$leafnode<A, T, K, B, IDX>,
        idx: usize,
        distance: A,
	entry: &[A; K]
    ) {
        let item = *leaf_node.content_items.get_unchecked(idx.az::<usize>());
	let element = BestNeighbour::new(distance, item, entry.clone());
        if best_items.len() < max_qty {
            best_items.push(element);
        } else {
            let mut top = best_items.peek_mut().unwrap();
            if element < *top {
                *top = element;
            }
        }
    }
}}}
