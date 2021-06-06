use crate::{Capacity, Edge, EdgeId, Flow, Node, ResidualEdge};

pub struct Network<C> {
    outgoing: Vec<Option<Vec<(EdgeId, Node)>>>, // outgoing edges
    incoming: Vec<Option<Vec<(EdgeId, Node)>>>, // incoming edges reversed
    capacities: Vec<C>, // index by the edge id, which is unique for each edge in the _input_ graph. Reversed edge have the same id
    num_edges: usize,
}

impl<C> Network<C> {
    pub fn empty() -> Self {
        Network {
            outgoing: Vec::new(),
            incoming: Vec::new(),
            capacities: Vec::new(),
            num_edges: 0,
        }
    }

    pub fn add_edge(&mut self, from: Node, to: Node, cap: C) {
        if from >= self.outgoing.len() {
            self.outgoing.resize(from + 1, None);
            self.incoming.resize(from + 1, None);
        }
        if to >= self.incoming.len() {
            self.outgoing.resize(to + 1, None);
            self.incoming.resize(to + 1, None);
        }

        if let Some(neighbors) = &mut self.outgoing.get_mut(from).unwrap() {
            neighbors.push((self.num_edges, to));
        } else {
            *self.outgoing.get_mut(from).unwrap() = Some(vec![(self.num_edges, to)]);
        }

        if let Some(neighbors) = &mut self.incoming.get_mut(to).unwrap() {
            neighbors.push((self.num_edges, from));
        } else {
            *self.incoming.get_mut(to).unwrap() = Some(vec![(self.num_edges, from)]);
        }
        self.capacities.push(cap);
        self.num_edges += 1;
    }

    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    pub fn num_vertices(&self) -> usize {
        self.outgoing.len().max(self.incoming.len())
    }

    pub fn capacity(&self, id: EdgeId) -> &C {
        self.capacities.get(id).unwrap()
    }

    pub fn res_adjacent<'a>(&'a self, node: Node, flow: &'a Flow<C>) -> ResidualAdjacentEdges<C> {
        ResidualAdjacentEdges {
            node,
            flow,
            capacities: &self.capacities,
            outgoing_edge_iter: self
                .outgoing
                .get(node)
                .unwrap()
                .as_ref()
                .map(|hm| hm.iter()),
            incoming_edge_iter: self
                .incoming
                .get(node)
                .unwrap()
                .as_ref()
                .map(|hm| hm.iter()),
        }
    }

    pub fn adjacent(&self, node: Node) -> AdjacentEdges<C> {
        AdjacentEdges {
            node,
            capacities: &self.capacities,
            outgoing_edge_iter: self
                .outgoing
                .get(node)
                .unwrap()
                .as_ref()
                .map(|hm| hm.iter()),
        }
    }
}

/// Iterator over edges in the network.
pub struct AdjacentEdges<'a, C> {
    node: Node,
    capacities: &'a Vec<C>,
    outgoing_edge_iter: Option<std::slice::Iter<'a, (EdgeId, Node)>>,
}

impl<'a, C> Iterator for AdjacentEdges<'a, C>
where
    C: Capacity,
{
    type Item = Edge<C>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(outgoing_edge_iter) = &mut self.outgoing_edge_iter {
            while let Some((id, t)) = outgoing_edge_iter.next() {
                let cap = self.capacities[*id];
                return Some(Edge::new(self.node, *t, *id, cap));
            }
        }
        None
    }
}

/// Iterator over edges in the residual network.
pub struct ResidualAdjacentEdges<'a, C> {
    node: Node,
    flow: &'a Flow<C>,
    capacities: &'a Vec<C>,
    outgoing_edge_iter: Option<std::slice::Iter<'a, (EdgeId, Node)>>,
    incoming_edge_iter: Option<std::slice::Iter<'a, (EdgeId, Node)>>,
}

impl<'a, C> Iterator for ResidualAdjacentEdges<'a, C>
where
    C: Capacity,
{
    type Item = ResidualEdge<C>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(outgoing_edge_iter) = &mut self.outgoing_edge_iter {
            while let Some((id, t)) = outgoing_edge_iter.next() {
                let cap = self.capacities[*id];
                let flow = self.flow[*id];
                if flow < cap {
                    return Some(ResidualEdge::Directed(Edge::new(
                        self.node,
                        *t,
                        *id,
                        cap - flow,
                    )));
                }
            }
        }
        if let Some(incoming_edge_iter) = &mut self.incoming_edge_iter {
            while let Some((id, s)) = incoming_edge_iter.next() {
                let flow = self.flow[*id];
                if flow > C::zero() {
                    return Some(ResidualEdge::Reversed(Edge::new(self.node, *s, *id, flow)));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod test_network {
    use ordered_float::OrderedFloat;

    use super::*;

    #[test]
    fn test_residual_edges() {
        let mut n = Network::<OrderedFloat<f64>>::empty();
        n.add_edge(0, 1, 6.0.into());
        n.add_edge(0, 2, 2.0.into());
        n.add_edge(1, 2, 5.0.into());
        n.add_edge(1, 3, 3.0.into());
        n.add_edge(2, 3, 4.0.into());

        assert_eq!(5, n.num_edges());
        assert_eq!(4, n.num_vertices());

        let mut flow = Flow::<OrderedFloat<f64>>::zero_flow(n.num_edges());

        {
            let mut iter = n.res_adjacent(2, &flow);
            assert_eq!(
                iter.next().unwrap().into_edge(),
                Edge::new(2, 3, 4, 4.0.into())
            );
            assert!(iter.next().is_none());
        }

        flow[1] = 2.0.into();
        flow[4] = 2.0.into();

        {
            let mut iter = n.res_adjacent(2, &flow);
            assert_eq!(
                iter.next().unwrap().into_edge(),
                Edge::new(2, 3, 4, 2.0.into())
            );
            assert_eq!(
                iter.next().unwrap().into_edge(),
                Edge::new(2, 0, 1, 2.0.into())
            );
            assert!(iter.next().is_none());
        }

        flow[0] = 3.0.into();
        flow[3] = 3.0.into();

        {
            let mut iter = n.res_adjacent(1, &flow);
            assert_eq!(
                iter.next().unwrap().into_edge(),
                Edge::new(1, 2, 2, 5.0.into())
            );
            assert_eq!(
                iter.next().unwrap().into_edge(),
                Edge::new(1, 0, 0, 3.0.into())
            );
            assert!(iter.next().is_none());
        }
    }
}
