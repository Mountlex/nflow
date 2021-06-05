use std::{
    collections::VecDeque,
    ops::{Add, AddAssign, Index, IndexMut, Sub, SubAssign},
};

use ordered_float::OrderedFloat; 

type Node = usize;

type EdgeId = usize;

trait Capacity:
    PartialEq + PartialOrd + Ord + Eq + Copy + Sub<Output = Self> + SubAssign + Add + AddAssign
{
    fn zero() -> Self;
    fn max_val() -> Self;
}

impl Capacity for OrderedFloat<f64> {
    fn zero() -> Self {
        num_traits::zero()
    }

    fn max_val() -> Self {
        OrderedFloat(f64::MAX)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Edge<C> {
    s: Node,
    t: Node,
    id: EdgeId,
    capacity: C,
}

impl<C> Edge<C> {
    fn new(s: Node, t: Node, id: EdgeId, capacity: C) -> Self {
        Self { s, t, id, capacity }
    }

    fn s(&self) -> Node {
        self.s
    }
    fn t(&self) -> Node {
        self.t
    }
    fn id(&self) -> EdgeId {
        self.id
    }
    fn capacity(&self) -> &C {
        &self.capacity
    }
}

#[derive(Clone, Debug)]
enum ResidualEdge<C> {
    Directed(Edge<C>),
    Reversed(Edge<C>),
}

impl<C> ResidualEdge<C> {
    fn as_edge(&self) -> &Edge<C> {
        match self {
            Self::Directed(e) => e,
            Self::Reversed(e) => e,
        }
    }

    fn into_edge(self) -> Edge<C> {
        match self {
            Self::Directed(e) => e,
            Self::Reversed(e) => e,
        }
    }
}

struct Network<C> {
    outgoing: Vec<Option<Vec<(EdgeId, Node)>>>, // outgoing edges
    incoming: Vec<Option<Vec<(EdgeId, Node)>>>, // incoming edges reversed
    capacities: Vec<C>, // index by the edge id, which is unique for each edge in the _input_ graph. Reversed edge have the same id
    num_edges: usize,
}

impl<C> Network<C> {
    fn empty() -> Self {
        Network {
            outgoing: Vec::new(),
            incoming: Vec::new(),
            capacities: Vec::new(),
            num_edges: 0,
        }
    }

    fn add_edge(&mut self, from: Node, to: Node, cap: C) {
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

    fn num_edges(&self) -> usize {
        self.num_edges
    }

    fn num_vertices(&self) -> usize {
        self.outgoing.len().max(self.incoming.len())
    }

    fn capacity(&self, id: EdgeId) -> &C {
        self.capacities.get(id).unwrap()
    }

    fn res_adjacent<'a>(&'a self, node: Node, flow: &'a Flow<C>) -> ResAdjacentEdges<C> {
        ResAdjacentEdges {
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
}

/// Iterator over edges in the residual network.
struct ResAdjacentEdges<'a, C> {
    node: Node,
    flow: &'a Flow<C>,
    capacities: &'a Vec<C>,
    outgoing_edge_iter: Option<std::slice::Iter<'a, (EdgeId, Node)>>,
    incoming_edge_iter: Option<std::slice::Iter<'a, (EdgeId, Node)>>,
}

impl<'a, C> Iterator for ResAdjacentEdges<'a, C>
where
    C: Capacity,
{
    type Item = ResidualEdge<C>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(outgoing_edge_iter) = &mut self.outgoing_edge_iter {
            if let Some((id, t)) = outgoing_edge_iter.next() {
                let cap = self.capacities[*id];
                let flow = self.flow[*id];
                if flow < cap {
                    return Some(ResidualEdge::Directed(Edge::new(self.node, *t, *id, cap - flow)));
                }
            }
        }
        if let Some(incoming_edge_iter) = &mut self.incoming_edge_iter {
            if let Some((id, s)) = incoming_edge_iter.next() {
                let flow = self.flow[*id];
                if flow > C::zero() {
                    return Some(ResidualEdge::Reversed(Edge::new(self.node, *s, *id, flow)));
                }
            }
        }
        None
    }
}

struct Flow<C> {
    flow_value: Vec<C>,
}

impl<C> Flow<C>
where
    C: Capacity,
{
    fn zero_flow(m: usize) -> Self {
        Flow {
            flow_value: vec![C::zero(); m],
        }
    }
}

impl<C> Index<EdgeId> for Flow<C> {
    type Output = C;

    fn index(&self, index: EdgeId) -> &Self::Output {
        &self.flow_value[index]
    }
}

impl<C> IndexMut<EdgeId> for Flow<C> {
    fn index_mut(&mut self, index: EdgeId) -> &mut Self::Output {
        &mut self.flow_value[index]
    }
}

struct Path<C> {
    // Make this generic over 'edge-like' types
    edges: Vec<ResidualEdge<C>>,
}

impl<C> Path<C> {
    fn from_pred(mut pred: Vec<Option<ResidualEdge<C>>>, path_length: usize, t: Node) -> Path<C> {
        let mut edges = Vec::<ResidualEdge<C>>::with_capacity(path_length);
        let mut current = t;
        while let Some(e) = std::mem::replace(&mut pred[current], None) {
            current = e.as_edge().s();
            edges.push(e);
        }
        edges.reverse();
        Path { edges }
    }
}

impl<'a, C> IntoIterator for &'a Path<C> {
    type IntoIter = std::slice::Iter<'a, ResidualEdge<C>>;
    type Item = &'a ResidualEdge<C>;

    fn into_iter(self) -> Self::IntoIter {
        self.edges.iter()
    }
}

trait MaxFlowAlgorithm<C>
where
    C: Capacity,
{
    fn max_flow(network: Network<C>, s: Node, t: Node) -> (C, Flow<C>);
}

struct EdmondsKarp;

impl<C> MaxFlowAlgorithm<C> for EdmondsKarp
where
    C: Capacity,
{
    fn max_flow(network: Network<C>, s: Node, t: Node) -> (C, Flow<C>) {
        let mut flow = Flow::<C>::zero_flow(network.num_edges());
        let mut flow_value = C::zero();

        loop {
            let mut pred: Vec<Option<ResidualEdge<C>>> = vec![None; network.num_vertices()];
            let mut path_length: usize = 0;
            let mut queue = VecDeque::<Node>::new();
            queue.push_back(s);
            while let Some(v) = queue.pop_front() {
                path_length += 1;
                for e in network.res_adjacent(v, &flow) {
                    let u = e.as_edge().t();
                    if pred[u].is_none() && u != s {
                        pred[u] = Some(e);
                        queue.push_back(u);
                    }
                    if u == t {
                        break;
                    }
                }
            }

            if pred[t].is_some() {
                let path = Path::from_pred(pred, path_length, t);
                let mut aug_value = C::max_val();
                for e in &path {
                    aug_value = aug_value.min(*e.as_edge().capacity());
                }
                flow_value += aug_value;
                for e in &path {
                    match e {
                        ResidualEdge::Directed(edge) => flow[edge.id()] += aug_value,
                        ResidualEdge::Reversed(edge) => flow[edge.id()] -= aug_value,
                    }
                }
            } else {
                break;
            }
        }

        (flow_value, flow)
    }
}




#[cfg(test)]
mod test_network {
    use super::*;

    #[test]
    fn test_residual_edges() {
        let mut N = Network::<OrderedFloat<f64>>::empty();
        N.add_edge(0, 1, 6.0.into());
        N.add_edge(0, 2, 2.0.into());
        N.add_edge(1, 2, 5.0.into());
        N.add_edge(1, 3, 3.0.into());
        N.add_edge(2, 3, 4.0.into());

        assert_eq!(5, N.num_edges());
        assert_eq!(4, N.num_vertices());

        let mut flow = Flow::<OrderedFloat<f64>>::zero_flow(N.num_edges());

        {
            let mut iter = N.res_adjacent(2, &flow);
            assert_eq!(iter.next().unwrap().into_edge(), Edge::new(2, 3, 4, 4.0.into()));
            assert!(iter.next().is_none());
        }

        flow[1] = 2.0.into();
        flow[4] = 2.0.into();

        {
            let mut iter = N.res_adjacent(2, &flow);
            assert_eq!(iter.next().unwrap().into_edge(), Edge::new(2, 3, 4, 2.0.into()));
            assert_eq!(iter.next().unwrap().into_edge(), Edge::new(2, 0, 1, 2.0.into()));
            assert!(iter.next().is_none());
        }

        flow[0] = 3.0.into();
        flow[3] = 3.0.into();

        {
            let mut iter = N.res_adjacent(1, &flow);
            assert_eq!(iter.next().unwrap().into_edge(), Edge::new(1, 2, 2, 5.0.into()));
            assert_eq!(iter.next().unwrap().into_edge(), Edge::new(1, 0, 0, 3.0.into()));
            assert!(iter.next().is_none());
        }
    }
}

#[cfg(test)]
mod test_max_flow {
    use super::*;

    fn network() -> Network::<OrderedFloat<f64>> {
        let mut N = Network::<OrderedFloat<f64>>::empty();
        N.add_edge(0, 1, 6.0.into());
        N.add_edge(0, 2, 2.0.into());
        N.add_edge(1, 2, 5.0.into());
        N.add_edge(1, 3, 3.0.into());
        N.add_edge(2, 3, 4.0.into());
        N
    }

    #[test]
    fn test_edmonds_karp() {
        let N = network();
        let (flow_value, flow) = EdmondsKarp::max_flow(N, 0, 3);
        assert_eq!(flow_value, OrderedFloat(7.0));
        assert_eq!(flow[0], OrderedFloat(5.0));
        assert_eq!(flow[1], OrderedFloat(2.0));
        assert_eq!(flow[2], OrderedFloat(2.0));
        assert_eq!(flow[3], OrderedFloat(3.0));
        assert_eq!(flow[4], OrderedFloat(4.0));
    }
}