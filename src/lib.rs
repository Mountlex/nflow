use std::{collections::VecDeque, fmt::{Display, Formatter}, iter::Sum, ops::{Add, AddAssign, Index, IndexMut, Sub, SubAssign}};

use fixedbitset::FixedBitSet;
use network::Network;
use ordered_float::OrderedFloat;

use crate::level::LevelGraph;

mod level;
mod network;

type Node = usize;

type EdgeId = usize;

pub trait Capacity:
    PartialEq + PartialOrd + Ord + Eq + Copy + Sub<Output = Self> + SubAssign + Add + AddAssign + Sum + Display
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
pub struct Edge<C> {
    s: Node,
    t: Node,
    id: EdgeId,
    capacity: C,
}

impl<C> Edge<C> {
    pub fn new(s: Node, t: Node, id: EdgeId, capacity: C) -> Self {
        Self { s, t, id, capacity }
    }

    pub fn s(&self) -> Node {
        self.s
    }
    pub fn t(&self) -> Node {
        self.t
    }
    pub fn id(&self) -> EdgeId {
        self.id
    }
    pub fn capacity(&self) -> &C {
        &self.capacity
    }
}


impl <C> Display for Edge<C> where C: Display {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} --> {} [cap={}, id={}]", self.s, self.t, self.capacity, self.id)
    }
}

#[derive(Clone, Debug)]
pub enum ResidualEdge<C> {
    Directed(Edge<C>),
    Reversed(Edge<C>),
}

impl<C> ResidualEdge<C> {
    pub fn as_edge(&self) -> &Edge<C> {
        match self {
            Self::Directed(e) => e,
            Self::Reversed(e) => e,
        }
    }

    pub fn into_edge(self) -> Edge<C> {
        match self {
            Self::Directed(e) => e,
            Self::Reversed(e) => e,
        }
    }
}

impl <C> Display for ResidualEdge<C> where C: Display {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ResidualEdge::Directed(e) => write!(f, "ResDi {}", e),
            ResidualEdge::Reversed(e) => write!(f, "ResRev {}", e),
        }
    }
}

pub struct Flow<C> {
    edge_flows: Vec<C>,
}

impl<C> Flow<C>
where
    C: Capacity,
{
    pub fn zero_flow(m: usize) -> Self {
        Flow {
            edge_flows: vec![C::zero(); m],
        }
    }

    pub fn augment_path_by_value(&mut self, path: &Path<C>, value: C) {
        for e in path {
            match e {
                ResidualEdge::Directed(edge) => self[edge.id()] += value,
                ResidualEdge::Reversed(edge) => self[edge.id()] -= value,
            }
        }
    }

    pub fn augment_path(&mut self, path: &Path<C>) -> C {
        let aug_value = path.min_capacity();
        self.augment_path_by_value(path, aug_value);
        aug_value
    }

    pub fn flow_value(&self, s: Node, network: &Network<C>) -> C {
        network.adjacent(s).map(|e| self.edge_flows[e.id()]).sum()
    }
}

impl<C> AddAssign for Flow<C>
where
    C: Capacity,
{
    fn add_assign(&mut self, rhs: Flow<C>) {
        debug_assert!(self.edge_flows.len() == rhs.edge_flows.len());
        for (a, b) in self.edge_flows.iter_mut().zip(rhs.edge_flows) {
            *a += b;
        }
    }
}

impl<C> Index<EdgeId> for Flow<C> {
    type Output = C;

    fn index(&self, index: EdgeId) -> &Self::Output {
        &self.edge_flows[index]
    }
}

impl<C> IndexMut<EdgeId> for Flow<C> {
    fn index_mut(&mut self, index: EdgeId) -> &mut Self::Output {
        &mut self.edge_flows[index]
    }
}

pub struct Path<C> {
    // Make this generic over 'edge-like' types
    edges: Vec<ResidualEdge<C>>,
}

impl<C> Path<C> {
    pub fn from_pred(pred: Vec<Option<ResidualEdge<C>>>, t: Node) -> Path<C> {
        Self::from_pred_helper(pred, Vec::new(), t)
    }

    pub fn from_pred_with_length(
        pred: Vec<Option<ResidualEdge<C>>>,
        path_length: usize,
        t: Node,
    ) -> Path<C> {
        let edges = Vec::<ResidualEdge<C>>::with_capacity(path_length);
        Self::from_pred_helper(pred, edges, t)
    }

    fn from_pred_helper(
        mut pred: Vec<Option<ResidualEdge<C>>>,
        mut edges: Vec<ResidualEdge<C>>,
        t: Node,
    ) -> Path<C> {
        let mut current = t;
        while let Some(e) = std::mem::replace(&mut pred[current], None) {
            current = e.as_edge().s();
            edges.push(e);
        }
        edges.reverse();
        Path { edges }
    }

    fn min_capacity(&self) -> C
    where
        C: Capacity,
    {
        let mut aug_value = C::max_val();
        for e in self {
            aug_value = aug_value.min(*e.as_edge().capacity());
        }
        aug_value
    }
}

impl <C> Display for Path<C> where C: Display {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Path [")?;
        for e in self {
            writeln!(f, "{}", e)?;
        }
        writeln!(f, "]")
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
    fn max_flow(network: &Network<C>, s: Node, t: Node) -> Flow<C>;
}

struct EdmondsKarp;

impl<C> MaxFlowAlgorithm<C> for EdmondsKarp
where
    C: Capacity,
{
    fn max_flow(network: &Network<C>, s: Node, t: Node) -> Flow<C> {
        let mut flow = Flow::<C>::zero_flow(network.num_edges());

        loop {
            let mut pred: Vec<Option<ResidualEdge<C>>> = vec![None; network.num_vertices()];
            let mut queue = VecDeque::<Node>::new();
            queue.push_back(s);
            while let Some(v) = queue.pop_front() {
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
                let path = Path::from_pred(pred,  t);
                println!("{}", path);
                flow.augment_path(&path);
            } else {
                break;
            }
        }

        flow
    }
}

struct Dinitz;

impl Dinitz {
    fn blocking_flow<C>(s: Node, t: Node, level_graph: &LevelGraph<C>) -> Flow<C>
    where
        C: Capacity,
    {
        let mut deleted = FixedBitSet::with_capacity(level_graph.base_network().num_edges());
        let mut blocking_flow = Flow::<C>::zero_flow(level_graph.base_network().num_edges());

        loop {
            let mut stack: Vec<Node> = vec![s];
            let mut pred: Vec<Option<ResidualEdge<C>>> =
                vec![None; level_graph.base_network().num_edges()];

            'dfs: while let Some(v) = stack.pop() {
                let mut retreat = true;
                for e in level_graph.adjacent(v) {
                    if pred[e.as_edge().t()].is_none() && !deleted.contains(e.as_edge().id()) {
                        retreat = false;
                        let u = e.as_edge().t();
                        pred[u] = Some(e);
                        stack.push(u);
                        if u == t {
                            break 'dfs;
                        }
                    }
                }
                if retreat {
                    if v == s {
                        return blocking_flow; // There are no s-t paths left
                    } else {
                        deleted.set(pred[v].as_ref().unwrap().as_edge().id(), true);
                        // delete edge on which we will track back.
                    }
                }
            }

            if Some(&t) == stack.last() {
                let path = Path::from_pred(pred, t);
                let aug_value = blocking_flow.augment_path(&path);
                for e in &path {
                    if &aug_value == e.as_edge().capacity() {
                        deleted.set(e.as_edge().id(), true);
                    }
                }
            }
        }
    }
}

impl<C> MaxFlowAlgorithm<C> for Dinitz
where
    C: Capacity,
{
    fn max_flow(network: &Network<C>, s: Node, t: Node) -> Flow<C> {
        let mut flow = Flow::<C>::zero_flow(network.num_edges());

        while let Some(level_graph) = LevelGraph::init(s, t, &network, &flow) {
            let blocking_flow = Self::blocking_flow(s, t, &level_graph);
            flow += blocking_flow;
        }

        flow
    }
}

#[cfg(test)]
mod test_max_flow {
    use super::*;

    fn network_small() -> Network<OrderedFloat<f64>> {
        let mut n = Network::<OrderedFloat<f64>>::empty();
        n.add_edge(0, 1, 6.0.into());
        n.add_edge(0, 2, 2.0.into());
        n.add_edge(1, 2, 5.0.into());
        n.add_edge(1, 3, 3.0.into());
        n.add_edge(2, 3, 4.0.into());
        n
    }

    fn check_network_small(flow: Flow<OrderedFloat<f64>>, n: &Network<OrderedFloat<f64>>){
        assert_eq!(flow.flow_value(0, &n), OrderedFloat(7.0));
        assert_eq!(flow[0], OrderedFloat(5.0));
        assert_eq!(flow[1], OrderedFloat(2.0));
        assert_eq!(flow[2], OrderedFloat(2.0));
        assert_eq!(flow[3], OrderedFloat(3.0));
        assert_eq!(flow[4], OrderedFloat(4.0));
    }

    fn network_large() -> Network<OrderedFloat<f64>> {
        let mut n = Network::<OrderedFloat<f64>>::empty();
        n.add_edge(0, 1, 9.0.into());
        n.add_edge(0, 2, 5.0.into());
        n.add_edge(0, 3, 6.0.into());
        n.add_edge(1, 5, 3.0.into());
        n.add_edge(1, 6, 7.0.into());
        n.add_edge(2, 3, 2.0.into());
        n.add_edge(2, 5, 9.0.into());
        n.add_edge(3, 4, 1.0.into());
        n.add_edge(4, 8, 8.0.into());
        n.add_edge(5, 4, 7.0.into());
        n.add_edge(5, 7, 3.0.into());
        n.add_edge(6, 7, 8.0.into());
        n.add_edge(7, 8, 7.0.into());
        n
    }

    fn check_network_large(flow: Flow<OrderedFloat<f64>>, n: &Network<OrderedFloat<f64>>){
        assert_eq!(flow.flow_value(0, &n), OrderedFloat(15.0));
        assert_eq!(flow[0], OrderedFloat(9.0));
        assert_eq!(flow[1], OrderedFloat(5.0));
        assert_eq!(flow[2], OrderedFloat(1.0));
        assert_eq!(flow[3], OrderedFloat(3.0));
        assert_eq!(flow[4], OrderedFloat(6.0));
        assert_eq!(flow[5], OrderedFloat(0.0));
        assert_eq!(flow[6], OrderedFloat(5.0));
        assert_eq!(flow[7], OrderedFloat(1.0));
        assert_eq!(flow[8], OrderedFloat(8.0));
        assert_eq!(flow[9], OrderedFloat(7.0));
        assert_eq!(flow[10], OrderedFloat(1.0));
        assert_eq!(flow[11], OrderedFloat(6.0));
        assert_eq!(flow[12], OrderedFloat(7.0));
    }

    #[test]
    fn test_edmonds_karp_small() {
        let n = network_small();
        let flow = EdmondsKarp::max_flow(&n, 0, 3);
        check_network_small(flow, &n);
    }

    #[test]
    fn test_dinitz_small() {
        let n = network_small();
        let flow = Dinitz::max_flow(&n, 0, 3);
        check_network_small(flow, &n);
    }

    #[test]
    fn test_edmonds_karp_large() {
        let n = network_large();
        let flow = EdmondsKarp::max_flow(&n, 0, 8);
        check_network_large(flow, &n);
    }

    #[test]
    fn test_dinitz_large() {
        let n = network_large();
        let flow = Dinitz::max_flow(&n, 0, 8);
        check_network_large(flow, &n);
    }
}
