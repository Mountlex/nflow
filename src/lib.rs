use std::{
    collections::VecDeque,
    fmt::{Display, Formatter},
    iter::Sum,
    ops::{Add, AddAssign, Index, IndexMut, Sub, SubAssign},
};

use fixedbitset::FixedBitSet;
use network::Network;
use ordered_float::OrderedFloat;

use crate::{level::LevelGraph, network::AdjacentEdge};

mod level;
mod network;

type Node = usize;

type EdgeId = usize;

pub trait Capacity:
    PartialEq
    + PartialOrd
    + Ord
    + Eq
    + Copy
    + Sub<Output = Self>
    + SubAssign
    + Add
    + AddAssign
    + Sum
    + Display
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

impl<C> Display for Edge<C>
where
    C: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} --> {} [cap={}, id={}]",
            self.s, self.t, self.capacity, self.id
        )
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

impl<C> Display for ResidualEdge<C>
where
    C: Display,
{
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

    pub fn augment_path_by_min_capacity(&mut self, path: &Path<C>) -> C {
        let aug_value = path.min_capacity();
        self.augment_path_by_value(path, aug_value);
        aug_value
    }

    pub fn flow_value(&self, s: Node, network: &Network<C>) -> C {
        network.outgoing(s).map(|e| self.edge_flows[e.id()]).sum()
    }
}

impl<C> Display for Flow<C>
where
    C: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Flow [")?;
        for (i, e) in self.edge_flows.iter().enumerate() {
            writeln!(f, "[id={}] {}", i, e)?;
        }
        writeln!(f, "]")
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
        let mut cap = C::max_val();
        for e in self {
            cap = cap.min(*e.as_edge().capacity());
        }
        cap
    }

    fn min_aug_value(&self, flow: &Flow<C>) -> C
    where
        C: Capacity,
    {
        let mut aug_value = C::max_val();
        for e in self {
            aug_value = aug_value.min(*e.as_edge().capacity() - flow[e.as_edge().id()]);
        }
        aug_value
    }
}

impl<C> Display for Path<C>
where
    C: Display,
{
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
            'bfs: while let Some(v) = queue.pop_front() {
                for e in network.res_adjacent(v, &flow) {
                    let u = e.as_edge().t();
                    if pred[u].is_none() && u != s {
                        pred[u] = Some(e);
                        queue.push_back(u);
                    }
                    if u == t {
                        break 'bfs;
                    }
                }
            }

            if pred[t].is_some() {
                let path = Path::from_pred(pred, t);
                flow.augment_path_by_min_capacity(&path);
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
                let aug_value = path.min_aug_value(&blocking_flow);
                blocking_flow.augment_path_by_value(&path, aug_value);
                for e in &path {
                    if blocking_flow[e.as_edge().id()] == *e.as_edge().capacity() {
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

struct PushRelabel<'a, C> {
    level: Vec<usize>,
    excess: Vec<C>,
    preflow: Flow<C>,
    network: &'a Network<C>,
    current_arc: Vec<usize>,
}

impl<'a, C> PushRelabel<'a, C>
where
    C: Capacity,
{
    fn init(s: Node, t: Node, network: &'a Network<C>) -> Self {
        let mut level = vec![0; network.num_vertices()];
        level[s] = network.num_vertices();

        // Init height function by BFS
        let mut queue = VecDeque::<Node>::new();
        queue.push_back(t);
        while let Some(v) = queue.pop_front() {
            for e in network.adjacent(v) {
                if let AdjacentEdge::Incoming(edge) = e {
                    let u = edge.s();
                    if level[u] == 0 {
                        level[u] = level[v] + 1;
                        queue.push_back(u);
                    }
                }
            }
        }

        let mut preflow = Flow::<C>::zero_flow(network.num_edges());
        let mut excess = vec![C::zero(); network.num_vertices()];
        for e in network.outgoing(s) {
            preflow[e.id()] = *e.capacity();
            excess[e.t()] = *e.capacity();
        }

        PushRelabel {
            level,
            excess,
            preflow,
            network,
            current_arc: vec![0; network.num_vertices()],
        }
    }

    fn height(&self, node: Node) -> usize {
        self.level[node]
    }

    fn preflow(self) -> Flow<C> {
        self.preflow
    }

    fn is_active(&self, v: Node) -> bool {
        self.excess[v] > C::zero()
    }

    fn push(&mut self, edge: &ResidualEdge<C>) {
        let id = edge.as_edge().id();
        let u = edge.as_edge().s();
        let v = edge.as_edge().t();
        debug_assert!(self.excess[u] > C::zero());
        debug_assert!(self.level[u] == self.level[v] + 1);
        let inc = self.excess[u].min(*edge.as_edge().capacity());
        match edge {
            ResidualEdge::Directed(_) => self.preflow[id] += inc,
            ResidualEdge::Reversed(_) => self.preflow[id] -= inc,
        }
        self.excess[u] -= inc;
        self.excess[v] += inc;
        println!(
            "Push {} from {} to {}",
            inc,
            edge.as_edge().s(),
            edge.as_edge().t()
        );
    }

    fn relabel(&mut self, node: Node) {
        debug_assert!(self.excess[node] > C::zero());
        if let Some(min_level) = self
            .network
            .res_adjacent(node, &self.preflow)
            .map(|e| self.level[e.as_edge().t()])
            .min()
        {
            println!(
                "Relabel node {} from {} to {}.",
                node,
                self.level[node],
                1 + min_level
            );
            debug_assert!(self.level[node] <= min_level);
            self.level[node] = 1 + min_level;
        }
    }

    fn discharge(&mut self, node: Node) -> Vec<Node> {
        let mut pushed_nodes = Vec::new();
        while self.excess[node] > C::zero() {
            if let Some(edge) = self.network.adjacent_by_index(node, self.current_arc[node]) {
                if let Some(res_edge) = edge.try_into_residual(&self.preflow) {
                    println!("Discharging {}: Edge {}", node, res_edge.as_edge());
                    if self.level[res_edge.as_edge().s()] == self.level[res_edge.as_edge().t()] + 1
                    {
                        self.push(&res_edge);
                        pushed_nodes.push(res_edge.as_edge().t());
                    } else {
                        self.current_arc[node] += 1;
                    }
                } else {
                    self.current_arc[node] += 1;
                }
            } else {
                debug_assert!(self.current_arc[node] == self.network.num_adjacent_edges(node));
                self.relabel(node);
                self.current_arc[node] = 0;
            }
        }
        pushed_nodes
    }
}

fn fifo_push_relabel<C>(network: &Network<C>, s: Node, t: Node) -> Flow<C>
where
    C: Capacity,
{
    let mut pr = PushRelabel::init(s, t, network);
    let mut queue = VecDeque::<Node>::new();

    for e in network.outgoing(s) {
        if e.t() != t {
            queue.push_back(e.t());
        }
    }

    while let Some(v) = queue.pop_front() {
        println!("Discharging {}...", v);
        let pushed = pr.discharge(v);
        println!("Preflow after discharging {}: {}", v, pr.preflow);
        for u in pushed {
            if u != t && u != s && !queue.contains(&u) {
                queue.push_back(u);
            }
        }
    }

    pr.preflow()
}

fn max_height_push_relabel<C>(network: &Network<C>, s: Node, t: Node) -> Flow<C>
where
    C: Capacity,
{
    let mut pr = PushRelabel::init(s, t, network);

    let mut max_height = 0;
    let mut heights: Vec<Vec<Node>> = vec![vec![]; 2 * network.num_vertices()];
    for e in network.outgoing(s) {
        if e.t() != t {
            let height = pr.height(e.t());
            heights[height].push(e.t());
            max_height = max_height.max(height);
        }
    }

    while let Some(v) = heights[max_height].pop() {
        println!("Discharging {}...", v);
        let pushed = pr.discharge(v);

        if heights[max_height].is_empty() && max_height > 1 {
            max_height -= 2;
        }
        println!("Preflow after discharging {}: {}", v, pr.preflow);
        for u in pushed {
            let height = pr.height(u);
            if u != s && u != t && !heights[height].contains(&u) {
                heights[height].push(u);
                max_height = max_height.max(height);
            }
        }
    }

    pr.preflow()
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

    fn check_network_small(flow: Flow<OrderedFloat<f64>>, n: &Network<OrderedFloat<f64>>) {
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

    fn check_network_large(flow: Flow<OrderedFloat<f64>>, n: &Network<OrderedFloat<f64>>) {
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
    fn test_fifo_small() {
        let n = network_small();
        let flow = fifo_push_relabel(&n, 0, 3);
        assert_eq!(flow.flow_value(0, &n), OrderedFloat(7.0));
        assert_eq!(flow[0], OrderedFloat(6.0));
        assert_eq!(flow[1], OrderedFloat(1.0));
        assert_eq!(flow[2], OrderedFloat(3.0));
        assert_eq!(flow[3], OrderedFloat(3.0));
        assert_eq!(flow[4], OrderedFloat(4.0));
    }

    #[test]
    fn test_max_height_small() {
        let n = network_small();
        let flow = max_height_push_relabel(&n, 0, 3);
        assert_eq!(flow.flow_value(0, &n), OrderedFloat(7.0));
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

    #[test]
    fn test_fifo_large() {
        let n = network_large();
        let flow = fifo_push_relabel(&n, 0, 8);
        assert_eq!(flow.flow_value(0, &n), OrderedFloat(15.0));
    }

    #[test]
    fn test_max_height_large() {
        let n = network_large();
        let flow = max_height_push_relabel(&n, 0, 8);
        assert_eq!(flow.flow_value(0, &n), OrderedFloat(15.0));
    }
}
