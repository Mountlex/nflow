use std::collections::VecDeque;

use crate::{
    network::{self, Network},
    Capacity, Flow, Node, ResidualEdge,
};

// Acyclic, connected s-t network
pub struct LevelGraph<'a, C> {
    levels: Vec<Option<usize>>,
    network: &'a Network<C>,
    flow: &'a Flow<C>,
}

impl<'a, C> LevelGraph<'a, C>
where
    C: Capacity,
{
    pub fn init(s: Node, t: Node, network: &'a Network<C>, flow: &'a Flow<C>) -> Option<Self> {
        let mut levels = vec![None; network.num_vertices()];
        let mut queue = VecDeque::<Node>::new();
        queue.push_back(s);
        levels[s] = Some(0);

        while let Some(v) = queue.pop_front() {
            for edge in network.res_adjacent(v, flow) {
                let l = levels[v].unwrap();
                let u = edge.as_edge().t();
                if levels[u].is_none() {
                    // u not visited yet
                    levels[u] = Some(l + 1);
                    queue.push_back(u);
                }
            }
        }

        if levels[t].is_some() {
            Some(LevelGraph {
                levels,
                network,
                flow,
            })
        } else {
            None
        }
    }

    pub fn base_network(&self) -> &'a Network<C> {
        self.network
    }

    pub fn adjacent(&'a self, node: Node) -> AdjacentEdges<'a, C> {
        AdjacentEdges {
            levels: &self.levels,
            res_iter: self.network.res_adjacent(node, self.flow),
        }
    }
}

pub struct AdjacentEdges<'a, C> {
    levels: &'a Vec<Option<usize>>,
    res_iter: network::ResidualAdjacentEdges<'a, C>,
}

impl<'a, C> Iterator for AdjacentEdges<'a, C>
where
    C: Capacity,
{
    type Item = ResidualEdge<C>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(edge) = self.res_iter.next() {
            if let Some(ls) = self.levels[edge.as_edge().s()] {
                if let Some(lt) = self.levels[edge.as_edge().t()] {
                    if ls + 1 == lt {
                        return Some(edge);
                    }
                }
            }
        }

        None
    }
}
