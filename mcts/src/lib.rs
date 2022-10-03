use std::collections::HashMap;

use pyo3::{prelude::*, types::PyDict};

mod mcts;
use mcts::Config;

// PYTHON MODULE
// ---------------------------------------------------------
// ---------------------------------------------------------
#[pymodule]
fn mcts(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<MCTS>()?;
    m.add_class::<Node>()?;
    m.add_class::<MinMaxStats>()?;

    Ok(())
}

#[pyclass]
pub struct MCTS {
    mcts: mcts::MCTS
}

#[pymethods]
impl MCTS {
    #[new]
    fn new(py: Python, config: &PyDict) -> Self {
        let config = convert_py_config(py, config);
        MCTS {
            mcts: mcts::MCTS::new(config)
        }
    }

    fn new_node(&mut self, prior: f64, idx: usize) -> PyResult<usize> {
        self.mcts.arena.push(mcts::Node::new(prior, idx));
        Ok(idx)
    }

    fn get_node_value(&self, idx: usize) -> PyResult<f64> {
        Ok(self.mcts.arena[idx].value().clone())
    }

    fn get_hidden_state(&self, idx: usize) -> PyResult<Option<PyObject>> {
        Ok(
            Some(
                self.mcts.hidden_states[self.mcts.arena[idx].hidden_state.unwrap()].clone()
            )
        )
    }
    
    pub fn sum_child_visit_count(&self, idx: usize) -> PyResult<Vec<f64>> {
        Ok(self.mcts.sum_child_visit_count(&idx))
    }

    pub fn visit_counts(&self, idx: usize ) -> PyResult<Vec<u32>> {
        Ok(self.mcts.visit_counts(&idx))
    }

    pub fn actions(&self, idx: usize) -> PyResult<Vec<u32>> {
        Ok(self.mcts.actions(&idx))
    }

    fn expand(
        &mut self,
        py: Python,
        idx: usize,
        actions: Vec<u32>,
        to_play: u8,
        reward: f64,
        policy_logits: PyObject,
        hidden_state: Option<PyObject>
    ) {
        let hidden_state = if hidden_state.is_some() {
            self.mcts.hidden_states.push(hidden_state.unwrap());
            Some(self.mcts.hidden_states.len() - 1)
        } else {
            None
        };

        let utilities =  PyModule::from_code(py, r#"
import torch

def get_policy_values(policy_logits, actions):
    return torch.softmax(
        torch.tensor([policy_logits[0][a] for a in actions]), dim=0
    ).tolist()

                "#, "utilities.py", "utilities").unwrap();

        mcts::expand(
            idx,
            &actions,
            &to_play,
            &reward,
            &policy_logits,
            Some(&hidden_state.unwrap()),
            &utilities,
            &mut self.mcts.arena
        )
    }

    fn run(
        &mut self,
        py: Python, 
        model: PyObject,
        observation: Option<Vec<Vec<Vec<i8>>>>,
        legal_actions: Vec<u32>,
        to_play: u8,
        add_exploration_noise: bool,
        test_mode: Option<bool>,
        override_root_with: Option<usize>,
        onnx_model: Option<PyObject>,
        onnx_device: Option<PyObject>
    ) -> PyResult<(Node, HashMap<String, f64>)> {
        let (root, info) = self.mcts.run(
            py, 
            &model,
            &observation,
            &legal_actions,
            &to_play,
            &add_exploration_noise,
            &test_mode.unwrap_or(false),
            override_root_with,
            onnx_model,
            onnx_device
        );
        
        let node = self.mcts.arena[root].clone();

        Ok((Node {node}, info))
    }
}

#[derive(Clone)]
#[pyclass]
struct Node {
    node: mcts::Node
}

#[pymethods]
impl Node {
    #[new]
    fn new(prior: f64, idx: usize) -> Self {
        Node {
            node: mcts::Node::new(prior, idx)
        }
    }
    
    #[getter]
    fn get_idx(&self) -> PyResult<usize> {
        Ok(self.node.idx)
    }

    #[getter]
    fn get_visit_count(&self) -> PyResult<u32> {
        Ok(self.node.visit_count)
    }

    #[getter]
    fn get_to_play(&self) -> PyResult<i8> {
        Ok(self.node.to_play)
    }

    #[getter]
    fn get_prior(&self) -> PyResult<f64> {
        Ok(self.node.prior)
    }
    
    #[getter]
    fn get_value_sum(&self) -> PyResult<f64> {
        Ok(self.node.value_sum)
    }

    #[getter]
    fn get_reward(&self) -> PyResult<f64> {
        Ok(self.node.reward)
    }

    fn value(&self) -> PyResult<f64> {
        Ok(self.node.value())
    }
}

#[pyclass]
struct MinMaxStats {
    min_max_stats: mcts::MinMaxStats
}

#[pymethods]
impl MinMaxStats {

    #[new]
    fn new() -> Self {
        MinMaxStats {
            min_max_stats: mcts::MinMaxStats::new()
        }
    }

    fn update(&mut self, value: f64) {
        self.min_max_stats.update(value)
    }

    fn normalize(&self, value: f64) -> PyResult<f64> {
        Ok(self.min_max_stats.normalize(value))
    }
} 

fn convert_py_config(_py: Python, config: &PyDict) -> Config {
    Config {
        action_space: config.get_item("action_space").unwrap().extract().unwrap(),
        players: config.get_item("players").unwrap().extract().unwrap(),
        num_simulations: config.get_item("num_simulations").unwrap().extract().unwrap(),
        discount: config.get_item("discount").unwrap().extract().unwrap(),
        root_dirichlet_alpha: config.get_item("root_dirichlet_alpha").unwrap().extract().unwrap(),
        root_exploration_fraction: config.get_item("root_exploration_fraction").unwrap().extract().unwrap(),
        pb_c_base: config.get_item("pb_c_base").unwrap().extract().unwrap(),
        pb_c_init: config.get_item("pb_c_init").unwrap().extract().unwrap(),
        support_size: config.get_item("support_size").unwrap().extract().unwrap()
    }
}