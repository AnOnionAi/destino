use rand::prelude::*;
use rand_distr::Dirichlet;
use std::{iter, collections::{HashMap, BTreeMap, HashSet}};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[path ="./config.rs"] mod config;
pub use config::Config;

#[derive(FromPyObject)]
struct RustyTuple(PyObject, PyObject, PyObject, PyObject);

pub struct MCTS {
    /* Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node. */
    
    config: Config,
    pub arena: Vec<Node>,
    pub hidden_states: Vec<PyObject>
}

impl MCTS {
    pub fn new(config: Config) -> Self {
        MCTS { 
            config,
            arena: Vec::new(),
            hidden_states: Vec::new()
        }
    }

    pub fn run(
        &mut self,
        py: Python, 
        model: &PyObject,
        observation: &Option<Vec<Vec<Vec<i8>>>>,
        legal_actions: &Vec<u32>,
        to_play: &u8,
        add_exploration_noise: &bool,
        test_mode: &bool,
        override_root_with: Option<usize>,
        onnx_model: Option<PyObject>,
        onnx_device: Option<PyObject>
    ) -> (usize, HashMap<String, f64>) {
        /* At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network. */
        let utilities =  PyModule::from_code(py, r#"
import torch
import models

def tensor_float_unsqueeze_0(tensor):
    return torch.tensor(tensor).float().unsqueeze(0)

def next_parameter_device(parameters):
    return next(parameters).device

def recurrent_inference(model, hidden_state, action):
    return model.recurrent_inference(
        hidden_state,
        torch.tensor([[action]]).to(hidden_state.device)
    )

def support_to_scalar(logits, support_size):
    return models.support_to_scalar(logits, support_size).item()

def get_policy_values(policy_logits, actions):
    return torch.softmax(
        torch.tensor([policy_logits[0][a] for a in actions]), dim=0
    ).tolist()

                "#, "utilities.py", "utilities").unwrap();

        let root: usize;
        let root_predicted_value_f64: Option<f64>;

        if override_root_with.is_some() {
            root = 0;
            root_predicted_value_f64 = None;
        } else {
            self.arena.push(Node::new(0f64, self.arena.len()));
            root = 0;

            let args = PyTuple::new(py, &[observation]);
            let mut observation = utilities.getattr("tensor_float_unsqueeze_0")
                .unwrap().call1(args).unwrap();

            if onnx_model.is_some() && onnx_model.unwrap().extract(py).unwrap() {
                let args = PyTuple::new(py, &[onnx_device]);
                observation = observation.call_method1("to", args).unwrap();
            } else {
                let parameters = model.call_method0(py, "parameters").unwrap();
                let args = PyTuple::new(py, &[parameters]);
                let device = utilities.getattr("next_parameter_device")
                    .unwrap().call1(args).unwrap();
                let args = PyTuple::new(py, &[device]);
                observation = observation.call_method1("to", args).unwrap();
            }

            let args = (observation,);
            let initial_inference_tuple: RustyTuple = model.call_method1(py, "initial_inference", args)
                .unwrap().extract(py).unwrap();

            let root_predicted_value = Some(initial_inference_tuple.0);
            let reward = initial_inference_tuple.1;
            let policy_logits = initial_inference_tuple.2;
            let hidden_state = self.hidden_states.len();
            self.hidden_states.push(initial_inference_tuple.3);

            let args = (root_predicted_value, self.config.support_size);
            root_predicted_value_f64 = utilities.getattr("support_to_scalar")
                .unwrap().call1(args).unwrap().extract().unwrap();
            let args = (reward, self.config.support_size);
            let reward: f64 = utilities.getattr("support_to_scalar")
                .unwrap().call1(args).unwrap().extract().unwrap();    

            assert!(!legal_actions.is_empty() , "Legal actions should not be an empty array. Got {legal_actions:?}");
            assert!(HashSet::<u32>::from_iter(legal_actions.iter().cloned()).is_subset(
                &HashSet::from_iter(self.config.action_space.iter().cloned())
                    ), "Legal actions should be a subset of the action space.");
            
            expand(root, legal_actions, to_play, &reward, &policy_logits, Some(&hidden_state), &utilities, &mut self.arena);
        }   
        if *add_exploration_noise {
            self.add_exploration_noise(root, self.config.root_dirichlet_alpha, self.config.root_exploration_fraction);
        }
        let mut min_max_stats = MinMaxStats::new();
        let mut max_tree_depth: u32 = 0;

        for _ in 0..self.config.num_simulations {
            let mut virtual_to_play = to_play;
            let mut node = root;
            let mut search_path = vec![node];
            let mut current_tree_depth = 0;

            let mut action = None;
            loop {                
                if !self.arena[node].expanded() { break; }                
                current_tree_depth += 1;                
                let child = self.select_child(node, &min_max_stats, test_mode, &self.arena);                
                action = Some(child.0);
                node = child.1;
                search_path.push(node);

                // Players play turn by turn
                if *virtual_to_play as usize + 1 < self.config.players.len() {
                    virtual_to_play = &self.config.players[*virtual_to_play as usize + 1]
                } else {
                    virtual_to_play = &self.config.players[0];
                }
            }
            // Inside the search tree we use the dynamics function to obtain the next hidden
            // state given an action and the previous hidden state
            let parent = search_path[search_path.len() - 2];

            let hidden_state = if self.arena[parent].hidden_state.is_some() {
                Some(&self.hidden_states[self.arena[parent].hidden_state.unwrap()])
            } else {
                None
            };
            let args = (model, hidden_state, action);
            let recurrent_inference: RustyTuple = utilities.getattr("recurrent_inference").unwrap()
                .call1(args).unwrap().extract().unwrap();
            let value = recurrent_inference.0;
            let reward = recurrent_inference.1;
            let policy_logits = recurrent_inference.2;
            let hidden_state = self.hidden_states.len();
            self.hidden_states.push(recurrent_inference.3);

            let args = (value, self.config.support_size);
            let value: f64 = utilities.getattr("support_to_scalar")
                .unwrap().call1(args).unwrap().extract().unwrap();
            let args = (reward, self.config.support_size);
            let reward: f64 = utilities.getattr("support_to_scalar")
                .unwrap().call1(args).unwrap().extract().unwrap();
            expand(
                node,
                &self.config.action_space, 
                virtual_to_play, 
                &reward, 
                &policy_logits, 
                Some(&hidden_state),
                &utilities,
                &mut self.arena
            );            
            
            backpropagate(&search_path, &value, virtual_to_play, &mut min_max_stats, &mut self.arena, &self.config);
            
            max_tree_depth = max_tree_depth.max(current_tree_depth);
        }

        let mut extra_info = HashMap::new();
        extra_info.insert("max_tree_depth".to_string(), max_tree_depth as f64);
        extra_info.insert("root_predicted_value".to_string(), root_predicted_value_f64.unwrap());

        (root, extra_info)
    }

    fn select_child(&self, node: usize, min_max_stats: &MinMaxStats, test_mode: &bool, arena: &Vec<Node>) -> (u32, usize) {
        // Select the child with the highest UCB score.

        let mut rng = rand::thread_rng();

        let max_ucb = arena[node].children.values()
            .map(|child| self.ucb_score(node, *child, min_max_stats, &arena))
                .fold(f64::MIN, f64::max);
        let actions: Vec<u32> = arena[node].children.iter()
            .filter(|(_, child)| self.ucb_score(node, **child, min_max_stats, &arena) == max_ucb)
                .map(|(action, _)| *action).collect();
        let action = if *test_mode {
                actions[0]
            } else {
                *actions.choose(&mut rng).unwrap()
            };

        (action, arena[node].children[&action])
    }

    fn ucb_score(&self, parent: usize, child: usize, min_max_stats: &MinMaxStats, arena: &Vec<Node>) -> f64 {
        // The score for a node is based on its value, plus an exploration bonus based on the prior.

        let mut pb_c = (
                (arena[parent].visit_count as f64 + self.config.pb_c_base + 1f64) / self.config.pb_c_base
            ).ln() + self.config.pb_c_init;
        pb_c *= (arena[parent].visit_count as f64).sqrt() / (arena[child].visit_count + 1) as f64;

        let prior_score = pb_c * arena[child].prior;

        let value_score = if arena[child].visit_count > 0 {
            // Mean value Q
            min_max_stats.normalize(
                arena[child].reward
                + self.config.discount
                * (if self.config.players.len() == 1 {arena[child].value()} else {-arena[child].value()})
            )
        } else {
            0f64
        };

        prior_score + value_score
    }

    pub fn sum_child_visit_count(&self, idx: &usize) -> Vec<f64> {
        let sum_visits = self.arena[*idx].children.values()
            .fold(0u32, |sum, i| sum + self.arena[*i].visit_count);
        self.config.action_space.iter().map(|a| {
            if self.arena[*idx].children.contains_key(a) {
                self.arena[self.arena[*idx].children[a]].visit_count as f64 / sum_visits as f64
            } else {
                0f64
            }
        }).collect()
    }

    pub fn visit_counts(&self, idx: &usize ) -> Vec<u32> {
        self.arena[*idx].children.values().map(|child| self.arena[*child].visit_count).collect()
    }

    pub fn actions(&self, idx: &usize) -> Vec<u32> {
        self.arena[*idx].children.clone().into_keys().collect()
    }

    fn add_exploration_noise(&mut self, idx: usize, dirichlet_alpha: f64, exploration_fraction: f64) {
        /* At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions. */
        if self.arena[idx].children.len() < 2 { return; }

        let children = self.arena[idx].children.clone();
        let actions = children.keys();

        let alpha: Vec<f64> = iter::repeat(dirichlet_alpha).take(actions.len()).collect();
        let dirichlet = Dirichlet::new(&alpha).unwrap();
        let noise = dirichlet.sample(&mut rand::thread_rng());
        
        let frac = exploration_fraction;
        for (a, n) in actions.zip(noise.iter()) {
            if let Some(idx) = children.get(a) {
                self.arena[*idx].prior = self.arena[*idx].prior * (1f64 - frac) + n * frac;
            }
        }
    }
}

#[derive(Debug)]
#[derive(Clone)]
pub struct Node {
    pub idx: usize,
    pub visit_count: u32,
    pub to_play: i8,
    pub prior: f64,
    pub value_sum: f64,
    pub children: BTreeMap<u32, usize>,
    pub hidden_state: Option<usize>,
    pub reward: f64
}

impl Node {
    pub fn new(prior: f64, idx: usize) -> Self {
        Node {
            idx,
            visit_count: 0,
            to_play: -1,
            prior,
            value_sum: 0f64,
            children: BTreeMap::new(),
            hidden_state: None,
            reward: 0f64
        }
    }

    fn expanded(&self) -> bool {
        self.children.len() > 0
    }

    pub fn value(&self) -> f64 {
        if self.visit_count == 0 {
            return 0f64;
        }
        self.value_sum as f64 / self.visit_count as f64
    }
}

pub struct MinMaxStats {
    maximum: f64,
    minimum: f64
}

impl MinMaxStats {
    pub fn new() -> Self {
        MinMaxStats {
            maximum: f64::MIN,
            minimum: f64::MAX
        }
    }

    pub fn update(&mut self, value: f64) {
        self.maximum = self.maximum.max(value);
        self.minimum = self.minimum.min(value);
    }

    pub fn normalize(&self, value: f64) -> f64 {
        if self.maximum > self.minimum {
            // We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum);
        }
        value
    }
}

pub fn expand(
    idx: usize,
    actions: &Vec<u32>,
    to_play: &u8,
    reward: &f64,
    policy_logits: &PyObject,
    hidden_state: Option<&usize>,
    utilities: &PyModule,
    arena: &mut Vec<Node>
) {
    /* We expand a node using the value, reward and policy prediction obtained from the
    neural network. */
    
    arena[idx].to_play = *to_play as i8;
    arena[idx].reward = *reward;
    arena[idx].hidden_state = Some(*hidden_state.unwrap());

    let args = (policy_logits, actions.clone());
    let policy_values: Vec<f64> = utilities.getattr("get_policy_values")
        .unwrap().call1(args).unwrap().extract().unwrap();
    let policy: HashMap<&u32, f64> = actions.iter().enumerate()
        .map(|(index, action)| (action, policy_values[index])).collect();
    let mut new_idx = arena.len();
    for (action, p) in &policy {                                                                  
        arena.push(Node::new(*p, new_idx));
        arena[idx].children.insert(**action, new_idx);
        new_idx += 1;
    }
}

pub fn backpropagate(
    search_path: &Vec<usize>,
    value: &f64,
    to_play: &u8,
    min_max_stats: &mut MinMaxStats,
    arena: &mut Vec<Node>,
    config: &Config
) {
    /* At the end of a simulation, we propagate the evaluation all the way up the tree
    to the root. */

    let mut value = *value;

    if config.players.len() == 1 {
        for node in search_path.iter().rev() {
            arena[*node].value_sum += value;
            arena[*node].visit_count += 1;
            min_max_stats.update(arena[*node].reward + config.discount * arena[*node].value());

            value = arena[*node].reward + config.discount * value;
        }
    } else if config.players.len() == 2 {
        for node in search_path.iter().rev() {
            arena[*node].value_sum += if arena[*node].to_play == *to_play as i8 { value } else { -value };
            arena[*node].visit_count += 1;
            min_max_stats.update(arena[*node].reward + config.discount * -arena[*node].value());

            value = if arena[*node].to_play == *to_play as i8 { -arena[*node].reward } else { arena[*node].reward } + config.discount * value;
        }
    } else {
        panic!("More than two player mode not implemented.");
    }
}