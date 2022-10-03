pub struct Config {
    pub action_space: Vec<u32>,
    pub players: Vec<u8>,
    pub num_simulations: u32,
    pub discount: f64,
    pub root_dirichlet_alpha: f64,
    pub root_exploration_fraction: f64,
    pub pb_c_base: f64,
    pub pb_c_init: f64,
    pub support_size: u32
}