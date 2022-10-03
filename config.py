import torch
from omegaconf import OmegaConf
import random


class Config:
    def __init__(self, game_name):
        config = OmegaConf.load("games/config/" + str(game_name) + ".yaml")

        ### Game
        self.game_name = str(game_name)

        self.seed = config.seed  # Seed for numpy, torch and the game
        self.max_num_gpus = (
            config.max_num_gpus
        )  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available
        self.selfplay_on_gpu = config.selfplay_on_gpu
        self.train_on_gpu = (
            torch.cuda.is_available() if config.train_on_gpu == "auto" else False
        )  # Train on GPU if available

        self.observation_shape = (
            config.one_dim,
            config.two_dim,
            config.three_dim,
        )  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(
            range(config.action_space)
        )  # Fixed list of all possible actions. You should only edit the length
        self.players = list(
            range(config.players)
        )  # List of players. You should only edit the length
        self.stacked_observations = (
            config.stacked_observations
        )  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = (
            random.choice(self.players)
            if config.muzero_player == "random"
            else config.muzero_player
        )  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = (
            config.opponent
        )  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = (
            config.num_workers
        )  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.max_moves = (
            config.max_moves
        )  # Maximum number of moves if game is not finished before
        self.num_simulations = (
            config.num_simulations
        )  # Number of future moves self-simulated
        self.discount = config.discount  # Chronological discount of the reward
        self.temperature_threshold = (
            config.temperature_threshold
        )  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = config.root_dirichlet_alpha
        self.root_exploration_fraction = config.root_exploration_fraction

        # UCB formula
        self.pb_c_base = config.pb_c_base
        self.pb_c_init = config.pb_c_init

        ### Network
        self.network = config.network  # "resnet" / "fullyconnected"
        self.support_size = (
            config.support_size
        )  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = (
            config.downsample
        )  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = config.blocks  # Number of blocks in the ResNet
        self.channels = config.channels  # Number of channels in the ResNet
        self.reduced_channels_reward = (
            config.reduced_channels_reward
        )  # Number of channels in reward head
        self.reduced_channels_value = (
            config.reduced_channels_value
        )  # Number of channels in value head
        self.reduced_channels_policy = (
            config.reduced_channels_policy
        )  # Number of channels in policy head
        self.resnet_fc_reward_layers = list(
            config.resnet_fc_reward_layers
        )  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = list(
            config.resnet_fc_value_layers
        )  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = list(
            config.resnet_fc_policy_layers
        )  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = config.encoding_size
        self.fc_representation_layers = list(
            config.fc_representation_layers
        )  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = list(
            config.fc_dynamics_layers
        )  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = list(
            config.fc_reward_layers
        )  # Define the hidden layers in the reward network
        self.fc_value_layers = list(
            config.fc_value_layers
        )  # Define the hidden layers in the value network
        self.fc_policy_layers = list(
            config.fc_policy_layers
        )  # Define the hidden layers in the policy network

        ### Training
        self.training_steps = (
            config.training_steps
        )  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = (
            config.batch_size
        )  # Number of parts of games to train on at each training step
        self.checkpoint_interval = (
            config.checkpoint_interval
        )  # Number of training steps before using the model for self-playing
        self.value_loss_weight = (
            config.value_loss_weight
        )  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.optimizer = config.optimizer  # "Adam", "AdamW" or "SGD". Paper uses SGD
        self.weight_decay = (
            config.weight_decay
        )  # L2 weights regularization for Adam. AdamW - weight decay coefficient
        self.momentum = config.momentum  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = config.lr_init  # Initial learning rate
        self.lr_decay_rate = (
            config.lr_decay_rate
        )  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = config.lr_decay_steps

        ### Replay Buffer
        self.replay_buffer_size = (
            config.replay_buffer_size
        )  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = (
            config.num_unroll_steps
        )  # Number of game moves to keep for every batch element
        self.td_steps = (
            config.td_steps
        )  # Number of steps in the future to take into account for calculating the target value
        self.PER = (
            config.PER
        )  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = (
            config.PER_alpha
        )  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = (
            config.use_last_model_value
        )  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)

        # self.num_reanalyse_workers = config.num_reanalyse_workers
        # self.value_target_update_freq = (
        #     config.value_target_update_freq
        # )  # Update frequency of the target model used to provide fresher value (and possibly policy) estimates
        # self.use_updated_mcts_value_targets = (
        #     config.use_updated_mcts_value_targets
        # )  # If True, root values targets are updated according to the re-execution of the MCTS (in this case, lagging parameters are used to run the MCTS to stabilize bootstrapping). Otherwise, a lagging value of the network (representation & value) is used to obtain the updated value targets.

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = (
            config.self_play_delay
        )  # Number of seconds to wait after each played game
        self.training_delay = (
            config.training_delay
        )  # Number of seconds to wait after each training step
        self.ratio = (
            config.ratio
        )  # None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """

        # e = 1 / self.training_steps * trained_steps
        # if 1 - e > 0.05:
        #     print(1 - e)
        #     return 1 - e
        # else:
        #     print(0.5)
        #     return 0.05

        return 1
