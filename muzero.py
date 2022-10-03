import contextlib
import copy
import importlib
import math
import os
import datetime
import pickle
import shutil
import numpy
import ray
import torch
import gcloud

# import diagnose_model
import config
import models
import replay_buffer
import play
import shared_storage
import trainer
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf


class MuZero:
    """
    Main class to manage MuZero.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

        config (dict, MuZeroConfig, optional): Override the default config of the game.

        split_resources_in (int, optional): Split the GPU usage when using concurent muzero instances.

    Example:
        >>> muzero = MuZero("cartpole")
        >>> muzero.train()
        >>> muzero.test(render=True)
    """

    def __init__(
        self, game_name, override_config=None, split_resources_in=1, use_ray=True
    ):

        main_config = OmegaConf.load("destino.yaml")

        self.run_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.results_path = os.path.join(
            "./results", str(game_name), self.run_id
        )  # Path to store the model weights and TensorBoard logs

        # # Load the game and the config from the module with the game name
        try:
            game_module = importlib.import_module(f"games.{game_name}")
            self.Game = game_module.Game
            # self.config = game_module.MuZeroConfig()
            self.config = config.Config(game_name)
            self.config.game_filename = game_name
            self.config.save_model = main_config.save_model
            self.config.save_buffer = main_config.save_buffer
            self.config.cloud_save = main_config.cloud_save
            self.config.cloud_load = main_config.cloud_load  # does not work yet
            if self.config.cloud_load:
                self.config.cloud_run_id = main_config.cloud_run_id
            self.config.reanalyse_on_gpu = main_config.reanalyse_gpu_enabled
            self.config.results_path = self.results_path
        except ModuleNotFoundError as err:
            print(
                f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'
            )
            raise err

        # Overwrite the config
        if override_config:
            if type(override_config) is dict:
                for param, value in override_config.items():
                    setattr(self.config, param, value)
            else:
                self.config = override_config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Manage GPUs
        if self.config.max_num_gpus == 0 and (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            raise ValueError(
                "Inconsistent MuZeroConfig: max_num_gpus = 0 but GPU requested by selfplay_on_gpu or train_on_gpu or reanalyse_on_gpu."
            )
        if (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            total_gpus = (
                self.config.max_num_gpus
                if self.config.max_num_gpus is not None
                else torch.cuda.device_count()
            )
        else:
            total_gpus = 0
        self.num_gpus = total_gpus / split_resources_in
        if self.num_gpus > 1:
            self.num_gpus = math.floor(self.num_gpus)

        if use_ray:
            ray.init(num_gpus=total_gpus, ignore_reinit_error=True)

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "terminate": False,
        }
        self.replay_buffer = {}

        if use_ray:
            cpu_actor = CPUActor.remote()
            cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
            self.checkpoint["weights"], self.summary = copy.deepcopy(
                ray.get(cpu_weights)
            )
        else:
            cpu_actor = CPUActor
            cpu_weights = cpu_actor.get_initial_weights(self.config)
            self.checkpoint["weights"], self.summary = copy.deepcopy(cpu_weights)

        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def start_workers(self):

        # Configure the number of gpus
        self.gpu_config(split_resources_in=1)

        # ray._private.services.address_to_ip = lambda _node_ip_address: _node_ip_address
        ray.init(num_gpus=self.total_gpus, ignore_reinit_error=True)

        # Don't know why this was needed
        self.cpu_actoring()

        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        try:
            if self.shared_storage_worker:
                self.shared_storage_worker.set_info.remote("terminate", True)
                self.checkpoint = ray.get(
                    self.shared_storage_worker.get_checkpoint.remote()
                )
            if self.replay_buffer_worker:
                self.replay_buffer = ray.get(
                    self.replay_buffer_worker.get_buffer.remote()
                )

            print("\n")
            print("\nTraining Complete.")
            print("Shutting down workers...")

            self.self_play_workers = None
            self.test_worker = None
            self.training_worker = None
            self.reanalyse_worker = None
            self.replay_buffer_worker = None
            self.shared_storage_worker = None
        except AttributeError:
            print("Workers are already terminated")

    def shutdown(self):
        print("\nShuting Down...")
        ray.shutdown()

    def gpu_config(self, split_resources_in=1):
        # Manage GPUs
        if self.config.max_num_gpus == 0 and (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            raise ValueError(
                "Inconsistent MuZeroConfig: max_num_gpus = 0 but GPU requested by selfplay_on_gpu or train_on_gpu or reanalyse_on_gpu."
            )
        if (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            self.total_gpus = (
                self.config.max_num_gpus
                if self.config.max_num_gpus is not None
                else torch.cuda.device_count()
            )
        else:
            self.total_gpus = 0
        self.num_gpus = self.total_gpus / split_resources_in
        if self.num_gpus > 1:
            self.num_gpus = math.floor(self.num_gpus)

    def train(self, log_in_tensorboard=True, overwrite_results_path=None):
        """
        Spawn ray workers and launch the training.

        Args:
            log_in_tensorboard (bool): Start a testing worker and log its performance in TensorBoard.
        """
        if log_in_tensorboard or self.config.save_model:
            if overwrite_results_path:
                self.results_path = overwrite_results_path
                self.run_id = overwrite_results_path.split("/")[2]
            else:
                os.makedirs(self.results_path, exist_ok=True)

        # Manage GPUs
        if self.num_gpus > 0:
            num_gpus_per_worker = self.num_gpus / (
                self.config.train_on_gpu
                + self.config.num_workers * self.config.selfplay_on_gpu
                + log_in_tensorboard * self.config.selfplay_on_gpu
                + self.config.use_last_model_value * self.config.reanalyse_on_gpu
            )
            if num_gpus_per_worker > 1:
                num_gpus_per_worker = math.floor(num_gpus_per_worker)
        else:
            num_gpus_per_worker = 0

        # Initialize workers
        self.training_worker = trainer.Trainer.options(
            num_cpus=0,
            num_gpus=num_gpus_per_worker if self.config.train_on_gpu else 0,
        ).remote(self.checkpoint, self.config)

        self.shared_storage_worker = shared_storage.SharedStorage.remote(
            self.checkpoint, self.config, self.results_path
        )
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
            self.checkpoint, self.replay_buffer, self.config
        )

        if self.config.use_last_model_value:
            self.reanalyse_worker = replay_buffer.Reanalyse.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.reanalyse_on_gpu else 0,
            ).remote(self.checkpoint, self.config)

        self.self_play_workers = [
            play.SelfPlay.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            ).remote(self.checkpoint, self.Game, self.config, self.config.seed + seed)
            for seed in range(self.config.num_workers)
        ]

        # Launch workers
        [
            self_play_worker.continuous_self_play.remote(
                self.shared_storage_worker, self.replay_buffer_worker
            )
            for self_play_worker in self.self_play_workers
        ]
        self.training_worker.continuous_update_weights.remote(
            self.replay_buffer_worker, self.shared_storage_worker
        )
        if self.config.use_last_model_value:
            self.reanalyse_worker.reanalyse.remote(
                self.replay_buffer_worker, self.shared_storage_worker
            )

        if log_in_tensorboard:
            self.logging_loop(
                num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            )

    def logging_loop(self, num_gpus):
        """
        Keep track of the training performance.
        """
        # Launch the test worker to get performance metrics
        self.test_worker = play.SelfPlay.options(num_cpus=0, num_gpus=num_gpus,).remote(
            self.checkpoint,
            self.Game,
            self.config,
            self.config.seed + self.config.num_workers,
            self.config.selfplay_on_gpu,
        )
        self.test_worker.continuous_self_play.remote(
            self.shared_storage_worker, None, True
        )

        # Write everything in TensorBoard
        writer = SummaryWriter(
            self.results_path, purge_step=self.checkpoint["training_step"]
        )

        print(
            "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
        )

        # Save hyperparameters to TensorBoard
        hp_table = [
            f"| {key} | {value} |" for key, value in self.config.__dict__.items()
        ]
        writer.add_text(
            "Hyperparameters",
            "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
        )
        # Save model representation
        writer.add_text(
            "Model summary",
            self.summary,
        )
        # Loop for updating the training performance
        keys = [
            "total_reward",
            "muzero_reward",
            "opponent_reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "num_played_games",
            "num_played_steps",
            "num_reanalysed_games",
        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        with contextlib.suppress(KeyboardInterrupt):
            # Sample rate in training steps
            while info["training_step"] < self.config.training_steps:
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                if last_step == info["training_step"] or info["training_step"] % sample_rate != 0:
                    continue
                last_step = info["training_step"]
                if info["training_step"] % save_interval == 0:
                    self.gcloud_storage_upload()
                writer.add_scalar(
                    "1.Total_reward/1.Total_reward",
                    info["total_reward"],
                    info["training_step"],
                )
                writer.add_scalar(
                    "1.Total_reward/2.Mean_value",
                    info["mean_value"],
                    info["training_step"],
                )
                writer.add_scalar(
                    "1.Total_reward/3.Episode_length",
                    info["episode_length"],
                    info["training_step"],
                )
                writer.add_scalar(
                    "1.Total_reward/4.MuZero_reward",
                    info["muzero_reward"],
                    info["training_step"],
                )
                writer.add_scalar(
                    "1.Total_reward/5.Opponent_reward",
                    info["opponent_reward"],
                    info["training_step"],
                )
                writer.add_scalar(
                    "2.Loss/1.Total_weighted_loss", info["total_loss"], info["training_step"]
                )
                writer.add_scalar("2.Loss/Value_loss", info["value_loss"], info["training_step"])
                writer.add_scalar(
                    "2.Loss/Reward_loss", info["reward_loss"], info["training_step"]
                )
                writer.add_scalar(
                    "2.Loss/Policy_loss", info["policy_loss"], info["training_step"]
                )
                writer.add_scalar(
                    "3.Workers/1.Self_played_games",
                    info["num_played_games"],
                    info["training_step"],
                )
                writer.add_scalar(
                    "3.Workers/2.Training_steps", info["training_step"], info["training_step"]
                )
                writer.add_scalar(
                    "3.Workers/3.Self_played_steps",
                    info["num_played_steps"],
                    info["training_step"],
                )
                writer.add_scalar(
                    "3.Workers/4.Reanalysed_games",
                    info["num_reanalysed_games"],
                    info["training_step"],
                )
                writer.add_scalar(
                    "3.Workers/5.Training_steps_per_self_played_step_ratio",
                    info["training_step"] / max(1, info["num_played_steps"]),
                    info["training_step"],
                )
                writer.add_scalar("3.Workers/6.Learning_rate", info["lr"], info["training_step"])
                print(
                    f'Last test reward: {info["total_reward"]:.2f}. Training step: {info["training_step"]}/{self.config.training_steps}. Played games: {info["num_played_games"]}. Loss: {info["total_loss"]:.2f}',
                    end="\r",
                )
                # I don't know why this timer was here.
                # time.sleep(0.5)

    def clean_up(self):
        self.terminate_workers()

        # save the config file
        if self.config.save_model:
            config_path = f"games/config/{self.config.game_name}.yaml"
            dest_path = os.path.join(self.results_path, f"{self.config.game_name}.yaml")
            shutil.copy(config_path, dest_path)

        # save the buffer
        if self.config.save_buffer:
            # Persist replay buffer to disk
            print("\n\nPersisting replay buffer games to disk...")
            pickle.dump(
                {
                    "buffer": self.replay_buffer,
                    "num_played_games": self.checkpoint["num_played_games"],
                    "num_played_steps": self.checkpoint["num_played_steps"],
                    "num_reanalysed_games": self.checkpoint["num_reanalysed_games"],
                },
                open(os.path.join(self.results_path, "replay_buffer.pkl"), "wb"),
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def gcloud_storage_upload(self):
        if self.config.save_buffer:
            # Persist replay buffer to disk
            print("\n\nPersisting replay buffer games to disk...")
            pickle.dump(
                {
                    "buffer": self.replay_buffer,
                    "num_played_games": self.checkpoint["num_played_games"],
                    "num_played_steps": self.checkpoint["num_played_steps"],
                    "num_reanalysed_games": self.checkpoint["num_reanalysed_games"],
                },
                open(os.path.join(self.results_path, "replay_buffer.pkl"), "wb"),
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        if self.config.cloud_save:
            gcloud.upload_blob(self.results_path, self.config.game_name, self.run_id)

    async def api_play(
        self, render=True, opponent="api", game_state=None, onnx_model=False
    ):
        """
        Play against a human for the APi

        Args:
            render (bool): To display or not the environment. Defaults to True.

            opponent (str): "self" for self-play, "human" for playing against MuZero and "random"
            for a random agent, None will use the opponent in the config. Defaults to None.

            muzero_player (int): Player number of MuZero in case of multiplayer
            games, None let MuZero play all players turn by turn, None will use muzero_player in
            the config. Defaults to None.

            num_tests (int): Number of games to average. Defaults to 1.

            num_gpus (int): Number of GPUs to use, 0 forces to use the CPU. Defaults to 0.
        """
        play_worker = play.ApiPlay(
            self.checkpoint,
            self.Game,
            self.config,
            numpy.random.randint(10000),
            onnx_model=onnx_model,
        )
        return play_worker.play_game(render, opponent, game_state=game_state)

    def test(
        self, render=True, opponent=None, muzero_player=None, num_tests=1, num_gpus=0
    ):
        """
        Test the model in a dedicated thread.

        Args:
            render (bool): To display or not the environment. Defaults to True.

            opponent (str): "self" for self-play, "human" for playing against MuZero and "random"
            for a random agent, None will use the opponent in the config. Defaults to None.

            muzero_player (int): Player number of MuZero in case of multiplayer
            games, None let MuZero play all players turn by turn, None will use muzero_player in
            the config. Defaults to None.

            num_tests (int): Number of games to average. Defaults to 1.

            num_gpus (int): Number of GPUs to use, 0 forces to use the CPU. Defaults to 0.
        """
        opponent = opponent or self.config.opponent
        muzero_player = muzero_player or self.config.muzero_player
        self_play_worker = play.SelfPlay.options(
            num_cpus=0,
            num_gpus=num_gpus,
        ).remote(self.checkpoint, self.Game, self.config, numpy.random.randint(10000))
        results = []
        for i in range(num_tests):
            print(f"Testing {i+1}/{num_tests}")
            results.append(
                ray.get(
                    self_play_worker.play_game.remote(
                        0, 0, render, opponent, muzero_player
                    )
                )
            )
        self_play_worker.close_game.remote()

        if len(self.config.players) == 1:
            result = numpy.mean([sum(history.reward_history) for history in results])
        else:
            result = numpy.mean(
                [
                    sum(
                        reward
                        for i, reward in enumerate(history.reward_history)
                        if history.to_play_history[i - 1] == muzero_player
                    )
                    for history in results
                ]
            )
            print("result")
            print(result)
        return result

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        """
        Load a model and/or a saved replay buffer.

        Args:
            checkpoint_path (str): Path to model.checkpoint or model.weights.

            replay_buffer_path (str): Path to replay_buffer.pkl
        """
        # Load checkpoint
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                self.checkpoint = torch.load(checkpoint_path)
                print(f"\nUsing checkpoint from {checkpoint_path}")
            else:
                print(f"\nThere is no model saved in {checkpoint_path}.")

        # Load replay buffer
        if replay_buffer_path:
            if os.path.exists(replay_buffer_path):
                self._extracted_from_load_model_21(replay_buffer_path)
            else:
                print(
                    f"Warning: Replay buffer path '{replay_buffer_path}' doesn't exist.  Using empty buffer."
                )
                self.checkpoint["training_step"] = 0
                self.checkpoint["num_played_steps"] = 0
                self.checkpoint["num_played_games"] = 0
                self.checkpoint["num_reanalysed_games"] = 0

    # TODO Rename this here and in `load_model`
    def _extracted_from_load_model_21(self, replay_buffer_path):
        with open(replay_buffer_path, "rb") as f:
            replay_buffer_infos = pickle.load(f)
        self.replay_buffer = replay_buffer_infos["buffer"]
        self.checkpoint["num_played_steps"] = replay_buffer_infos[
            "num_played_steps"
        ]
        self.checkpoint["num_played_games"] = replay_buffer_infos[
            "num_played_games"
        ]
        self.checkpoint["num_reanalysed_games"] = replay_buffer_infos[
            "num_reanalysed_games"
        ]

        print(f"\nInitializing replay buffer with {replay_buffer_path}")

    def diagnose_model(self, horizon):
        """
        Play a game only with the learned model then play the same trajectory in the real
        environment and display information.

        Args:
            horizon (int): Number of timesteps for which we collect information.
        """
        game = self.Game(self.config.seed)
        obs = game.reset()
        dm = diagnose_model.DiagnoseModel(self.checkpoint, self.config)
        dm.compare_virtual_with_real_trajectories(obs, game, horizon)
        input("Press enter to close all plots")
        dm.close_all()

    def cpu_actoring(self):
        cpu_actor = CPUActor.remote()
        cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
        self.checkpoint["weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))


@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = models.MuZeroNetwork(config)
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary
