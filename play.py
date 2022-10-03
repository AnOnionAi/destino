import math
import time

import numpy
import ray
import torch

import models
import models_onnx
import onnxruntime
from history import GameHistory
from mcts import MCTS


class Play:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, initial_checkpoint, Game, config, seed, onnx_model=False):
        self.config = config
        self.game = Game(seed)
        self.onnx_model = onnx_model
        self.onnx_device = None

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        if onnx_model:
            self.model = models_onnx.MuZeroNetwork(self.config)
            if (
                "CUDAExecutionProvider" in self.model.rep_net_session.get_providers()
                and "CUDAExecutionProvider" in onnxruntime.get_available_providers()
            ):
                self.onnx_device = "cuda"
            else:
                self.onnx_device = "cpu"
        else:
            self.model = models.MuZeroNetwork(self.config)
            self.model.set_weights(initial_checkpoint["weights"])
            self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.model.eval()

    def play_game(
        self, temperature, temperature_threshold, render, opponent, muzero_player
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        game_history = GameHistory()
        observation = self.game.reset()
        self._extracted_from_play_game_74(game_history, 0, observation, 0)
        done = False

        if render:
            self.game.render()

        with torch.no_grad():
            while (
                not done and len(game_history.action_history) <= self.config.max_moves
            ):
                assert (
                    len(numpy.array(observation).shape) == 3
                ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
                assert (
                    numpy.array(observation).shape == self.config.observation_shape
                ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."
                stacked_observations = game_history.get_stacked_observations(
                    -1, self.config.stacked_observations, len(self.config.action_space)
                )

                # Choose the action
                if opponent == "self" or muzero_player == self.game.to_play():
                    mcts = MCTS(self.config.__dict__)
                    root, mcts_info = mcts.run(
                        self.model,
                        stacked_observations,
                        self.game.legal_actions(),
                        self.game.to_play(),
                        True,
                    )
                    action = self.select_action(
                        mcts,
                        root,
                        temperature
                        if not temperature_threshold
                        or len(game_history.action_history) < temperature_threshold
                        else 0,
                    )

                    if render:
                        print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                        print(
                            f"Root value for player {self.game.to_play()}: {root.value():.2f}"
                        )
                else:
                    action, root = self.select_opponent_action(
                        opponent, stacked_observations
                    )

                observation, reward, done = self.game.step(action)

                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()

                game_history.store_search_statistics(
                    mcts, root, self.config.action_space
                )

                self._extracted_from_play_game_74(game_history, action, observation, reward)
        return game_history

    # TODO Rename this here and in `play_game`
    def _extracted_from_play_game_74(self, game_history, arg1, observation, arg3):
                # Next batch
        game_history.action_history.append(arg1)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(arg3)
        game_history.to_play_history.append(self.game.to_play())

    def close_game(self):
        self.game.close()

    def get_action(self, stacked_observations):
        mcts = MCTS(self.config.__dict__)
        root, mcts_info = mcts.run(
            self.model,
            stacked_observations,
            self.game.legal_actions(),
            self.game.to_play(),
            True,
            onnx_model=self.onnx_model,
            onnx_device=self.onnx_device,
        )
        muzero_action = self.game.action_to_string(self.select_action(mcts, root, 0))
        print(f'Tree depth: {mcts_info["max_tree_depth"]}')
        print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
        print(f"Player {self.game.to_play()} turn. MuZero suggests {muzero_action}")
        return muzero_action, root

    @staticmethod
    def select_action(mcts, node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = numpy.array(mcts.visit_counts(node.idx), dtype="int32")

        actions = mcts.actions(node.idx)
        if temperature == 0:
            return actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            return numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            return numpy.random.choice(actions, p=visit_count_distribution)


@ray.remote
class SelfPlay(Play):
    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            if not test_mode:
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=ray.get(
                            shared_storage.get_info.remote("training_step")
                        )
                    ),
                    self.config.temperature_threshold,
                    False,
                    "self",
                    0,
                )

                replay_buffer.save_game.remote(game_history, shared_storage)

            else:
                # Take the best action (no exploration) in test mode
                game_history = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                    "self" if len(self.config.players) == 1 else self.config.opponent,
                    self.config.muzero_player,
                )

                # Save to the shared storage
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": numpy.mean(
                            [value for value in game_history.root_values if value]
                        ),
                    }
                )
                if len(self.config.players) > 1:

                    shared_storage.set_info.remote(
                        {
                            "muzero_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                == self.config.muzero_player
                            ),
                            "opponent_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                != self.config.muzero_player
                            ),
                        }
                    )
            # Managing the self-play / training ratio
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            if not test_mode and self.config.ratio:
                while (
                    ray.get(shared_storage.get_info.remote("training_step"))
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

        self.close_game()

    def select_opponent_action(self, opponent, stacked_observations):
        """
        Select opponent action for evaluating MuZero level.
        """
        if opponent == "human":
            _, root = self.get_action(stacked_observations)
            return self.game.human_to_action(), root
        elif opponent == "expert":
            return self.game.expert_agent(), None
        elif opponent == "random":
            assert (
                self.game.legal_actions()
            ), f"Legal actions should not be an empty array. Got {self.game.legal_actions()}."
            assert set(self.game.legal_actions()).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            return numpy.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )


class ApiPlay(Play):
    def play_game(self, render, opponent, game_state):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        game_history = GameHistory()
        observation = self.game.set_state(game_state)  # Initialize state
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())

        if render:
            self.game.render()

        with torch.no_grad():
            stacked_observations = game_history.get_stacked_observations(
                -1, self.config.stacked_observations, len(self.config.action_space)
            )
            # Choose the action
            if opponent == "api":
                return self.select_opponent_action(opponent, stacked_observations)

    def select_opponent_action(self, opponent, stacked_observations):
        """
        Select opponent action for evaluating MuZero level.
        """
        if opponent == "api":
            muzero_action, _ = self.get_action(stacked_observations)
            return muzero_action
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "api"'
            )
