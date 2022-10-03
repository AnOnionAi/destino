import copy
import os

import ray
import torch


@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self, checkpoint, config, results_path):
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)
        self.results_path = results_path

    def save_checkpoint(self, path=None):
        if not path:
            path = os.path.join(self.results_path, "model.checkpoint")

        torch.save(self.current_checkpoint, path)

        # print(
        #     "Total Reward: ",
        #     self.current_checkpoint["total_reward"],
        #     "Muzero Reward: ",
        #     self.current_checkpoint["muzero_reward"],
        #     "Opponent Reward: ",
        #     self.current_checkpoint["opponent_reward"],
        #     " ",
        # )
        # print("\n")
        # print(
        #     "Training Step: ",
        #     self.current_checkpoint["training_step"],
        #     "Total Loss: ",
        #     self.current_checkpoint["total_loss"],
        #     "Played Games: ",
        #     self.current_checkpoint["num_played_games"],
        # )

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError
