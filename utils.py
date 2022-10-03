import muzero
import os
import time
import gcloud
from glob import glob


class Utils:
    def load_model_menu(self, game_name):
        # Configure running options
        options = ["Specify paths manually"] + sorted(glob(f"results/{game_name}/*/"))
        options.reverse()
        print()
        for i in range(len(options)):
            print(f"{i}. {options[i]}")

        choice = input("Enter a number to choose a model to load: ")
        valid_inputs = [str(i) for i in range(len(options))]
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")
        choice = int(choice)

        if choice == (len(options) - 1):
            # manual path option
            checkpoint_path = input(
                "Enter a path to the model.checkpoint, or ENTER if none: "
            )
            while checkpoint_path and not os.path.isfile(checkpoint_path):
                checkpoint_path = input("Invalid checkpoint path. Try again: ")
            replay_buffer_path = input(
                "Enter a path to the replay_buffer.pkl, or ENTER if none: "
            )
            while replay_buffer_path and not os.path.isfile(replay_buffer_path):
                replay_buffer_path = input("Invalid replay buffer path. Try again: ")
        else:
            checkpoint_path = f"{options[choice]}model.checkpoint"
            replay_buffer_path = f"{options[choice]}replay_buffer.pkl"

        self.load_model(checkpoint_path=checkpoint_path, replay_buffer_path=replay_buffer_path)

    def cloud_load_model_menu(muzero, game_name):
        # Configure running options
        options = Utils.get_run_ids(gcloud.get_blobs_list(game_name))
        if options == 0:
            print("No cloud models found")
            time.sleep(3)
            return None
        options.reverse()
        print()
        for i in range(len(options)):
            print(f"{i}. {options[i]}")

        choice = input("Enter a number to choose a model to load: ")
        valid_inputs = [str(i) for i in range(len(options))]
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")
        choice = int(choice)

        destination_folder_name = gcloud.download_blob(game_name, options[choice])

        checkpoint_path = os.path.join(destination_folder_name, "model.checkpoint")
        replay_buffer_path = os.path.join(destination_folder_name, "replay_buffer.pkl")

        muzero.load_model(
            checkpoint_path=checkpoint_path, replay_buffer_path=replay_buffer_path
        )
        return destination_folder_name

    def get_run_ids(blobs):
        ids = []
        for blob in blobs:
            id = blob.name.split("/")[2]
            if not id in ids:
                ids.append(id)
        return ids
