import os
import sys
import time
import datetime
import muzero
import hparams
import utils

if __name__ == "__main__":

    print("\nWelcome to MuZero! Here's a list of games:")

    # Let user pick a game
    games = [filename[:-3] for filename in sorted(os.listdir(f"{os.path.dirname(os.path.realpath(__file__))}/games")) if filename.endswith(".py") and filename != "abstract_game.py"]

    for i in range(len(games)):
        print(f"{i}. {games[i]}")
    choice = input("Enter a number to choose the game: ")
    valid_inputs = [str(i) for i in range(len(games))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")

    # Initialize MuZero
    choice = int(choice)
    game_name = games[choice]
    mu = muzero.MuZero(game_name)
    # mu.start_workers()
    overwrite_result_path = None

    while True:
        # Configure running options
        options = [
            "Train",
            "Load pretrained model",
            "Load pretrained model from cloud",
            "Diagnose model",
            "Render some self play games",
            "Play against MuZero",
            "Test the game manually",
            "Hyperparameter search",
            "Exit",
        ]
        print()
        for i in range(len(options)):
            print(f"{i}. {options[i]}")

        choice = input("Enter a number to choose an action: ")
        valid_inputs = [str(i) for i in range(len(options))]
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")
        choice = int(choice)
        if choice == 0:
            start = time.time()
            if overwrite_result_path:
                mu.train(overwrite_results_path=overwrite_result_path)
            else:
                mu.train()
            end = time.time()
            diff = end - start
            mu.clean_up()
            train_steps = mu.checkpoint["training_step"]
            num_games = mu.checkpoint["num_played_games"]
            total_steps = mu.checkpoint["num_played_steps"]
            reanalyze_games = mu.checkpoint["num_reanalysed_games"]

            console = sys.stdout
            try:
                analysis = os.path.join(mu.results_path, "analysis.txt")
                sys.stdout = open(analysis, "wt")
                print("Total Training Time:", datetime.timedelta(seconds=diff))
                print("Total Training Steps:", train_steps)
                print("Total Games:", num_games)
                print("Total Moves:", total_steps)  # Total of all game moves
                print("Moves per Game:", total_steps / num_games)
                print("Games Reanalyzed:", reanalyze_games)
                print(
                    "Games per Reanalyzed:", reanalyze_games / num_games
                )  # each game can be reanalyzed any # of times

                print("\nTime per Training Step", datetime.timedelta(seconds=diff / train_steps))

                print("Time per Game:", datetime.timedelta(seconds=diff / num_games))
                print("Time per Move:", datetime.timedelta(seconds=diff / total_steps))
                print("Time per Reanalyzed:", datetime.timedelta(seconds=diff / reanalyze_games))

            finally:
                sys.stdout.close()
                sys.stdout = console

            mu.gcloud_storage_upload()

        elif choice == 1:
            utils.Utils.load_model_menu(mu, game_name)
        elif choice == 2:
            overwrite_result_path = utils.Utils.cloud_load_model_menu(mu, game_name)
        elif choice == 3:
            mu.diagnose_model(30)
        elif choice == 4:
            mu.test(render=True, opponent="self", muzero_player=None)
        elif choice == 5:
            mu.test(render=True, opponent="human", muzero_player=0)
        elif choice == 6:
            env = mu.Game()
            env.reset()
            env.render()

            done = False
            while not done:
                action = env.human_to_action()
                observation, reward, done = env.step(action)
                print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
                env.render()
        elif choice == 7:
            # Define here the parameters to tune
            # Parametrization documentation: https://facebookresearch.github.io/nevergrad/parametrization.html
            mu.terminate_workers()
            hp = hparams.Hyperparams().hyperparameter_search(game_name)
        else:
            break

    mu.shutdown()
