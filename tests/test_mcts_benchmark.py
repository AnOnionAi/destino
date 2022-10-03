# import time
# import models
# import ray
# import numpy as np
# from muzero import MuZero

# # Initial trees
# game = "tictactoe"  # chess or tictactoe
# config = MuZero(game).config
# model = models.MuZeroNetwork(config)

# # Initial Nodes
# if game == "tictactoe":
#     observation = np.array(
#         [[[0] * 3, [0] * 3, [0] * 3], [[0] * 3, [0] * 3, [0] * 3]], dtype=np.int8
#     )
#     legal_actions = list(range(9))
# elif game == "chess":
#     observation = np.array([[[0] * 8] * 8] * 2, dtype=np.int8)
#     legal_actions = list(range(4101))

# to_play = 0
# add_exploration_noise = True
# # test_mode = True


# def test_benchmark():
#     num_episodes = 20
#     n = num_episodes // 4

#     start_python = time.time()
#     ray.get([run_py_mcts.remote(n)] * 4)
#     end_python = time.time()

#     start_rust = time.time()
#     ray.get([run_rs_mcts.remote(n)] * 4)
#     end_rust = time.time()

#     diff_python = end_python - start_python
#     diff_rust = end_rust - start_rust

#     print("Total episodes", num_episodes)
#     print("Total time Python", diff_python, "s.")
#     print("Total time Rust", diff_rust, "s.")

#     # Rust must be faster than Python
#     assert diff_python > diff_rust
#     # assert diff_python / 4 < diff_rust


# # Tictactoe
# # Total episodes 10
# # Total time Python 0.7940387725830078 s.
# # Total time Rust 0.6324665546417236 s.

# # Chess
# # Total episodes 10
# # Total time Python 61.47204327583313 s.
# # Total time Rust 14.950337648391724 s.


# @ray.remote
# def run_py_mcts(num_episodes):
#     from mcts import MCTS

#     for _ in range(num_episodes):
#         MCTS(config).run(
#             model,
#             observation,
#             legal_actions,
#             to_play,
#             add_exploration_noise,
#             # test_mode
#         )


# @ray.remote
# def run_rs_mcts(num_episodes):
#     from mcts import MCTS

#     for _ in range(num_episodes):
#         MCTS(config.__dict__).run(
#             model,
#             observation,
#             legal_actions,
#             to_play,
#             add_exploration_noise,
#             # test_mode
#         )


# if __name__ == "__main__":
#     run_test_funcs(__name__)
