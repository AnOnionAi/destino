# import torch
# import models
# import numpy
# import config
# from muzero import MuZero
# from mcts import MCTS, NodeRs, MinMaxStatsRs

# # Initial trees
# config = config.Config("tictactoe_static")
# MuZero("tictactoe")
# mcts = MCTS(config.__dict__)
# model = models.MuZeroNetwork(config)

# # Initial Nodes
# observation = [[[1] * 3, [0] * 3, [0] * 3], [[0] * 3, [1] * 3, [0] * 3]]
# legal_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# to_play = 0
# add_exploration_noise = False
# test_mode = True

# rs_result = mcts.run(
#     model, observation, legal_actions, to_play, add_exploration_noise, test_mode
# )

# # Test run
# def test_info():
#     assert rs_result[1] == {
#         'max_tree_depth': 3.0,
#         'root_predicted_value': 0.10186457633972168
#         }

# def test_visit_count():
#     assert rs_result[0].visit_count == 25

# def test_to_play():
#     assert rs_result[0].to_play == 0

# def test_prior():
#     assert rs_result[0].prior == 0.0

# def test_value_sum():
#     assert rs_result[0].value_sum == -5.22237753868103

# def test_sum_child_visit_count():
#     assert mcts.sum_child_visit_count(
#         rs_result[0].idx
#     ) == [0.04, 0.24, 0.2, 0.24, 0.04, 0.04, 0.04, 0.12, 0.04]

# # def test_hidden_state():
# #     tensorA = mcts.get_hidden_state(rs_result[0].idx)
# #     tensorB = torch.tensor(
# #         [[[[1.0000, 0.5799, 0.0000],[0.0000, 0.7213, 0.0000],[0.0000, 0.5132, 0.0000]],
# #         [[0.2753, 0.0000, 0.0000],[1.0000, 0.1921, 0.2898],[0.0000, 0.0806, 0.0000]],
# #         [[0.0000, 0.0000, 1.0000],[0.1761, 0.0000, 0.0000],[0.7987, 0.4391, 0.0000]],
# #         [[0.0000, 0.0000, 0.1975],[1.0000, 0.1933, 0.2786],[0.0000, 0.3331, 0.0121]],
# #         [[0.0000, 0.0000, 0.0938],[0.1618, 1.0000, 0.1721],[0.1288, 0.0000, 0.3218]],
# #         [[0.5081, 0.0000, 0.0000],[1.0000, 0.9183, 0.0000],[0.5201, 0.5334, 0.0000]],
# #         [[0.8127, 0.5095, 0.0000],[0.0585, 0.0000, 0.3237],[1.0000, 0.0632, 0.0294]],
# #         [[1.0000, 0.4336, 0.2066],[0.3346, 0.1930, 0.0000],[0.0000, 0.0000, 0.1517]],
# #         [[0.0138, 0.6336, 0.0000],[0.9337, 0.3751, 0.0000],[0.0000, 1.0000, 0.4909]],
# #         [[0.0000, 0.0000, 1.0000],[0.0000, 0.0000, 0.0000],[0.0524, 0.9340, 0.6105]],
# #         [[0.1509, 0.0000, 0.0761],[0.0000, 0.2961, 0.0754],[1.0000, 0.0959, 0.8209]],
# #         [[0.1210, 0.9702, 0.5901],[0.0000, 0.0000, 0.0000],[0.0975, 1.0000, 0.4971]],
# #         [[1.0000, 0.4599, 0.0604],[0.1603, 0.2163, 0.0000],[0.2773, 0.0000, 0.0000]],
# #         [[0.1451, 0.5066, 0.0000],[0.0000, 1.0000, 0.0421],[0.0832, 0.0000, 0.2748]],
# #         [[0.0733, 0.5008, 0.0000],[0.7320, 0.0000, 0.0000],[1.0000, 0.4644, 0.8557]],
# #         [[0.0000, 0.0000, 0.7866],[0.0000, 0.5826, 0.3525],[0.0896, 0.4507, 1.0000]]]]
# #     )
# #     assert torch.equal(tensorA, tensorB)

# def test_hidden_reward():
#     assert rs_result[0].reward == -0.0

# def test_value():
#     assert rs_result[0].value() == -0.20889510154724122

# # MinMaxStats
# def test_min_max_stats():
#     min_max_rust = MinMaxStatsRs()
#     min_max_rust.update(-100)
#     min_max_rust.update(100)

#     assert(min_max_rust.normalize(5.5) == 0.5275)
#     assert(min_max_rust.normalize(0) == 0.5)
#     assert(min_max_rust.normalize(-5.5) == 0.4725)

# # Test select_action
# visit_counts_rust = mcts.visit_counts(0)

# def test_visit_count_order():
#     assert visit_counts_rust == [1, 6, 5, 6, 1, 1, 1, 3, 1]

# actions_rust = mcts.actions(rs_result[0].idx)

# def test_actions_order():
#     assert actions_rust == [0, 1, 2, 3, 4, 5, 6, 7, 8]

# def test_select_action():
#     action_rust = actions_rust[numpy.argmax(visit_counts_rust)]
#     assert action_rust == 1

# if __name__ == "__main__":
#     run_test_funcs(__name__)
