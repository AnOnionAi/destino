import torch
import models
import numpy
from muzero import MuZero
from mcts import MCTS, Node, MinMaxStats
from mcts_rs import MCTSRs, NodeRs, MinMaxStatsRs

# Initial trees
config = MuZero("tictactoe").config
mcts_rs = MCTSRs(config.__dict__)
mcts_py = MCTS(config)
model = models.MuZeroNetwork(config)

# Initial Nodes
observation = [[[1] * 3, [0] * 3, [0] * 3], [[0] * 3, [1] * 3, [0] * 3]]
legal_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
to_play = 0
add_exploration_noise = False
test_mode = True

py_result = mcts_py.run(
    model, observation, legal_actions, to_play, add_exploration_noise, test_mode
)

rs_result = mcts_rs.run(
    model, observation, legal_actions, to_play, add_exploration_noise, test_mode
)

# Test run
def test_info():
    assert py_result[1] == rs_result[1]


def test_visit_count():
    assert py_result[0].visit_count == rs_result[0].visit_count


def test_to_play():
    assert py_result[0].to_play == rs_result[0].to_play


def test_prior():
    assert py_result[0].prior == rs_result[0].prior


def test_value_sum():
    assert py_result[0].value_sum == rs_result[0].value_sum


def test_sum_child_visit_count():
    assert sum_child_visit_count(py_result[0]) == mcts_rs.sum_child_visit_count(
        rs_result[0].idx
    )


def test_hidden_state():
    tensorA = py_result[0].hidden_state
    tensorB = mcts_rs.get_hidden_state(rs_result[0].idx)

    assert torch.equal(tensorA, tensorB)


def test_hidden_reward():
    assert py_result[0].reward == rs_result[0].reward


def test_value():
    assert py_result[0].value() == rs_result[0].value()


def sum_child_visit_count(node):
    sum_visits = sum(child.visit_count for child in node.children.values())
    return [
        node.children[a].visit_count / sum_visits if a in node.children else 0
        for a in config.action_space
    ]


# MinMaxStats
def test_min_max_stats():
    min_max_python = MinMaxStats()
    min_max_rust = MinMaxStatsRs()

    min_max_python.update(-100)
    min_max_rust.update(-100)

    min_max_python.update(100)
    min_max_rust.update(100)

    assert min_max_python.normalize(5.5) == min_max_rust.normalize(5.5)
    assert min_max_python.normalize(0) == min_max_rust.normalize(0)
    assert min_max_python.normalize(-5.5) == min_max_rust.normalize(-5.5)


# Test select_action
visit_counts_python = [child.visit_count for child in py_result[0].children.values()]
visit_counts_rust = mcts_rs.visit_counts(0)


def test_visit_count_order():
    assert visit_counts_python == visit_counts_rust


actions_python = list(py_result[0].children.keys())
actions_rust = mcts_rs.actions(rs_result[0].idx)


def test_actions_order():
    assert actions_python == actions_rust


def test_select_action():
    action_python = actions_python[numpy.argmax(visit_counts_python)]
    action_rust = actions_rust[numpy.argmax(visit_counts_rust)]
    assert action_python == action_rust


if __name__ == "__main__":
    run_test_funcs(__name__)
