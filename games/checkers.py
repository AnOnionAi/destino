import gym
from zarena import gym_checkers

from .abstract_game import AbstractGame


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = gym.make("CheckersEnv-v1")

    def step(self, action):
        """
        Apply action to the game.
        Args:
            action : action of the action_space to take.
        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.
        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.
        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.print()

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        for action in self.legal_actions():
            print(f"{str(action)}:{self.action_to_string(action)}")
        choice = input(f"Enter the action to play for the player {self.to_play()}: ")
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter the action : ")

        return int(choice)

    def set_state(self, game_state):
        """
        Args:
            game_state: the state to be established in the game
        Returns:
            observation of the game.
        """
        return self.env.set_state(game_state)

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.
        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        positions = action_to_positions(action_number)
        (x0, y0) = positions[0]
        (x1, y1) = positions[1]
        return f"({x0 + 1},{letters[y0]}) -> ({x1 + 1},{letters[y1]})"

def action_to_positions(action):
    positions = []
    _from = action // 32
    _to = action % 32
    x0 = _from // 4
    y0 = _from % 4
    positions.append((x0, y0 * 2 + (0 if x0 % 2 == 0 else 1)))
    x1 = _to // 4
    y1 = _to % 4
    positions.append((x1, y1 * 2 + (0 if x1 % 2 == 0 else 1)))
    return positions
