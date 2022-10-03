from abc import ABC, abstractmethod

# Inherit this class for muzero to play
class AbstractGame(ABC):
    @abstractmethod
    def __init__(self, seed=None):
        pass

    # Apply action ot the game.
    # Args: action : Action of the action_space to take
    @abstractmethod
    def step(self, action):
        pass

    # Returns the current player
    # it should be an element of the players list in the config.
    def to_play(self):
        return 0

    # Should return the legal actions at each turn, if it is not available, it can return
    # the whole action space. At each turn, the game have to be able to handle one of returned actions.
    # For complex game where calculating legal moves is too long, the idea is to define the legal actions
    # equal to the action space but to return a negative reward if the action is illegal.
    # Returns: An array of integers, subset of the action space.
    @abstractmethod
    def legal_actions(self):
        pass

    # Reset the game for a new game. Inital observation of the game
    @abstractmethod
    def reset(self):
        pass

    # Properly close the game
    def close(self):
        pass

    # Display the game ovservation
    @abstractmethod
    def render(self):
        pass

    # Ask the user for a legal action
    # and return the corresponding action number.
    # Returns an integer from the action space
    def human_to_action(self):

        choice = input(f"Enter the action to play for the player {self.to_play()}: ")
        while int(choice) not in self.legal_actions():
            choice = input("Ilegal action. Enter another action : ")
        return int(choice)

    # Hard coded agent that MuZero faces to assess his progress in multiplayer games.
    # It doesn't influence training
    # Returns: Action as an integer to take in the current game state
    def expert_agent(self):

        raise NotImplementedError

    # Convert an action number to a string representing the action.
    # Arugs: action_number: an integer from the action space.
    # Returns: String representing the action
    def action_to_string(self, action_number):

        return str(action_number)
