import os
import sys
import datetime
import numpy
import toml
import gym
from zarena import gym_chess

from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from six import StringIO
from pprint import pprint
from .abstract_game import AbstractGame

# The Abstract Class Game Wrapper
class Game(AbstractGame):
    def __init__(self, seed=None):
        self.env = gym_chess.ChessEnv(log=False)
        # self.env = gym.make("ChessEnv")

        if seed is not None:
            self.env.seed(seed)

    # Apply action ot the game.
    # Args: action : Action of the action_space to take
    # Returns: The new observation, the reward and a boolean if the game has ended.
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return numpy.array(observation), reward, done

    # Returns the current player
    # it should be an element of the players list in the config.
    def to_play(self):
        return self.env.to_play()

    # Should return the legal actions at each turn, if it is not available, it can return
    # the whole action space. At each turn, the game have to be able to handle one of returned actions.
    # For complex game where calculating legal moves is too long, the idea is to define the legal actions
    # equal to the action space but to return a negative reward if the action is illegal.
    # Returns: An array of integers, subset of the action space.
    def legal_actions(self):
        return self.env.legal_actions()

    # Reset the game for a new game. Inital observation of the game
    def reset(self):
        return numpy.array(self.env.reset())

    # Properly close the game
    def close(self):
        self.env.close()

    # Display the game ovservation
    def render(self):
        self.env.render()
        # input("Press enter to take a step ")

    # Ask the user for a legal action
    # and return the corresponding action number.
    # Returns an integer from the action space
    def human_to_action(self):
        for action in self.legal_actions():
            print(str(action))
        choice = input(f"Enter the action to play for the player {self.to_play()}: ")
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter the action : ")

        return int(choice)

    # Convert an action number to a string representing the action.
    # Arugs: action_number: an integer from the action space.
    # Returns: String representing the action
    def action_to_string(self, action_number):
        # Convert action_number to coordinates
        # Call Env3 function to convert
        move = self.env.action_to_move(action_number)
        action_string = self.env.move_to_str_code(move)

        return action_string


def highlight(string, background="white", color="gray"):
    return utils.colorize(utils.colorize(string, color), background, highlight=True)
