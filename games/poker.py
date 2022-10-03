import datetime
import os
import gym
from zarena import gym_poker

import torch

from .abstract_game import AbstractGame


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = gym.make("PokerR-v1")

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
        observation = self.env.reset()
        return observation

    def render(self):
        """
        Display the game observation.
        """
        (
            community_cards,
            players,
            pots,
            total_players,
            n_players_in_hand,
            current_player,
            button,
            poker_phase,
            turn_in_phase,
            bet_phase,
        ) = self.env.get_state()
        print("Pots: ", pots)
        print("Community cards: ", community_cards)
        print("Poker phase: ", poker_phase)
        print("Bet: ", bet_phase)
        for player in players:
            s_button = ""
            if player["id"] == button:
                s_button = "⨀"
            print("Credits: ", player["credits"])
            print(
                "Player:",
                player["id"],
                " Hand:",
                player["hand"],
                " Bet:",
                player["bet"],
                " ",
                s_button,
            )

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        s = ""
        legal_actions = self.legal_actions()
        for i in range(0, len(legal_actions)):
            s += (
                "("
                + str(legal_actions[i])
                + ") "
                + self.action_to_string(legal_actions[i])[3:]
                + ", "
            )
        choice = input(f"Enter the option {s} for the player {self.env.to_play()}: ")
        while choice not in [str(action) for action in self.env.legal_actions()]:
            choice = input(f"Enter either {s} : ")
        return int(choice)

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            11: "all in",
            10: "raise to $1000",
            9: "raise to $500",
            8: "raise to $100",
            7: "raise to $50",
            6: "raise to $25",
            5: "call",
            4: "bet",
            3: "check",
            2: "fold",
            1: "big blind",
            0: "small blind",
        }
        return f"{action_number}. {actions[action_number]}"
