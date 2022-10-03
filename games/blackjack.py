import datetime
import os
import gym
from zarena import gym_blackjack

import numpy
import torch

from .abstract_game import AbstractGame


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = gym.make("BlackjackR-v1")

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done, info = self.env.step(action)
        print(reward)
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
        players_hand, players_value, _, _, _, _ = self.env.get_state()
        print("Player Hand: ", players_hand[1], " Player value: ", players_value[1])
        print("Dealer Hand: ", players_hand[0], " Dealer value: ", players_value[0])

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        if len(self.legal_actions()) > 4:
            choice = input(
                f"Enter the bet (4) $1, (5) $5, (6) $10, (7) $25, (8) $50, (9) $100, (10) $500 or (11) $1000 for the player {self.env.to_play()}: "
            )
            while choice not in [str(action) for action in self.env.legal_actions()]:
                choice = input(
                    "Enter either (4) $1, (5) $5, (6) $10, (7) $25, (8) $50, (9) $100, (10) $500 or (11) $1000 : "
                )
        if len(self.legal_actions()) == 3:
            choice = input(
                f"Enter the action (0) Stand, (1) Hit or (2) Double down for the player {self.env.to_play()}: "
            )
            while choice not in [str(action) for action in self.env.legal_actions()]:
                choice = input("Enter either (0) Stand, (1) Hit or (2) Double down : ")
        if len(self.legal_actions()) == 2:
            choice = input(
                f"Enter the action (0) Stand, or (1) Hit for the player {self.env.to_play()}: "
            )
            while choice not in [str(action) for action in self.env.legal_actions()]:
                choice = input("Enter either (0) Stand or (1) Hit : ")
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
            11: "$1000",
            10: "$500",
            9: "$100",
            8: "$50",
            7: "$25",
            6: "$10",
            5: "$5",
            4: "$1",
            3: "pull apart",
            2: "Double down",
            1: "Hit",
            0: "Stand",
        }
        return f"{action_number}. {actions[action_number]}"


# class TwentyOne:
#     def __init__(self, seed):
#         # No need to initalize hands as the game resets automatically
#         self.random = numpy.random.RandomState(seed)

#         self.player = 1

#     def to_play(self):
#         return 1 if self.player == 1 else 0

#     def reset(self):

#         # Could make deck a class variable if we wanted to see if MuZero can card count
#         self.deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] * 4
#         self.player_hand = self.deal()
#         self.dealer_hand = self.deal()
#         self.player_value = self.deal_card_value(self.player_hand)
#         self.dealer_value = self.deal_card_value(self.dealer_hand)

#         self.player = 1
#         return self.get_observation()

#     """
#     Action: 0 = Stand
#     Action: 1 = Hit
#     """

#     def step(self, action):

#         if action == 1:
#             self.player_value = self.deal_card_value(self.hit(self.player_hand))

#         done = self.is_busted() or action == 0 or self.player_value == 21

#         if done:
#             self.dealer_plays()

#         return self.get_observation(), self.get_reward(done), done

#     def get_observation(self):
#         return [
#             numpy.full((3, 3), self.player_value, dtype="float32"),
#             numpy.full((3, 3), self.dealer_value, dtype="float32"),
#             numpy.full((3, 3), 0),
#         ]

#     def legal_actions(self):
#         # 0 = hit
#         # 1 = stand
#         return [0, 1]

#     def get_reward(self, done):
#         if not done:
#             return 0
#         if self.player_value <= 21 and self.dealer_value < self.player_value:
#             return 1
#         if self.player_value <= 21 and self.dealer_value > 21:
#             return 1
#         if self.player_value > 21:
#             return -1
#         if self.player_value == self.dealer_value:
#             return 0
#         return -1

#     def deal(self):

#         hand = []
#         for i in range(2):
#             # random.shuffle(deck)
#             # card = deck.pop()
#             # TODO - Remove card drawn from the deck
#             card = self.deck[self.random.randint(0, len(self.deck))]
#             if card == 11:
#                 card = "J"
#             if card == 12:
#                 card = "Q"
#             if card == 13:
#                 card = "K"
#             if card == 14:
#                 card = "A"
#             hand.append(card)

#         return hand

#     def hit(self, hand):
#         card = self.deck[self.random.randint(0, len(self.deck))]
#         hand.append(card)
#         return hand

#     def deal_card_value(self, hand):

#         value = 0
#         for card in hand:
#             if card == "J" or card == "Q" or card == "K":
#                 value += 10
#             elif card == "A":
#                 if value >= 11:
#                     value += 1
#                 else:
#                     value += 11
#             else:
#                 value += card

#         return value

#     def dealer_plays(self):
#         if self.player_value > 21:
#             return
#         while self.dealer_value <= 16:
#             self.dealer_value = self.deal_card_value(self.hit(self.dealer_hand))

#     def is_busted(self):
#         if self.player_value > 21:
#             return True

#     def render(self):
#         print(
#             "Player Hand: "
#             + str(self.player_hand)
#             + " Player Value: "
#             + str(self.player_value)
#         )
#         print(
#             "Dealer Hand: "
#             + str(self.dealer_hand)
#             + " Dealer Value: "
#             + str(self.dealer_value)
#         )
