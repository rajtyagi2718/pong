#!/usr/local/bin/python3

from player import User, AI, Random, Model
from ql4 import ql
from table import Table
from game import Game
from ui import UI

def play(left_player, right_player):
    table = Table()
    game = Game(table, left_player, right_player)
    ui = UI(game)
    ui.start()

def play_ai():
    table = Table()
    left_player = AI('left_pad')
    right_player = Model('right_pad')
    game = Game(table, left_player, right_player, max_score=100)
    game.play()
    print('winner:', game.left_score, game.right_score)

def play_ql_model():
    table = Table()
    left_player = ql
    right_player = Model('right_pad')
    game = Game(table, left_player, right_player)
    ui = UI(game)
    ui.start()
    print('winner:', game.winner)

import time

if __name__ == '__main__':
    # play_ai()
    # play(User(), User())
    # play(User(), Model())
    # play(AI(), Model())
    # play(Model(), Model())
    play_ql_model()
