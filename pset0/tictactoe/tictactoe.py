"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.

    """
    # check if empty board, return X


    # check if full board, return X


    # check if number of X is more than 0, return 0, else return X

        
    raise NotImplementedError


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # loop through board to check if empty
    # remember empty spot and return the list

    raise NotImplementedError


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    # look into the board directly and change the board for the given player

    raise NotImplementedError


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    # check if any player fit winning siutation: 
    # (1,1 1,2 1,3) or (2,1 2,2 2,3) or (3,1 3,2 3,3) or (1,1 2,1 3,1) or (1,2 2,2 3,3) or (1,3 2,3 3,3) or (1,1 2,2 3,3) or (1,3 2,2 3,1)



    raise NotImplementedError


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    # check if winner return true

    # else loop through board to check if no empty slot

    raise NotImplementedError


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    # use the same logic as winner

    raise NotImplementedError


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # user player(board) to check turn
    # if terminal true
    #   return utility
    # moves = available moves for game (actions)
    # if X turn:
    #   value = -infinity
    #   for move in moves:
    #       value = max(value,minimax(move))
    # else:
    #   value = infinty
    #   for move in moves:
    #       value = min(value,minimax(move))

    raise NotImplementedError
