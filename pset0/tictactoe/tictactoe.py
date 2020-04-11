"""
Tic Tac Toe Player
"""
import copy
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
    emp = True
    numx = 0
    num0 = 0
    
    # loop through board to count number of X and O
    for cell in board:
        if cell is not EMPTY:
            if cell is X:
                numx += 1
            else:
                num0 += 1
            emp = False
    
    # check if empty board, return X
    if emp is True:
        return X

    # check if number of X is more than 0, return 0, else return X
    elif numx > num0:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = []
    # loop through board to check if empty
    for i in range(3):
        for j in range(3):
            if board[i][j] is EMPTY:
                # remember empty spot and return the list
                actions.append((i,j))
    
    return actions



def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    move = player(board)
    result = copy.deepcopy(board)
    # check for invalid action
    if result[action(1)][action(2)] is EMPTY:
        raise NameError('Invalid Action')
    # look into the board directly and change the board for the given player
    else:
        result[action(1)][action(2)] = move
    
    return result


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
