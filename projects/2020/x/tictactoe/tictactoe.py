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
    for rolls in board:
        for cell in rolls:
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
                actions.append([i,j])
    
    return actions



def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action is None:
        return board
    
    move = player(board)
    result = copy.deepcopy(board)
    # check for invalid action
    if result[action[0]][action[1]] is not EMPTY:
        raise NameError('Invalid Action')
    # look into the board directly and change the board for the given player
    else:
        result[action[0]][action[1]] = move
        return result
    
    


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    
    # winning siutation: 
    # (1,1 1,2 1,3) or (2,1 2,2 2,3) or (3,1 3,2 3,3) or (1,1 2,1 3,1) or (1,2 2,2 3,3) or (1,3 2,3 3,3) or (1,1 2,2 3,3) or (1,3 2,2 3,1)
    winning = [[[1,1],[1,2],[1,3]], [[2,1],[2,2],[2,3]], [[3,1],[3,2],[3,3]], [[1,1],[2,1],[3,1]], [[1,2],[2,2],[3,2]], [[1,3],[2,3],[3,3]], [[1,1],[2,2],[3,3]], [[1,3],[2,2],[3,1]]]

    for win in winning:
        # check whether there is a winning situation
        win_set = []
        for cell in win:
            
            win_set.append(board[cell[0]-1][cell[1]-1])
            
        check = all(x == win_set[0] for x in win_set)
        if check:
            # check which player fit the winning siutation
            if win_set[0] is X:
                return X
            elif win_set[0] is O:
                return O
    
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    full = True
    # check if winner return true
    if winner is None:
        return False
    elif winner(board) is X or winner(board) is O:
        return True
    # else loop through board to check if no empty slot
    else:
        for rolls in board:
            for cell in rolls:
                if cell is EMPTY:
                    full = False

    return full


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    # use the same logic as winner
    if winner(board) is X:
        return 1
    elif winner(board) is O:
        return -1
    else:
        return 0



def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    # user player(board) to check turn
    turn = player(board)

    # if terminal board, return None
    if terminal(board):
        return None
    
    # best starting position is the middle block
    if empty(board):
        return [1,1]
    
    moves = actions(board)
    best_move = None

    # find value for a given move
    if turn is X:
        best_value = -math.inf
        for move in moves:
            value = minimax_value(result(board,move), O)
            if value > best_value:
                best_value = value
                best_move = move
    elif turn is O:
        best_value = math.inf
        for move in moves:
            value = minimax_value(result(board,move), X)
            if value < best_value:
                best_value = value
                best_move = move
    
    return best_move
    

def minimax_value(board,turn):
    # finding score for given board using minimax recursively
    if terminal(board):
        return utility(board)
    
    moves = actions(board)

    if turn is X:
        value = -math.inf
        for move in moves:
            value = max(value, minimax_value(result(board,move), O))
    
    elif turn is O:
        value = math.inf
        for move in moves:
            value = min(value, minimax_value(result(board,move), X))
    
    return value

def empty(board):
    # check if the board is empty
    emp = True
    for rolls in board:
            for cell in rolls:
                if cell is not EMPTY:
                    emp = False
    
    return emp