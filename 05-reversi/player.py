from myboard import MyBoard
from numpy import inf
import random

class MyPlayer():
    '''
    A basic Alpha-Beta pruning player with prioritization of corner grabs.
    '''

    def __init__(self, my_color,opponent_color, board_size=8):
        self.name = 'sanozan'
        self.my_color = my_color
        self.opponent_color = opponent_color
        self.board_size = board_size

    def move(self,board):
        '''
        The main method for choosing the next move for the AI player.
        :param board: the matrix representation of the current state of the board.
        :return: move: a tuple (x,y) , indicating a move.
        '''
        # Construct a board object with the given board.
        boardObj = MyBoard(self.board_size, board, self.my_color, self.opponent_color, -1)
        # Initialize the alpha and beta values
        result_util, move = self.move_alpha_beta(boardObj, -inf, +inf, True, 4, self.my_color, self.opponent_color)
        return move

    def move_alpha_beta(self, boardObj, alpha, beta, isPlayer1, depth, p1color, p2color):
        '''
        The main method for alpha-beta pruning strategy.

        :param boardObj: an object of type MyBoard, containing info & useful functions on the board
        :param alpha: the positive cut-off evaluation value.
        :param beta: the negative cut-off evaluation value.
        :param isPlayer1: boolean indicating whether we are the maximizing (1st) player.
        :param depth: the maximum depth of the state search tree. Hard-coded to 4 for BRUTE submission.
        :param p1color: the color value for player 1, needed for certain functions
        :param p2color: the color value for player 2, needed for certain functions
        :return: a tuple (E, M),  for the evaluation value E, and the move selected as M.
        '''

        # We generate all possible moves, using board object's get all valid moves method.
        # This is needed to check if we are at a terminal state.
        if isPlayer1:
            moves = boardObj.get_all_valid_moves(p1color)
        else:
            moves = boardObj.get_all_valid_moves(p2color)

        # If we are too far deep in the state search tree, or the state is terminal (no other moves),
        # We return the static evaluation.
        # Static evaluation takes into account the 1st player's stones v. 2nd player's stones, and substracts.
        if (depth == 0) or moves is None:
            evaluation = boardObj.get_score()
            return evaluation[0] - evaluation[1], None

        # If we check the corners first, it will be beneficial for us.
        # This is why we create a lambda function, which generates 0 for a corner move, else 1.
        # If we sort (ascending) by this key function, we will basically explore the corner moves first.
        cornerlambda = lambda x : 0 if (x[0] in [0, self.board_size] and x[1] in [0, self.board_size]) else 1
        moves.sort(key=cornerlambda)

        a = alpha # a copy of the function argument alpha
        b = beta # a copy of the function argument beta
        m = None # move to be selected.

        # We must initialize the evaluation value, for each case (maximizing, or minimizing) of the players.
        if isPlayer1:
            value = -inf
        else:
            value = inf

        # We loop over all the possible moves.
        for move_possibility in moves:
            # We create a copy of the current board.
            boardcopy = boardObj.get_board_copy()
            # We create a new instance of the board, with this board's representation.
            newboard = MyBoard(self.board_size, boardcopy, p1color, p2color, -1)
            # We play the move on the board.
            if isPlayer1:
                newboard.play_move(move_possibility, p1color)
            else:
                newboard.play_move(move_possibility, p2color)

            # We obtain the evaluation score from recursive call.
            evalscore, _ = self.move_alpha_beta(newboard, a, b, (not isPlayer1), depth-1, p1color, p2color)

            # If we have a better solution as the maximizing player...
            if isPlayer1 and (evalscore > value):
                # Update the evaluation value.
                value = evalscore
                # Record this move (the best we've found)
                m = move_possibility
                # Update alpha value.
                a = max(a, value)
                # If we have a case of b <= a, that means we can prune the search, and not need to look anywhere else.
                if b <= a:
                    # Break the loop, we do not need to check any more possibilities.
                    break

            # If we have a better solution as the minimizing player...
            if (not isPlayer1) and (evalscore < value):
                # Update the evaluation value
                value = evalscore
                # Record this move (the best we've found)
                m = move_possibility
                # update beta value.
                b = min(b, value)
                # If we have a case of b <= a, that means we can prune the search, and not need to look anywhere else.
                if b <= a:
                    # Break the loop, we do not need to check any more possibilities.
                    break
        # Return the best evaluation value, with the move that produced it.
        return value, m



    def __is_correct_move(self, move, board):
        dx = [-1, -1, -1, 0, 1, 1, 1, 0]
        dy = [-1, 0, 1, 1, 1, 0, -1, -1]
        for i in range(len(dx)):
            if self.__confirm_direction(move, dx[i], dy[i], board)[0]:
                return True, 
        return False

    def __confirm_direction(self, move, dx, dy, board):
        posx = move[0]+dx
        posy = move[1]+dy
        opp_stones_inverted = 0
        if (posx >= 0) and (posx < self.board_size) and (posy >= 0) and (posy < self.board_size):
            if board[posx][posy] == self.opponent_color:
                opp_stones_inverted += 1
                while (posx >= 0) and (posx <= (self.board_size-1)) and (posy >= 0) and (posy <= (self.board_size-1)):
                    posx += dx
                    posy += dy
                    if (posx >= 0) and (posx < self.board_size) and (posy >= 0) and (posy < self.board_size):
                        if board[posx][posy] == -1:
                            return False, 0
                        if board[posx][posy] == self.my_color:
                            return True, opp_stones_inverted
                    opp_stones_inverted += 1

        return False, 0

    def get_all_valid_moves(self, board):
        valid_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if (board[x][y] == -1) and self.__is_correct_move([x, y], board):
                    valid_moves.append( (x, y) )

        if len(valid_moves) <= 0:
            print('No possible move!')
            return None
        return valid_moves
    
