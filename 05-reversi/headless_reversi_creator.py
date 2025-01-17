import getopt
import sys
import time
import random_player
from game_board import GameBoard


class HeadlessReversiCreator(object):
    """
    Creator of the Reversi game without the GUI.
    """

    def __init__(self, player1, player1_color, player2, player2_color, board_size=8):
        """
        :param player1: Instance of first player
        :param player1_color: color of player1
        :param player2: Instance of second player
        :param player1_color: color of player2
        :param board_size: Board will have size [board_size x board_size]
        """
        self.board = GameBoard(board_size, player1_color, player2_color)
        self.player1 = player1
        self.player2 = player2
        self.current_player = self.player1
        self.current_player_color = player1_color
        self.player1_color = player1_color
        self.player2_color = player2_color

    def play_game(self):
        """
        This function contains game loop that plays the game.
        """
        # TODO: Time limit for move
        correct_finish = True
        while self.board.can_play(self.current_player_color):
            start_time = time.time()
            move = self.current_player.move(self.board.get_board_copy())
            end_time = time.time()
            move_time = (end_time - start_time) * 1000
            if move is None:
                print('Player %d returns None istead of a valid move. Move takes %.3f ms.' % (self.current_player_color,
                                                                                              move_time))
                correct_finish = False
                break
            else:
                print('Player %d wants move [%d,%d]. Move takes %.3f ms.' % (self.current_player_color,
                                                                             move[0],
                                                                             move[1],
                                                                             move_time))

            if self.board.is_correct_move(move, self.current_player_color):
                print('Move is correct')
                self.board.play_move(move, self.current_player_color)

            else:
                print('Player %d made the wrong move [%d,%d]' % (self.current_player_color,
                                                                 move[0],
                                                                 move[1]))
                correct_finish = False
                break

            self.change_player()
            if not self.board.can_play(self.current_player_color):
                print('No possible move for Player %d' % (self.current_player_color))
                self.change_player()
                if self.board.can_play(self.current_player_color):
                    print('Player %d plays again ' % (self.current_player_color))
                else:
                    print('Game over')

            self.board.print_board()
        if correct_finish:
            self.print_final_score()
        else:
            print('Game over.')
            if self.current_player_color == self.player1_color:
                print('Winner is player %d.' % (self.player2_color))
            else:
                print('Winner is player %d.' % (self.player1_color))

    def change_player(self):
        """
        Change the current_player
        """
        if self.current_player == self.player1:
            self.current_player = self.player2
            self.current_player_color = self.player2_color
        else:
            self.current_player = self.player1
            self.current_player_color = self.player1_color

    def print_final_score(self):
        p1_stones = 0
        p2_stones = 0
        for x in range(self.board.board_size):
            for y in range(self.board.board_size):
                if self.board.board[x][y] == 0:
                    p1_stones += 1
                if self.board.board[x][y] == 1:
                    p2_stones += 1

        print('\n\n-----------------------------\n')
        print('Final score:\n\nPlayer%d:Player%d\n\t[%d:%d]\n' % (self.player1_color,
                                                                  self.player2_color,
                                                                  p1_stones,
                                                                  p2_stones))
        if p1_stones > p2_stones:
            print('Player %d wins!' % (self.player1_color))
        elif p2_stones > p1_stones:
            print('Player %d wins!' % (self.player2_color))
        else:
            print('Draw')
        print('\n-----------------------------\n\n')


if __name__ == "__main__":
    (choices, args) = getopt.getopt(sys.argv[1:], "")
    p1_color = 0
    p2_color = 1

    if len(args) == 0:
        print('No arguments given.\nRunning game with two random players.')
        p1 = random_player.MyPlayer(p1_color, p2_color)
        p2 = random_player.MyPlayer(p2_color, p1_color)
        game = HeadlessReversiCreator(p1, p1_color, p2, p2_color, 8)
        game.play_game()

    elif len(args) == 1:
        print('One player given in argument.\nRunning game with given player against the random player.')
        p1 = random_player.MyPlayer(p1_color, p2_color)
        try:
            player_module = __import__(args[0])
            p2 = player_module.MyPlayer(p2_color, p1_color)

            game = HeadlessReversiCreator(p1, p1_color, p2, p2_color, 8)
            game.play_game()

        except ImportError:
            print('Error: Cannot import given player: %s.' % (args[0]))

    else:
        if len(args) > 2:
            print('More than two arguments given. Ignoring other arguments and using only the first and the second as players.')

        importsCorrect = True
        try:
            player_module = __import__(args[0])
            p1 = player_module.MyPlayer(p1_color, p2_color)
        except ImportError:
            importsCorrect = False
            print('Error: Cannot import given player: %s.' %(args[0]))

        try:
            player_module = __import__(args[1])
            p2 = player_module.MyPlayer(p2_color, p1_color)
        except ImportError:
            importsCorrect = False
            print('Error: Cannot import given player: %s.' %(args[1]))

        if importsCorrect:
            game = HeadlessReversiCreator(p1, p1_color, p2, p2_color, 8)
            game.play_game()

