from math import inf
from random import choice


class TicTacToe():
    def __init__(self, max_player):
        board = []
        for _ in range(3):
            board.append([' ', ' ', ' '])
        self.board = board
        self.free_spots = []
        for x in range(3):
            for y in range(3):
                self.free_spots.append((x, y))
        if max_player == 'X':
            self.max_player = 'X'
            self.min_player = 'O'
            self.score = {'X': 1, 'O': -1, 'tie': 0}
        else:
            self.max_player = 'O'
            self.min_player = 'X'
            self.score = {'O': 1, 'X': -1, 'tie': 0}
        self.max_states = 0
        self.min_states = 0

    def show_board(self):
        for line in self.board:
            print(line)
        print('\n')

    def is_free(self, coord):
        x, y = coord
        return True if self.board[x][y] == ' ' else False

    def make_move(self, coord, player):
        if self.is_free(coord):
            x, y = coord
            self.board[x][y] = player
            self.free_spots.remove((x, y))

    def undo_move(self, coord):
        x, y = coord
        self.board[x][y] = ' '
        self.free_spots.append((x, y))

    def check_winner(self):
        winner = None

        # horizontal
        for line in self.board:
            if line[0] == line[1] == line[2] != ' ':
                winner = line[0]

        # vertical
        for i in range(3):
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != ' ':
                winner = self.board[0][i]

        # diagonal
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            winner = self.board[0][0]

        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            winner = self.board[0][2]

        # tie
        if winner is None and len(self.free_spots) == 0:
            winner = 'tie'

        return winner

    def random_move(self, player):
        coord = choice(self.free_spots)
        self.make_move(coord, player)

    def minimax(self, depth, is_max):

        # check if someone wins (or there is a tie)
        if depth == 0 or self.check_winner() is not None:
            if is_max:
                self.max_states += 1
            else:
                self.min_states += 1
            return self.score[self.check_winner()]*depth

        player = (self.max_player if is_max else self.min_player)

        score = []
        for x in range(3):
            for y in range(3):
                spot = x, y
                if self.is_free(spot):
                    self.make_move(spot, player)
                    score.append(self.minimax(depth-1, not is_max))
                    self.undo_move(spot)

        return max(score) if is_max else min(score)

    def make_best_move_max(self, player, depth, ab,):
        best_score = -inf
        best_move = None

        for x in range(3):
            for y in range(3):
                spot = x, y
                if self.is_free(spot):
                    self.make_move(spot, player)
                    score = self.alfa_beta(depth, False, -inf, inf) if ab else self.minimax(depth, False)
                    self.undo_move(spot)

                    if score > best_score:
                        best_score = score
                        best_move = spot

        print(f'max looked states: {self.max_states}')
        self.make_move(best_move, player)

    def make_best_move_min(self, player, depth, ab):
        best_score = inf
        best_move = None

        for x in range(3):
            for y in range(3):
                spot = x, y
                if self.is_free(spot):
                    self.make_move(spot, player)
                    score = self.alfa_beta(depth, True, -inf, inf) if ab else self.minimax(depth, True)
                    self.undo_move(spot)

                    if score < best_score:
                        best_score = score
                        best_move = spot

        print(f'min looked states: {self.min_states}')
        self.make_move(best_move, player)

    def alfa_beta(self, depth, is_max, alfa, beta):
        # check if someone wins (or there is a tie)
        if depth == 0 or self.check_winner() is not None:
            if is_max:
                self.max_states += 1
            else:
                self.min_states += 1
            return self.score[self.check_winner()]*depth

        player = (self.max_player if is_max else self.min_player)

        if is_max:
            value = -inf
            score = []
            for x in range(3):
                for y in range(3):
                    spot = x, y
                    if self.is_free(spot):
                        self.make_move(spot, player)
                        score.append(self.alfa_beta(depth-1, not is_max, alfa, beta))
                        self.undo_move(spot)
                        value = max(score)
                        if value >= beta:
                            break
                    alfa = max(alfa, value)
        else:
            value = inf
            score = []
            for x in range(3):
                for y in range(3):
                    spot = x, y
                    if self.is_free(spot):
                        self.make_move(spot, player)
                        score.append(self.alfa_beta(depth-1, not is_max, alfa, beta))
                        self.undo_move(spot)
                        value = min(score)
                        if value <= alfa:
                            break
                    beta = min(beta, value)

        return value
