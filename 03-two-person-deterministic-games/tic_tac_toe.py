class Board():
    def __init__(self):
        board = []
        for _ in range(3):
            board.append(['', '', ''])
        self.board = board
        self.players = ('X', 'O')

    def show_board(self):
        for line in self.board:
            print(line)
        print('\n')

    def is_free(self, coord):
        x, y = coord
        return True if self.board[x][y] == '' else False

    def make_move(self, coord, player):
        if self.is_free(coord):
            x, y = coord
            self.board[x][y] = player

    def check_winner(self):
        # horizontal
        for line in self.board:
            if line[0] == line[1] == line[2] != '':
                return line[0]

        # vertical
        for i in range(3):
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != '':
                return self.board[0][i]

        # diagonal
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != '':
            return self.board[0][0]
        
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != '':
            return self.board[0][2]

        return 0
        

tic_tac_toe = Board()
tic_tac_toe.show_board()
print(tic_tac_toe.check_winner())
for i in range(3):
    tic_tac_toe.make_move((i, i), 'X')

tic_tac_toe.show_board()
print(tic_tac_toe.check_winner())


