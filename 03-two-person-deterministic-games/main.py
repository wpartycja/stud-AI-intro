from tic_tac_toe import TicTacToe


if __name__ == "__main__":

    player1 = 'X'
    player2 = 'O'
    tic_tac_toe = TicTacToe('X')
    tic_tac_toe.show_board()

    winner_stats = []
    looked_states_min = []
    looked_states_max = []

    for _ in range(100):
        while tic_tac_toe.check_winner() is None:
            tic_tac_toe.make_best_move_max(player1, 1, True)
            
            if tic_tac_toe.check_winner() is not None:
                break

            tic_tac_toe.make_best_move_min(player2, 0, True)
        
        winner_stats.append(tic_tac_toe.check_winner())
        looked_states_min.append(tic_tac_toe.min_states)
        looked_states_max.append(tic_tac_toe.max_states)
        tic_tac_toe.show_board()
        tic_tac_toe = TicTacToe('X') # ckeaing all parameters
        
    wins = winner_stats.count(player1)
    loses = winner_stats.count(player2)
    ties = winner_stats.count('tie')
    states_avg_min = sum(looked_states_min)/len(looked_states_min)
    states_avg_max = sum(looked_states_max)/len(looked_states_max)
    print(f'wins: {wins}\nloses: {loses}\nties: {ties}\nstates min: {states_avg_min}\nstates max: {states_avg_max}')
        
