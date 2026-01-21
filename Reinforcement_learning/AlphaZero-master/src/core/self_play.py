"""
Self-Play Data Generation for AlphaZero

This module implements the self-play mechanism where the agent
plays games against itself to generate training data.
"""

import numpy as np
from tqdm import tqdm
from .mcts import MCTS


class SelfPlay:
    """
    Self-play system for generating training data
    """
    
    def __init__(self, game, network, args):
        """
        Args:
            game: Game instance
            network: Neural network
            args: Training arguments
        """
        self.game = game
        self.network = network
        self.args = args
        self.mcts = MCTS(game, network, args)
        
    def execute_episode(self):
        """
        Play one game of self-play
        
        Returns:
            train_examples: List of (board, pi, v) training examples
                - board: Board state
                - pi: MCTS policy (improved policy from search)
                - v: Game outcome from this position
        """
        train_examples = []
        board = self.game.get_init_board()
        current_player = 1
        episode_step = 0
        
        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_form(board, current_player)
            
            # Temperature for exploration
            temp = 1 if episode_step < self.args.temp_threshold else 0
            
            # Get policy from MCTS
            pi = self.mcts.get_action_prob(canonical_board, temp=temp)
            
            # Get symmetries and add to training examples
            sym = self.game.get_symmetries(canonical_board, pi)
            for b, p in sym:
                # Convert board to network input format
                if hasattr(self.game, 'get_board_for_network'):
                    b_input = self.game.get_board_for_network(b, current_player)
                else:
                    b_input = b
                train_examples.append([b_input, current_player, p, None])
            
            # Sample action from policy
            action = np.random.choice(len(pi), p=pi)
            
            # Execute action
            board, current_player = self.game.get_next_state(board, current_player, action)
            
            # Check if game ended
            r = self.game.get_game_ended(board, current_player)
            
            if r != 0:
                # Game ended, assign rewards
                return_values = []
                for example in train_examples:
                    # Reward from perspective of each position's player
                    example[3] = r * ((-1) ** (example[1] != current_player))
                    return_values.append(example)
                
                return return_values
    
    def generate_self_play_data(self, num_episodes):
        """
        Generate training data from multiple self-play games
        
        Args:
            num_episodes: Number of games to play
            
        Returns:
            iteration_train_examples: List of training examples
        """
        iteration_train_examples = []
        
        print(f"Generating self-play data from {num_episodes} games...")
        for _ in tqdm(range(num_episodes)):
            # Reset MCTS search tree for new game
            self.mcts = MCTS(self.game, self.network, self.args)
            
            # Play one game
            episode_examples = self.execute_episode()
            iteration_train_examples.extend(episode_examples)
        
        return iteration_train_examples


class Arena:
    """
    Arena for evaluating two players against each other
    """
    
    def __init__(self, player1, player2, game):
        """
        Args:
            player1: Function that takes board and returns action
            player2: Function that takes board and returns action
            game: Game instance
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        
    def play_game(self, verbose=False):
        """
        Play one game between two players
        
        Returns:
            1: Player 1 won
            -1: Player 2 won
            0: Draw
        """
        players = [self.player2, None, self.player1]
        current_player = 1
        board = self.game.get_init_board()
        it = 0
        
        while self.game.get_game_ended(board, current_player) == 0:
            it += 1
            if verbose:
                print(f"Turn {it}, Player {current_player}")
                self.game.display(board)
            
            canonical_board = self.game.get_canonical_form(board, current_player)
            
            action = players[current_player + 1](canonical_board)
            
            valids = self.game.get_valid_moves(canonical_board, 1)
            
            if valids[action] == 0:
                print(f"Invalid move: {action}")
                print(f"Valid moves: {np.where(valids == 1)[0]}")
                assert valids[action] > 0
            
            board, current_player = self.game.get_next_state(board, current_player, action)
        
        if verbose:
            print(f"Game over: Turn {it}, Result {self.game.get_game_ended(board, 1)}")
            self.game.display(board)
        
        return current_player * self.game.get_game_ended(board, current_player)
    
    def play_games(self, num_games, verbose=False):
        """
        Play multiple games and return statistics
        
        Args:
            num_games: Number of games to play
            verbose: Print game details
            
        Returns:
            (wins, losses, draws): Statistics for player1
        """
        num = num_games // 2
        
        one_won = 0
        two_won = 0
        draws = 0
        
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            game_result = self.play_game(verbose=verbose)
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1
            else:
                draws += 1
        
        # Swap players
        self.player1, self.player2 = self.player2, self.player1
        
        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            game_result = self.play_game(verbose=verbose)
            if game_result == -1:
                one_won += 1
            elif game_result == 1:
                two_won += 1
            else:
                draws += 1
        
        return one_won, two_won, draws

