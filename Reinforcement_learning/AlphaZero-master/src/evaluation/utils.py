"""
Utility Functions for AlphaZero

Helper functions for visualization, analysis, and playing against the trained model.
"""

import numpy as np
import torch
from ..core.mcts import MCTS, DummyArgs


class HumanPlayer:
    """Interactive human player"""
    
    def __init__(self, game):
        self.game = game
    
    def play(self, board):
        """
        Get move from human player
        
        Args:
            board: Current board state
            
        Returns:
            action: Selected action
        """
        valid = self.game.get_valid_moves(board, 1)
        
        while True:
            self.game.display(board)
            
            if hasattr(self.game, 'get_action_size'):
                action_size = self.game.get_action_size()
            else:
                action_size = len(valid)
            
            print(f"Valid moves: {[i for i, v in enumerate(valid) if v == 1]}")
            
            try:
                action = int(input("Enter your move: "))
                if 0 <= action < action_size and valid[action] == 1:
                    return action
                else:
                    print("Invalid move! Try again.")
            except (ValueError, IndexError):
                print("Invalid input! Please enter a number.")


class AIPlayer:
    """AI player using trained model"""
    
    def __init__(self, game, network, num_mcts_sims=100, temp=0):
        """
        Args:
            game: Game instance
            network: Trained neural network
            num_mcts_sims: Number of MCTS simulations
            temp: Temperature for move selection
        """
        self.game = game
        self.network = network
        self.num_mcts_sims = num_mcts_sims
        self.temp = temp
        self.args = DummyArgs(num_mcts_sims=num_mcts_sims)
    
    def play(self, board):
        """
        Get move from AI player
        
        Args:
            board: Current board state
            
        Returns:
            action: Selected action
        """
        mcts = MCTS(self.game, self.network, self.args)
        pi = mcts.get_action_prob(board, temp=self.temp)
        
        if self.temp == 0:
            # Deterministic: choose best move
            action = np.argmax(pi)
        else:
            # Stochastic: sample from distribution
            action = np.random.choice(len(pi), p=pi)
        
        return action


class RandomPlayer:
    """Random player for baseline comparison"""
    
    def __init__(self, game):
        self.game = game
    
    def play(self, board):
        """
        Get random valid move
        
        Args:
            board: Current board state
            
        Returns:
            action: Random valid action
        """
        valid = self.game.get_valid_moves(board, 1)
        valid_moves = [i for i, v in enumerate(valid) if v == 1]
        return np.random.choice(valid_moves)


def play_game(game, player1, player2, display=True):
    """
    Play a game between two players
    
    Args:
        game: Game instance
        player1: Player 1 (plays first)
        player2: Player 2
        display: Whether to display the game
        
    Returns:
        result: 1 if player1 won, -1 if player2 won, 0 for draw
    """
    board = game.get_init_board()
    current_player = 1
    move_count = 0
    
    players = {1: player1, -1: player2}
    
    while True:
        move_count += 1
        
        if display:
            print(f"\nMove {move_count}, Player {current_player}")
            game.display(board)
        
        # Get canonical form
        canonical_board = game.get_canonical_form(board, current_player)
        
        # Get action
        action = players[current_player].play(canonical_board)
        
        # Validate action
        valids = game.get_valid_moves(canonical_board, 1)
        if valids[action] == 0:
            print(f"Invalid move! Player {current_player} loses.")
            return -current_player
        
        # Execute action
        board, current_player = game.get_next_state(board, current_player, action)
        
        # Check if game ended
        r = game.get_game_ended(board, current_player)
        
        if r != 0:
            if display:
                print(f"\nGame Over!")
                game.display(board)
                
                if abs(r) < 0.01:
                    print("Result: Draw")
                elif r == 1:
                    print(f"Winner: Player {current_player}")
                else:
                    print(f"Winner: Player {-current_player}")
            
            return -r * current_player


def evaluate_against_random(game, network, num_games=100, num_mcts_sims=50):
    """
    Evaluate trained model against random player
    
    Args:
        game: Game instance
        network: Trained neural network
        num_games: Number of games to play
        num_mcts_sims: MCTS simulations for AI
        
    Returns:
        (wins, losses, draws): Statistics
    """
    ai_player = AIPlayer(game, network, num_mcts_sims=num_mcts_sims)
    random_player = RandomPlayer(game)
    
    wins = 0
    losses = 0
    draws = 0
    
    print(f"Evaluating against random player ({num_games} games)...")
    
    for i in range(num_games):
        # Alternate who plays first
        if i % 2 == 0:
            result = play_game(game, ai_player, random_player, display=False)
        else:
            result = -play_game(game, random_player, ai_player, display=False)
        
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{num_games} - W:{wins} L:{losses} D:{draws}")
    
    print(f"\nResults: Wins={wins}, Losses={losses}, Draws={draws}")
    print(f"Win Rate: {wins/num_games:.2%}")
    
    return wins, losses, draws


def load_model(checkpoint_path, game, num_channels=256, num_res_blocks=10):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        game: Game instance
        num_channels: Number of channels in network
        num_res_blocks: Number of residual blocks
        
    Returns:
        network: Loaded neural network
    """
    from ..core.neural_network import AlphaZeroNetwork
    
    network = AlphaZeroNetwork(
        game=game,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        dropout=0.3
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.load_state_dict(torch.load(checkpoint_path, map_location=device))
    network.to(device)
    network.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    
    return network


def play_against_human(game, network, human_plays_first=True, num_mcts_sims=100):
    """
    Interactive game: human vs AI
    
    Args:
        game: Game instance
        network: Trained neural network
        human_plays_first: Whether human plays first
        num_mcts_sims: MCTS simulations for AI
    """
    human = HumanPlayer(game)
    ai = AIPlayer(game, network, num_mcts_sims=num_mcts_sims, temp=0)
    
    if human_plays_first:
        print("\nYou are X (Player 1)")
        result = play_game(game, human, ai, display=True)
        if result == 1:
            print("Congratulations! You won!")
        elif result == -1:
            print("AI wins!")
        else:
            print("It's a draw!")
    else:
        print("\nYou are O (Player 2)")
        result = play_game(game, ai, human, display=True)
        if result == -1:
            print("Congratulations! You won!")
        elif result == 1:
            print("AI wins!")
        else:
            print("It's a draw!")


def plot_training_progress(checkpoint_dir):
    """
    Plot training progress from saved examples
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    """
    import os
    import pickle
    import matplotlib.pyplot as plt
    
    iterations = []
    num_examples = []
    
    # Load training examples from each iteration
    for filename in sorted(os.listdir(checkpoint_dir)):
        if filename.startswith('examples_iter_') and filename.endswith('.pkl'):
            iter_num = int(filename.split('_')[2].split('.')[0])
            filepath = os.path.join(checkpoint_dir, filename)
            
            with open(filepath, 'rb') as f:
                examples = pickle.load(f)
            
            iterations.append(iter_num)
            num_examples.append(len(examples))
    
    # Plot
    if iterations:
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, num_examples, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Number of Training Examples')
        plt.title('Training Progress: Examples per Iteration')
        plt.grid(True)
        plt.savefig(os.path.join(checkpoint_dir, 'training_progress.png'))
        plt.close()
        print(f"Training progress plot saved to {checkpoint_dir}/training_progress.png")
    else:
        print("No training examples found.")

