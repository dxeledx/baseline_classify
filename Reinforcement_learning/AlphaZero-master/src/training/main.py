"""
Main Training Script for AlphaZero

This script sets up and runs the complete AlphaZero training pipeline.
"""

import os
import argparse
import torch
from neural_network import AlphaZeroNetwork
from coach import Coach, TrainingArgs
from connect4 import Connect4Game, TicTacToeGame


def main():
    """Main training function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='AlphaZero Training')
    parser.add_argument('--game', type=str, default='connect4',
                        choices=['connect4', 'tictactoe'],
                        help='Game to train on')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of training iterations')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of self-play episodes per iteration')
    parser.add_argument('--mcts_sims', type=int, default=50,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs per iteration')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_channels', type=int, default=256,
                        help='Number of channels in CNN')
    parser.add_argument('--num_res_blocks', type=int, default=10,
                        help='Number of residual blocks')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to model checkpoint to resume training')
    parser.add_argument('--arena_games', type=int, default=40,
                        help='Number of arena games for evaluation')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    
    args = parser.parse_args()
    
    # Setup
    print("="*70)
    print("AlphaZero Training")
    print("="*70)
    print(f"Game: {args.game}")
    print(f"Iterations: {args.iterations}")
    print(f"Episodes per iteration: {args.episodes}")
    print(f"MCTS simulations: {args.mcts_sims}")
    print(f"Device: {'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'}")
    print("="*70)
    
    # Initialize game
    if args.game == 'connect4':
        game = Connect4Game()
        print("\nGame: Connect4 (6x7 board, win with 4 in a row)")
    elif args.game == 'tictactoe':
        game = TicTacToeGame()
        print("\nGame: Tic-Tac-Toe (3x3 board)")
    else:
        raise ValueError(f"Unknown game: {args.game}")
    
    # Initialize neural network
    print(f"\nInitializing neural network...")
    print(f"- Channels: {args.num_channels}")
    print(f"- Residual blocks: {args.num_res_blocks}")
    
    network = AlphaZeroNetwork(
        game=game,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        dropout=0.3
    )
    
    # Load model if specified
    if args.load_model:
        print(f"\nLoading model from {args.load_model}")
        device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        network.load_state_dict(torch.load(args.load_model, map_location=device))
    
    # Setup training arguments
    training_args = TrainingArgs()
    training_args.num_iterations = args.iterations
    training_args.num_episodes = args.episodes
    training_args.num_mcts_sims = args.mcts_sims
    training_args.epochs = args.epochs
    training_args.batch_size = args.batch_size
    training_args.lr = args.lr
    training_args.num_channels = args.num_channels
    training_args.num_res_blocks = args.num_res_blocks
    training_args.checkpoint_dir = args.checkpoint_dir
    training_args.arena_compare_games = args.arena_games
    training_args.device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize coach and start training
    print("\nInitializing coach...")
    coach = Coach(game, network, training_args)
    
    print("\nStarting training...")
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    
    try:
        coach.learn()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model...")
        coach._save_checkpoint(0, 'interrupted')
        print("Model saved. You can resume training with --load_model")
    
    print("\n" + "="*70)
    print("Training complete!")
    print(f"Best model saved in: {args.checkpoint_dir}/best.pth")
    print("="*70)


if __name__ == '__main__':
    main()

