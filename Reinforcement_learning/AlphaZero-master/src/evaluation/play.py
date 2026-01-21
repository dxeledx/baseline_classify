"""
Play against the trained AlphaZero model

This script allows you to play games against the trained AI.
"""

import argparse
import torch
from connect4 import Connect4Game, TicTacToeGame
from chess_game import ChessGame
from utils import load_model, play_against_human, evaluate_against_random


def main():
    parser = argparse.ArgumentParser(description='Play against AlphaZero')
    parser.add_argument('--game', type=str, default='connect4',
                        choices=['connect4', 'tictactoe', 'chess'],
                        help='Game to play')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--human_first', action='store_true',
                        help='Human plays first')
    parser.add_argument('--mcts_sims', type=int, default=100,
                        help='Number of MCTS simulations for AI')
    parser.add_argument('--num_channels', type=int, default=256,
                        help='Number of channels in network')
    parser.add_argument('--num_res_blocks', type=int, default=10,
                        help='Number of residual blocks')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model against random player instead of playing')
    parser.add_argument('--eval_games', type=int, default=100,
                        help='Number of games for evaluation')
    
    args = parser.parse_args()
    
    # Initialize game
    if args.game == 'connect4':
        game = Connect4Game()
        print("\nConnect4 (6x7, win with 4 in a row)")
    elif args.game == 'tictactoe':
        game = TicTacToeGame()
        print("\nTic-Tac-Toe (3x3)")
    elif args.game == 'chess':
        game = ChessGame()
        print("\nChess (8x8, standard rules)")
    else:
        raise ValueError(f"Unknown game: {args.game}")
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    network = load_model(
        args.model,
        game,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks
    )
    
    if args.evaluate:
        # Evaluate against random player
        print(f"\nEvaluating model with {args.mcts_sims} MCTS simulations...")
        evaluate_against_random(
            game,
            network,
            num_games=args.eval_games,
            num_mcts_sims=args.mcts_sims
        )
    else:
        # Play against human
        print(f"\nAI using {args.mcts_sims} MCTS simulations per move")
        print("="*50)
        
        play_again = True
        while play_again:
            play_against_human(
                game,
                network,
                human_plays_first=args.human_first,
                num_mcts_sims=args.mcts_sims
            )
            
            response = input("\nPlay again? (y/n): ").lower()
            play_again = response == 'y'
            
            if play_again:
                # Switch sides
                args.human_first = not args.human_first
        
        print("\nThanks for playing!")


if __name__ == '__main__':
    main()

