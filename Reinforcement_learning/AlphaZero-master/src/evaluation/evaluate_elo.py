"""
Evaluate Chess Engine ELO Rating

This script runs a tournament to estimate the ELO rating of your
AlphaZero chess engine against various opponents.
"""

import argparse
import torch
import numpy as np
from chess_game import ChessGame
from neural_network import AlphaZeroNetwork
from elo_rating import (
    Tournament, ELOCalculator, AlphaZeroEngine, 
    RandomEngine, MaterialEngine, estimate_elo
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Chess Engine ELO')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--mcts_sims', type=int, default=100,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--num_channels', type=int, default=256,
                        help='Number of channels in network')
    parser.add_argument('--num_res_blocks', type=int, default=19,
                        help='Number of residual blocks')
    parser.add_argument('--games_per_opponent', type=int, default=20,
                        help='Number of games against each opponent')
    parser.add_argument('--max_moves', type=int, default=200,
                        help='Maximum moves per game')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    parser.add_argument('--quick', action='store_true',
                        help='Quick evaluation (fewer games, fewer MCTS sims)')
    
    args = parser.parse_args()
    
    # Adjust for quick evaluation
    if args.quick:
        args.games_per_opponent = 10
        args.mcts_sims = 50
        print("\n⚡ Quick evaluation mode enabled")
    
    print("="*70)
    print("Chess Engine ELO Evaluation")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"MCTS simulations: {args.mcts_sims}")
    print(f"Games per opponent: {args.games_per_opponent}")
    print(f"Max moves per game: {args.max_moves}")
    print("="*70)
    
    # Initialize chess game
    game = ChessGame()
    
    # Load model
    print("\nLoading model...")
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    network = AlphaZeroNetwork(
        game=game,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        dropout=0.3
    )
    network.load_state_dict(torch.load(args.model, map_location=device))
    network.to(device)
    network.eval()
    print(f"✓ Model loaded on {device}")
    
    # Create engines
    print("\nInitializing engines...")
    
    # Your AlphaZero engine
    alphazero = AlphaZeroEngine(
        game=game,
        network=network,
        name="AlphaZero",
        initial_rating=1500,  # Start at average
        num_mcts_sims=args.mcts_sims,
        temp=0  # Deterministic play for evaluation
    )
    
    # Opponent engines with known approximate ELO
    random_weak = RandomEngine("Random-Weak", initial_rating=500)
    random_normal = RandomEngine("Random-Normal", initial_rating=800)
    material_simple = MaterialEngine("Material-Simple", initial_rating=1200)
    
    engines = [alphazero, random_weak, random_normal, material_simple]
    
    print(f"✓ {len(engines)} engines ready")
    for engine in engines:
        print(f"  - {engine.name}: Starting ELO {engine.rating:.0f}")
    
    # Create tournament
    elo_calculator = ELOCalculator(k_factor=32)
    tournament = Tournament(engines, elo_calculator)
    
    # Run tournament
    print("\n" + "="*70)
    print("Starting Tournament")
    print("="*70)
    print(f"This will play ~{len(engines) * (len(engines)-1) * args.games_per_opponent // 2} games")
    print("Depending on your settings, this may take 30 minutes to several hours")
    print()
    
    tournament.run_round_robin(
        num_games=args.games_per_opponent,
        max_moves=args.max_moves,
        verbose=False
    )
    
    # Estimate ELO with confidence interval
    print("\n" + "="*70)
    print("ELO RATING ANALYSIS")
    print("="*70)
    
    elo_stats = estimate_elo("AlphaZero", tournament)
    
    print(f"\n{'='*70}")
    print(f"  YOUR CHESS ENGINE RATING")
    print(f"{'='*70}")
    print(f"  Engine:           {elo_stats['engine']}")
    print(f"  ELO Rating:       {elo_stats['rating']:.0f}")
    print(f"  95% CI:           {elo_stats['ci_low']:.0f} - {elo_stats['ci_high']:.0f}")
    print(f"  Strength:         {elo_stats['strength']}")
    print(f"  Games Played:     {elo_stats['games']}")
    print(f"  Record:           {elo_stats['wins']}W - {elo_stats['draws']}D - {elo_stats['losses']}L")
    print(f"{'='*70}")
    
    # Performance breakdown
    print("\n" + "="*70)
    print("PERFORMANCE BREAKDOWN")
    print("="*70)
    
    for game_record in tournament.game_history:
        if game_record['white'] == 'AlphaZero' or game_record['black'] == 'AlphaZero':
            is_white = game_record['white'] == 'AlphaZero'
            opponent = game_record['black'] if is_white else game_record['white']
            result = game_record['result'] if is_white else 1.0 - game_record['result']
            
            result_str = "WIN" if result == 1.0 else ("DRAW" if result == 0.5 else "LOSS")
            print(f"  vs {opponent:<20} : {result_str}")
    
    # Rating interpretation
    print("\n" + "="*70)
    print("RATING INTERPRETATION")
    print("="*70)
    print()
    print("ELO Rating Ranges:")
    print("  <1000   : Beginner (knows rules, random-like play)")
    print("  1000-1400: Novice (understands basic tactics)")
    print("  1400-1800: Intermediate (solid tactics, simple strategy)")
    print("  1800-2000: Advanced (good tactics, developing strategy)")
    print("  2000-2200: Expert (strong tactical and strategic play)")
    print("  2200-2400: Master (near-professional level)")
    print("  2400-2600: International Master")
    print("  2600+    : Grandmaster (world-class)")
    print()
    print("Note: These ratings are estimates based on limited games against")
    print("      simple opponents. For accurate rating, play against rated")
    print("      opponents (e.g., chess engines with known ELO).")
    print()
    
    # Recommendations
    print("="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if elo_stats['rating'] < 1000:
        print("""
Your engine is still learning basic chess concepts.
Recommendations:
  1. Train for more iterations (50-100 more)
  2. Increase MCTS simulations during training
  3. Increase number of self-play games per iteration
  4. Consider using a larger network
        """)
    elif elo_stats['rating'] < 1400:
        print("""
Your engine understands basic tactics but needs improvement.
Recommendations:
  1. Continue training (30-50 more iterations)
  2. Ensure diverse self-play data
  3. Consider increasing network depth
  4. Monitor training loss to ensure convergence
        """)
    elif elo_stats['rating'] < 1800:
        print("""
Your engine has solid tactical understanding!
Recommendations:
  1. Train longer for strategic improvement
  2. Evaluate against stronger opponents
  3. Consider implementing opening book
  4. Fine-tune with high-quality games
        """)
    elif elo_stats['rating'] < 2000:
        print("""
Excellent! Your engine plays at an advanced level.
Recommendations:
  1. Continue training for strategic refinement
  2. Test against established chess engines (Stockfish at low depth)
  3. Analyze critical positions for improvement
  4. Consider ensemble methods or larger networks
        """)
    else:
        print("""
Outstanding! Your engine is playing at expert level or higher!
Recommendations:
  1. Test against strong engines (Stockfish, Leela Chess Zero)
  2. Participate in computer chess tournaments
  3. Analyze games for blind spots
  4. Consider publishing your results!
        """)
    
    print("="*70)
    print("\nEvaluation complete!")
    print(f"Your chess engine's estimated ELO: {elo_stats['rating']:.0f}")
    print(f"Classification: {elo_stats['strength']}")
    print("="*70)


if __name__ == '__main__':
    main()

