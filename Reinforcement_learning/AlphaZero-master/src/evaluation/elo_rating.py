"""
ELO Rating System for Chess Engines

Implements standard ELO rating calculations and tournament management
to evaluate the strength of trained AlphaZero models.
"""

import numpy as np
import chess
from collections import defaultdict
from tqdm import tqdm
import time


class ELOCalculator:
    """Calculate ELO ratings based on game results"""
    
    def __init__(self, k_factor=32):
        """
        Args:
            k_factor: K-factor for ELO calculation (higher = more volatile)
                      32 is standard for new players
                      16 is standard for established players
        """
        self.k_factor = k_factor
    
    def expected_score(self, rating_a, rating_b):
        """
        Calculate expected score for player A against player B
        
        Args:
            rating_a: ELO rating of player A
            rating_b: ELO rating of player B
            
        Returns:
            expected: Expected score (0.0 to 1.0)
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    
    def update_rating(self, rating, expected, actual):
        """
        Update ELO rating based on game result
        
        Args:
            rating: Current ELO rating
            expected: Expected score (0.0 to 1.0)
            actual: Actual score (1.0 for win, 0.5 for draw, 0.0 for loss)
            
        Returns:
            new_rating: Updated ELO rating
        """
        return rating + self.k_factor * (actual - expected)
    
    def update_ratings(self, rating_a, rating_b, result):
        """
        Update ratings for both players after a game
        
        Args:
            rating_a: Player A's current rating
            rating_b: Player B's current rating
            result: Game result from A's perspective (1.0, 0.5, or 0.0)
            
        Returns:
            (new_rating_a, new_rating_b): Updated ratings
        """
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a
        
        new_rating_a = self.update_rating(rating_a, expected_a, result)
        new_rating_b = self.update_rating(rating_b, expected_b, 1.0 - result)
        
        return new_rating_a, new_rating_b


class ChessEngine:
    """Base class for chess engines"""
    
    def __init__(self, name, initial_rating=1500):
        self.name = name
        self.rating = initial_rating
        self.games_played = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
    
    def get_move(self, board):
        """
        Get move for current position
        
        Args:
            board: chess.Board object
            
        Returns:
            move: chess.Move object
        """
        raise NotImplementedError
    
    def update_stats(self, result):
        """Update engine statistics"""
        self.games_played += 1
        if result == 1.0:
            self.wins += 1
        elif result == 0.5:
            self.draws += 1
        else:
            self.losses += 1
    
    def get_stats(self):
        """Get engine statistics"""
        return {
            'name': self.name,
            'rating': self.rating,
            'games': self.games_played,
            'wins': self.wins,
            'draws': self.draws,
            'losses': self.losses,
            'win_rate': self.wins / max(self.games_played, 1)
        }
    
    def __str__(self):
        return f"{self.name} (ELO: {self.rating:.0f})"


class RandomEngine(ChessEngine):
    """Random move engine for baseline"""
    
    def __init__(self, name="Random", initial_rating=800):
        super().__init__(name, initial_rating)
    
    def get_move(self, board):
        """Select random legal move"""
        legal_moves = list(board.legal_moves)
        return np.random.choice(legal_moves) if legal_moves else None


class AlphaZeroEngine(ChessEngine):
    """AlphaZero-trained engine"""
    
    def __init__(self, game, network, name="AlphaZero", initial_rating=1500, 
                 num_mcts_sims=100, temp=0):
        super().__init__(name, initial_rating)
        self.game = game
        self.network = network
        self.num_mcts_sims = num_mcts_sims
        self.temp = temp
        
        from mcts import MCTS, DummyArgs
        self.args = DummyArgs(num_mcts_sims=num_mcts_sims)
    
    def get_move(self, board):
        """Get move using MCTS and neural network"""
        from ..core.mcts import MCTS
        
        # Determine player (1 for white, -1 for black)
        player = 1 if board.turn == chess.WHITE else -1
        
        # Get canonical form
        canonical_board = self.game.get_canonical_form(board, player)
        
        # Run MCTS
        mcts = MCTS(self.game, self.network, self.args)
        pi = mcts.get_action_prob(canonical_board, temp=self.temp)
        
        # Get valid moves
        valids = self.game.get_valid_moves(canonical_board, 1)
        
        # Mask invalid moves
        pi = pi * valids
        if np.sum(pi) > 0:
            pi = pi / np.sum(pi)
        else:
            # Fallback to uniform over valid moves
            pi = valids / np.sum(valids)
        
        # Select move
        if self.temp == 0:
            action = np.argmax(pi)
        else:
            action = np.random.choice(len(pi), p=pi)
        
        # Convert action to move
        move = self.game._action_to_move(canonical_board, action)
        
        # Verify it's legal
        if move not in board.legal_moves:
            # Fallback: pick first legal move
            legal_moves = list(board.legal_moves)
            return legal_moves[0] if legal_moves else None
        
        return move


class MaterialEngine(ChessEngine):
    """Simple engine that maximizes material"""
    
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    def __init__(self, name="Material", initial_rating=1200):
        super().__init__(name, initial_rating)
    
    def evaluate_board(self, board):
        """Evaluate board based on material count"""
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.PIECE_VALUES[piece.piece_type]
                score += value if piece.color == chess.WHITE else -value
        return score
    
    def get_move(self, board):
        """Select move that maximizes material"""
        best_score = float('-inf')
        best_move = None
        
        for move in board.legal_moves:
            board_copy = board.copy()
            board_copy.push(move)
            
            score = self.evaluate_board(board_copy)
            if not board.turn:  # If black's turn, negate
                score = -score
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move if best_move else list(board.legal_moves)[0]


class Tournament:
    """Manage chess tournament and calculate ELO ratings"""
    
    def __init__(self, engines, elo_calculator=None):
        """
        Args:
            engines: List of ChessEngine objects
            elo_calculator: ELOCalculator instance
        """
        self.engines = {engine.name: engine for engine in engines}
        self.elo_calculator = elo_calculator or ELOCalculator()
        self.game_history = []
    
    def play_game(self, engine1, engine2, max_moves=100, verbose=False):
        """
        Play one game between two engines
        
        Args:
            engine1: First engine (plays white)
            engine2: Second engine (plays black)
            max_moves: Maximum moves before declaring draw
            verbose: Print game progress
            
        Returns:
            result: 1.0 if engine1 wins, 0.0 if engine2 wins, 0.5 for draw
        """
        board = chess.Board()
        move_count = 0
        
        if verbose:
            print(f"\n{engine1.name} (White) vs {engine2.name} (Black)")
        
        while not board.is_game_over() and move_count < max_moves:
            # Get current engine
            current_engine = engine1 if board.turn == chess.WHITE else engine2
            
            try:
                # Get move with timeout
                move = current_engine.get_move(board)
                
                if move is None or move not in board.legal_moves:
                    # Invalid move - opponent wins
                    result = 0.0 if board.turn == chess.WHITE else 1.0
                    if verbose:
                        print(f"Invalid move by {current_engine.name}")
                    return result
                
                board.push(move)
                move_count += 1
                
                if verbose and move_count % 10 == 0:
                    print(f"Move {move_count}: {board.fen()}")
                
            except Exception as e:
                # Error in engine - opponent wins
                if verbose:
                    print(f"Error in {current_engine.name}: {e}")
                result = 0.0 if board.turn == chess.WHITE else 1.0
                return result
        
        # Game ended
        if board.is_checkmate():
            # Winner is opposite of current turn
            result = 0.0 if board.turn == chess.WHITE else 1.0
        else:
            # Draw (stalemate, insufficient material, or max moves)
            result = 0.5
        
        if verbose:
            outcome = "1-0" if result == 1.0 else ("0-1" if result == 0.0 else "1/2-1/2")
            print(f"Game over: {outcome}")
            if board.is_checkmate():
                print("Checkmate!")
            elif board.is_stalemate():
                print("Stalemate!")
            elif move_count >= max_moves:
                print("Max moves reached!")
        
        return result
    
    def run_round_robin(self, num_games=10, max_moves=100, verbose=False):
        """
        Run round-robin tournament
        
        Args:
            num_games: Number of games between each pair (alternating colors)
            max_moves: Maximum moves per game
            verbose: Print progress
        """
        engine_names = list(self.engines.keys())
        total_games = len(engine_names) * (len(engine_names) - 1) * num_games // 2
        
        print(f"\nStarting round-robin tournament with {len(engine_names)} engines")
        print(f"Total games: {total_games}")
        print("="*60)
        
        pbar = tqdm(total=total_games, desc="Tournament progress")
        
        for i, name1 in enumerate(engine_names):
            for name2 in engine_names[i+1:]:
                engine1 = self.engines[name1]
                engine2 = self.engines[name2]
                
                for game_num in range(num_games):
                    # Alternate colors
                    if game_num % 2 == 0:
                        white, black = engine1, engine2
                    else:
                        white, black = engine2, engine1
                    
                    # Play game
                    result = self.play_game(white, black, max_moves, verbose=False)
                    
                    # Update ratings
                    if white == engine1:
                        result1, result2 = result, 1.0 - result
                    else:
                        result1, result2 = 1.0 - result, result
                    
                    engine1.rating, engine2.rating = self.elo_calculator.update_ratings(
                        engine1.rating, engine2.rating, result1
                    )
                    
                    # Update stats
                    engine1.update_stats(result1)
                    engine2.update_stats(result2)
                    
                    # Record game
                    self.game_history.append({
                        'white': white.name,
                        'black': black.name,
                        'result': result,
                        'white_rating': white.rating,
                        'black_rating': black.rating
                    })
                    
                    pbar.update(1)
        
        pbar.close()
        print("\nTournament complete!")
        self.print_standings()
    
    def print_standings(self):
        """Print tournament standings"""
        print("\n" + "="*80)
        print("TOURNAMENT STANDINGS")
        print("="*80)
        print(f"{'Rank':<6} {'Engine':<20} {'ELO':<8} {'Games':<7} {'W':<5} {'D':<5} {'L':<5} {'Win%':<7}")
        print("-"*80)
        
        # Sort by rating
        sorted_engines = sorted(self.engines.values(), key=lambda e: e.rating, reverse=True)
        
        for rank, engine in enumerate(sorted_engines, 1):
            stats = engine.get_stats()
            print(f"{rank:<6} {stats['name']:<20} {stats['rating']:<8.0f} "
                  f"{stats['games']:<7} {stats['wins']:<5} {stats['draws']:<5} "
                  f"{stats['losses']:<5} {stats['win_rate']:<7.1%}")
        
        print("="*80)


def estimate_elo(your_engine_name, tournament_results):
    """
    Estimate ELO rating with confidence interval
    
    Args:
        your_engine_name: Name of the engine to estimate
        tournament_results: Tournament object with results
        
    Returns:
        dict with rating, confidence interval, and strength description
    """
    engine = tournament_results.engines[your_engine_name]
    rating = engine.rating
    games = engine.games_played
    
    # Calculate confidence interval (simplified)
    # Standard error decreases with more games
    std_error = 200 / np.sqrt(max(games, 1))
    confidence_95 = 1.96 * std_error
    
    # Strength classification
    if rating < 1000:
        strength = "Beginner"
    elif rating < 1400:
        strength = "Novice"
    elif rating < 1800:
        strength = "Intermediate"
    elif rating < 2000:
        strength = "Advanced"
    elif rating < 2200:
        strength = "Expert"
    elif rating < 2400:
        strength = "Master"
    elif rating < 2600:
        strength = "International Master"
    else:
        strength = "Grandmaster"
    
    return {
        'engine': your_engine_name,
        'rating': rating,
        'games': games,
        'ci_low': rating - confidence_95,
        'ci_high': rating + confidence_95,
        'strength': strength,
        'wins': engine.wins,
        'draws': engine.draws,
        'losses': engine.losses
    }


if __name__ == '__main__':
    print("ELO Rating System Test")
    print("="*60)
    
    # Create test engines
    random_engine = RandomEngine("Random", 800)
    material_engine = MaterialEngine("Material", 1200)
    
    # Create tournament
    engines = [random_engine, material_engine]
    tournament = Tournament(engines)
    
    # Run small tournament
    tournament.run_round_robin(num_games=10)
    
    # Print final ratings
    print("\nFinal Ratings:")
    for engine in engines:
        print(f"{engine.name}: {engine.rating:.0f}")

