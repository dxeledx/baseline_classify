"""
Chess Game Implementation for AlphaZero

This module implements chess using the python-chess library,
adapted to work with the AlphaZero training system.
"""

import numpy as np
import chess
from ..core.game import Game


class ChessGame(Game):
    """
    Chess game implementation using python-chess library
    
    Board representation:
    - 8x8 board
    - 4672 possible moves (from_square * to_square + promotions)
    - 119 input planes for neural network
    """
    
    def __init__(self):
        self.board_size = 8
        self.action_size = 4672  # 64*64 + 64*3 (promotions) simplified to under promotions
        
    def get_init_board(self):
        """
        Returns initial chess board
        
        Returns:
            board: chess.Board object
        """
        return chess.Board()
    
    def get_board_size(self):
        """Board is 8x8"""
        return (8, 8)
    
    def get_action_size(self):
        """
        Total possible moves in chess (simplified)
        64 from_squares * 64 to_squares + promotions = ~4672
        """
        return self.action_size
    
    def get_num_planes(self):
        """
        Number of input planes for neural network:
        - 12 planes for pieces (6 piece types × 2 colors)
        - 2 planes for repetition counts
        - 1 plane for en passant
        - 4 planes for castling rights
        - 1 plane for current player
        - 1 plane for move count
        Total: 21 planes (simplified version)
        """
        return 21
    
    def get_next_state(self, board, player, action):
        """
        Apply action to board
        
        Args:
            board: chess.Board object
            player: 1 for white, -1 for black
            action: Move index
            
        Returns:
            next_board: New board state
            next_player: -player (switch turns)
        """
        b = board.copy()
        move = self._action_to_move(b, action)
        
        if move in b.legal_moves:
            b.push(move)
        
        return b, -player
    
    def get_valid_moves(self, board, player):
        """
        Get valid moves for current player
        
        Args:
            board: chess.Board object
            player: Current player (1 or -1)
            
        Returns:
            valid_moves: Binary array of valid moves
        """
        valids = np.zeros(self.action_size, dtype=np.int8)
        
        for move in board.legal_moves:
            action = self._move_to_action(move)
            if 0 <= action < self.action_size:
                valids[action] = 1
        
        return valids
    
    def get_game_ended(self, board, player):
        """
        Check if game has ended
        
        Args:
            board: chess.Board object
            player: Current player
            
        Returns:
            0: Game continues
            1: Player won
            -1: Player lost
            1e-4: Draw
        """
        if not board.is_game_over():
            return 0
        
        # Checkmate
        if board.is_checkmate():
            # If it's current player's turn and checkmate, they lost
            return -1
        
        # Draw (stalemate, insufficient material, etc.)
        return 1e-4
    
    def get_canonical_form(self, board, player):
        """
        Get board from perspective of player
        
        For chess, if player is -1 (black), we need to flip the board
        
        Args:
            board: chess.Board object
            player: Current player
            
        Returns:
            canonical_board: Board from player's perspective
        """
        if player == 1:
            return board.copy()
        else:
            # For black, flip the board
            flipped_board = board.copy()
            flipped_board = flipped_board.mirror()
            return flipped_board
    
    def get_symmetries(self, board, pi):
        """
        Chess has no symmetries (unlike Go or Connect4)
        
        Args:
            board: Board state
            pi: Policy vector
            
        Returns:
            symmetries: List with only original (board, pi)
        """
        return [(board, pi)]
    
    def string_representation(self, board):
        """
        Convert board to string for hashing
        
        Args:
            board: chess.Board object
            
        Returns:
            board_string: FEN string representation
        """
        return board.fen()
    
    def display(self, board):
        """
        Print board in human-readable format
        
        Args:
            board: chess.Board object
        """
        print("\n" + str(board))
        print(f"FEN: {board.fen()}")
        print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
        if board.is_check():
            print("CHECK!")
        print()
    
    def get_board_for_network(self, board, player):
        """
        Convert chess board to neural network input format
        
        Creates a 21-plane representation:
        Planes 0-5: White pieces (P, N, B, R, Q, K)
        Planes 6-11: Black pieces (P, N, B, R, Q, K)
        Plane 12-13: Repetition counters
        Plane 14: En passant square
        Plane 15-18: Castling rights (WK, WQ, BK, BQ)
        Plane 19: Current player (all 1s for white, all 0s for black)
        Plane 20: Move count normalized
        
        Args:
            board: chess.Board object
            player: Current player (1 or -1)
            
        Returns:
            planes: numpy array of shape (21, 8, 8)
        """
        planes = np.zeros((21, 8, 8), dtype=np.float32)
        
        # Get canonical board (from current player's perspective)
        canonical_board = self.get_canonical_form(board, player)
        
        # Piece planes (0-11)
        piece_map = canonical_board.piece_map()
        for square, piece in piece_map.items():
            rank = square // 8
            file = square % 8
            
            piece_type = piece.piece_type - 1  # 0-5 for P,N,B,R,Q,K
            color_offset = 0 if piece.color == chess.WHITE else 6
            plane_idx = color_offset + piece_type
            
            planes[plane_idx, rank, file] = 1.0
        
        # Repetition counters (12-13)
        # Simplified: just mark if position has been seen before
        if canonical_board.is_repetition(2):
            planes[12, :, :] = 1.0
        if canonical_board.is_repetition(3):
            planes[13, :, :] = 1.0
        
        # En passant (14)
        if canonical_board.ep_square is not None:
            ep_rank = canonical_board.ep_square // 8
            ep_file = canonical_board.ep_square % 8
            planes[14, ep_rank, ep_file] = 1.0
        
        # Castling rights (15-18)
        if canonical_board.has_kingside_castling_rights(chess.WHITE):
            planes[15, :, :] = 1.0
        if canonical_board.has_queenside_castling_rights(chess.WHITE):
            planes[16, :, :] = 1.0
        if canonical_board.has_kingside_castling_rights(chess.BLACK):
            planes[17, :, :] = 1.0
        if canonical_board.has_queenside_castling_rights(chess.BLACK):
            planes[18, :, :] = 1.0
        
        # Current player (19)
        if canonical_board.turn == chess.WHITE:
            planes[19, :, :] = 1.0
        
        # Move count normalized (20)
        move_count = canonical_board.fullmove_number
        planes[20, :, :] = min(move_count / 100.0, 1.0)
        
        return planes
    
    def _move_to_action(self, move):
        """
        Convert chess.Move to action index
        
        Action encoding: from_square * 64 + to_square
        For promotions, add offset based on promotion piece
        
        Args:
            move: chess.Move object
            
        Returns:
            action: Integer action index
        """
        from_square = move.from_square
        to_square = move.to_square
        
        # Basic move encoding
        action = from_square * 64 + to_square
        
        # Handle promotions (add offset)
        if move.promotion is not None:
            # Q=5, R=4, B=3, N=2
            promotion_offset = 4096  # After all normal moves
            promotion_type = move.promotion - 2  # Normalize (N=0, B=1, R=2, Q=3)
            action = promotion_offset + from_square * 4 + promotion_type
        
        return action
    
    def _action_to_move(self, board, action):
        """
        Convert action index to chess.Move
        
        Args:
            board: chess.Board object
            action: Integer action index
            
        Returns:
            move: chess.Move object
        """
        if action >= 4096:
            # Promotion move
            action_offset = action - 4096
            from_square = action_offset // 4
            promotion_type = (action_offset % 4) + 2  # +2 because N=2, B=3, R=4, Q=5
            
            # Find which to_square makes sense for this pawn
            for move in board.legal_moves:
                if (move.from_square == from_square and 
                    move.promotion == promotion_type):
                    return move
            
            # Fallback: try queen promotion
            rank = 7 if board.turn == chess.WHITE else 0
            to_square = from_square % 8 + rank * 8
            return chess.Move(from_square, to_square, promotion=chess.QUEEN)
        else:
            # Normal move
            from_square = action // 64
            to_square = action % 64
            
            move = chess.Move(from_square, to_square)
            
            # Check if this is a legal move
            if move in board.legal_moves:
                return move
            
            # Check for promotions (default to queen)
            promo_move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
            if promo_move in board.legal_moves:
                return promo_move
            
            # Return the move anyway (will be filtered by legal moves)
            return move


def test_chess_game():
    """Test the chess implementation"""
    print("Testing Chess Implementation...")
    
    game = ChessGame()
    board = game.get_init_board()
    
    print("Initial board:")
    game.display(board)
    
    print(f"Board size: {game.get_board_size()}")
    print(f"Action size: {game.get_action_size()}")
    print(f"Number of planes: {game.get_num_planes()}")
    
    # Test valid moves
    valids = game.get_valid_moves(board, 1)
    print(f"Number of legal moves: {np.sum(valids)}")
    
    # Test board encoding
    encoding = game.get_board_for_network(board, 1)
    print(f"Board encoding shape: {encoding.shape}")
    
    # Play a few moves
    print("\nPlaying e2e4...")
    move = chess.Move.from_uci("e2e4")
    action = game._move_to_action(move)
    board, player = game.get_next_state(board, 1, action)
    game.display(board)
    
    print("Testing complete!")


if __name__ == '__main__':
    test_chess_game()

