"""
Connect4 Game Implementation

A simple implementation of Connect4 to demonstrate AlphaZero.
The game is played on a 6x7 board where players alternate dropping
pieces, and the first to get 4 in a row wins.
"""

import numpy as np
from ..core.game import Game


class Connect4Game(Game):
    """Connect4 game implementation"""
    
    def __init__(self, height=6, width=7, win_length=4):
        self.height = height
        self.width = width
        self.win_length = win_length
        
    def get_init_board(self):
        """Initialize empty board"""
        return np.zeros((self.height, self.width), dtype=np.int8)
    
    def get_board_size(self):
        """Board dimensions"""
        return (self.height, self.width)
    
    def get_action_size(self):
        """Number of columns (possible actions)"""
        return self.width
    
    def get_num_planes(self):
        """
        3 planes for input:
        - Player 1 pieces
        - Player -1 pieces
        - Current player indicator
        """
        return 3
    
    def get_next_state(self, board, player, action):
        """
        Drop a piece in the specified column
        """
        b = np.copy(board)
        
        # Find the lowest empty row in the column
        for row in range(self.height - 1, -1, -1):
            if b[row][action] == 0:
                b[row][action] = player
                break
        
        return b, -player
    
    def get_valid_moves(self, board, player):
        """
        Valid moves are columns that are not full
        """
        valid = np.zeros(self.width, dtype=np.int8)
        for col in range(self.width):
            if board[0][col] == 0:  # Top row is empty
                valid[col] = 1
        return valid
    
    def get_game_ended(self, board, player):
        """
        Check if game has ended and return result
        
        Returns:
            0: Game continues
            1: Player won
            -1: Player lost
            1e-4: Draw
        """
        # Check horizontal
        for row in range(self.height):
            for col in range(self.width - self.win_length + 1):
                window = board[row, col:col + self.win_length]
                if np.all(window == player):
                    return 1
                if np.all(window == -player):
                    return -1
        
        # Check vertical
        for row in range(self.height - self.win_length + 1):
            for col in range(self.width):
                window = board[row:row + self.win_length, col]
                if np.all(window == player):
                    return 1
                if np.all(window == -player):
                    return -1
        
        # Check diagonal (down-right)
        for row in range(self.height - self.win_length + 1):
            for col in range(self.width - self.win_length + 1):
                window = [board[row + i][col + i] for i in range(self.win_length)]
                if all(x == player for x in window):
                    return 1
                if all(x == -player for x in window):
                    return -1
        
        # Check diagonal (up-right)
        for row in range(self.win_length - 1, self.height):
            for col in range(self.width - self.win_length + 1):
                window = [board[row - i][col + i] for i in range(self.win_length)]
                if all(x == player for x in window):
                    return 1
                if all(x == -player for x in window):
                    return -1
        
        # Check for draw (board full)
        if not np.any(board == 0):
            return 1e-4
        
        # Game continues
        return 0
    
    def get_canonical_form(self, board, player):
        """
        Return board from perspective of player
        """
        return player * board
    
    def get_symmetries(self, board, pi):
        """
        Connect4 has one symmetry: horizontal flip
        
        Args:
            board: Board state
            pi: Policy vector
            
        Returns:
            symmetries: List of (board, pi) tuples
        """
        # Original
        symmetries = [(board, pi)]
        
        # Horizontal flip
        flipped_board = np.fliplr(board)
        flipped_pi = np.flip(pi)
        symmetries.append((flipped_board, flipped_pi))
        
        return symmetries
    
    def string_representation(self, board):
        """
        Convert board to string for hashing
        """
        return board.tobytes()
    
    def display(self, board):
        """
        Print board in human-readable format
        """
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print("\n  " + " ".join(str(i) for i in range(self.width)))
        for row in board:
            print("  " + " ".join(symbols[x] for x in row))
        print()
    
    def get_board_for_network(self, board, player):
        """
        Convert board to 3-plane representation for neural network
        
        Planes:
        0: Current player's pieces
        1: Opponent's pieces
        2: Current player indicator (all 1s)
        """
        planes = np.zeros((3, self.height, self.width), dtype=np.float32)
        
        # Player 1 pieces
        planes[0] = (board == player).astype(np.float32)
        
        # Player -1 pieces
        planes[1] = (board == -player).astype(np.float32)
        
        # Current player indicator
        planes[2] = np.ones((self.height, self.width), dtype=np.float32) if player == 1 else np.zeros((self.height, self.width), dtype=np.float32)
        
        return planes


class TicTacToeGame(Game):
    """
    Simple Tic-Tac-Toe implementation for faster testing
    """
    
    def __init__(self):
        self.size = 3
        
    def get_init_board(self):
        return np.zeros((self.size, self.size), dtype=np.int8)
    
    def get_board_size(self):
        return (self.size, self.size)
    
    def get_action_size(self):
        return self.size * self.size
    
    def get_num_planes(self):
        return 3
    
    def get_next_state(self, board, player, action):
        b = np.copy(board)
        row = action // self.size
        col = action % self.size
        b[row][col] = player
        return b, -player
    
    def get_valid_moves(self, board, player):
        valid = np.zeros(self.get_action_size(), dtype=np.int8)
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 0:
                    valid[i * self.size + j] = 1
        return valid
    
    def get_game_ended(self, board, player):
        # Check rows
        for i in range(self.size):
            if np.all(board[i, :] == player):
                return 1
            if np.all(board[i, :] == -player):
                return -1
        
        # Check columns
        for j in range(self.size):
            if np.all(board[:, j] == player):
                return 1
            if np.all(board[:, j] == -player):
                return -1
        
        # Check diagonals
        if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
            return 1
        if np.all(np.diag(board) == -player) or np.all(np.diag(np.fliplr(board)) == -player):
            return -1
        
        # Check draw
        if not np.any(board == 0):
            return 1e-4
        
        return 0
    
    def get_canonical_form(self, board, player):
        return player * board
    
    def get_symmetries(self, board, pi):
        """8 symmetries: 4 rotations × 2 (with/without flip)"""
        pi_board = pi.reshape(self.size, self.size)
        symmetries = []
        
        for i in range(4):
            for flip in [True, False]:
                new_board = np.rot90(board, i)
                new_pi = np.rot90(pi_board, i)
                
                if flip:
                    new_board = np.fliplr(new_board)
                    new_pi = np.fliplr(new_pi)
                
                symmetries.append((new_board, new_pi.flatten()))
        
        return symmetries
    
    def string_representation(self, board):
        return board.tobytes()
    
    def display(self, board):
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for row in board:
            print(" ".join(symbols[x] for x in row))
        print()
    
    def get_board_for_network(self, board, player):
        planes = np.zeros((3, self.size, self.size), dtype=np.float32)
        planes[0] = (board == player).astype(np.float32)
        planes[1] = (board == -player).astype(np.float32)
        planes[2] = np.ones((self.size, self.size), dtype=np.float32) if player == 1 else np.zeros((self.size, self.size), dtype=np.float32)
        return planes

