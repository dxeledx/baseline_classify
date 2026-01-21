"""
Base Game Interface for AlphaZero

This module defines the interface that any game must implement
to work with the AlphaZero training system.
"""

from abc import ABC, abstractmethod
import numpy as np


class Game(ABC):
    """
    Abstract base class for games
    
    All games must implement these methods to work with AlphaZero
    """
    
    @abstractmethod
    def get_init_board(self):
        """
        Returns:
            board: Initial board configuration
        """
        pass
    
    @abstractmethod
    def get_board_size(self):
        """
        Returns:
            (x, y): Dimensions of the board
        """
        pass
    
    @abstractmethod
    def get_action_size(self):
        """
        Returns:
            action_size: Number of possible actions
        """
        pass
    
    @abstractmethod
    def get_next_state(self, board, player, action):
        """
        Apply action to board
        
        Args:
            board: Current board state
            player: Current player (1 or -1)
            action: Action to take
            
        Returns:
            next_board: Board after action
            next_player: Next player to move
        """
        pass
    
    @abstractmethod
    def get_valid_moves(self, board, player):
        """
        Get valid moves for current player
        
        Args:
            board: Current board state
            player: Current player
            
        Returns:
            valid_moves: Binary array of size action_size
        """
        pass
    
    @abstractmethod
    def get_game_ended(self, board, player):
        """
        Check if game has ended
        
        Args:
            board: Current board state
            player: Current player
            
        Returns:
            r: 0 if not ended, 1 if player won, -1 if player lost, 
               small value for draw (e.g., 1e-4)
        """
        pass
    
    @abstractmethod
    def get_canonical_form(self, board, player):
        """
        Get board from perspective of player
        
        Args:
            board: Current board state
            player: Current player
            
        Returns:
            canonical_board: Board from player's perspective
        """
        pass
    
    @abstractmethod
    def get_symmetries(self, board, pi):
        """
        Get all symmetries of board and policy
        
        Args:
            board: Board state
            pi: Policy vector
            
        Returns:
            symmetries: List of (board, pi) tuples
        """
        pass
    
    @abstractmethod
    def string_representation(self, board):
        """
        Convert board to string for hashing
        
        Args:
            board: Board state
            
        Returns:
            board_string: String representation
        """
        pass
    
    @abstractmethod
    def get_num_planes(self):
        """
        Returns:
            num_planes: Number of input planes for neural network
        """
        pass
    
    def display(self, board):
        """
        Print board in human-readable format (optional)
        
        Args:
            board: Board state
        """
        print(board)

