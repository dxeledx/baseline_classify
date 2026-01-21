"""
Monte Carlo Tree Search (MCTS) Implementation for AlphaZero

This module implements the MCTS algorithm as described in the AlphaZero paper.
MCTS guides the agent's decision-making by simulating potential future game states.
"""

import math
import numpy as np
from collections import defaultdict


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance
    
    The search maintains statistics for each state-action pair:
    - Q(s,a): average action value
    - N(s,a): visit count
    - P(s,a): prior probability from neural network
    """
    
    def __init__(self, game, network, args):
        """
        Args:
            game: Game instance
            network: Neural network for position evaluation
            args: Configuration arguments
        """
        self.game = game
        self.network = network
        self.args = args
        
        # Statistics for each state
        self.Qsa = {}  # Q values: (state, action) -> float
        self.Nsa = {}  # Visit counts: (state, action) -> int
        self.Ns = {}   # Total visits: state -> int
        self.Ps = {}   # Policy priors: state -> array
        
        self.Es = {}   # Terminal state cache: state -> game result
        self.Vs = {}   # Valid moves cache: state -> array
        
    def get_action_prob(self, canonical_board, temp=1):
        """
        Get action probabilities from MCTS search
        
        Args:
            canonical_board: Board from current player's perspective
            temp: Temperature parameter for exploration
                  temp -> 0: deterministic (argmax)
                  temp -> inf: uniform
                  
        Returns:
            probs: Probability distribution over actions
        """
        for _ in range(self.args.num_mcts_sims):
            self.search(canonical_board)
        
        s = self.game.string_representation(canonical_board)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.get_action_size())]
        
        if temp == 0:
            # Deterministic: choose most visited action
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)
            probs = np.zeros(len(counts))
            probs[best_action] = 1
            return probs
        
        # Temperature scaling
        counts = np.array(counts, dtype=np.float64)
        counts = counts ** (1. / temp)
        probs = counts / counts.sum()
        return probs
    
    def search(self, canonical_board):
        """
        Perform one iteration of MCTS
        
        This performs the four MCTS steps:
        1. Selection: traverse tree using UCB
        2. Expansion: add new node using neural network
        3. Simulation: evaluate position with neural network
        4. Backpropagation: update statistics along path
        
        Args:
            canonical_board: Current board state
            
        Returns:
            v: Value of the position from current player's perspective
        """
        s = self.game.string_representation(canonical_board)
        
        # Check if terminal state
        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(canonical_board, 1)
        if self.Es[s] != 0:
            # Terminal state
            return -self.Es[s]
        
        # Check if leaf node (not yet expanded)
        if s not in self.Ps:
            # Expansion: evaluate with neural network
            # Convert board to network input format
            if hasattr(self.game, 'get_board_for_network'):
                board_input = self.game.get_board_for_network(canonical_board, 1)
            else:
                board_input = canonical_board
            self.Ps[s], v = self.network.predict(board_input)
            
            # Mask invalid moves
            valids = self.game.get_valid_moves(canonical_board, 1)
            self.Ps[s] = self.Ps[s] * valids  # Mask invalid moves
            sum_ps = np.sum(self.Ps[s])
            if sum_ps > 0:
                self.Ps[s] /= sum_ps  # Normalize
            else:
                # All valid moves were masked, use uniform distribution
                print("Warning: all valid moves were masked, using uniform distribution")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v
        
        # Selection: pick action with highest UCB
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1
        
        for a in range(self.game.get_action_size()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)
                
                if u > cur_best:
                    cur_best = u
                    best_act = a
        
        a = best_act
        next_board, next_player = self.game.get_next_state(canonical_board, 1, a)
        next_board = self.game.get_canonical_form(next_board, next_player)
        
        # Recursion: continue search
        v = self.search(next_board)
        
        # Backpropagation: update statistics
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        
        self.Ns[s] += 1
        return -v


class MCTSArgs:
    """Configuration arguments for MCTS"""
    
    def __init__(self):
        self.num_mcts_sims = 800  # Number of MCTS simulations per move
        self.cpuct = 1.0          # Exploration constant for UCB


class DummyArgs:
    """Default arguments for MCTS"""
    
    def __init__(self, num_mcts_sims=50, cpuct=1.0):
        self.num_mcts_sims = num_mcts_sims
        self.cpuct = cpuct

