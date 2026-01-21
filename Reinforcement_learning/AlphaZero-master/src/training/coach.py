"""
Coach - Training Pipeline for AlphaZero

This module implements the main training loop that orchestrates:
1. Self-play data generation
2. Neural network training
3. Model evaluation and selection
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
from collections import deque
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ..core.neural_network import AlphaZeroLoss
from ..core.self_play import SelfPlay, Arena
from ..core.mcts import MCTS, DummyArgs


class TrainDataset(Dataset):
    """PyTorch Dataset for training examples"""
    
    def __init__(self, examples):
        self.examples = examples
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        board, pi, v = self.examples[idx]
        return torch.FloatTensor(board), torch.FloatTensor(pi), torch.FloatTensor([v])


class Coach:
    """
    Main training pipeline for AlphaZero
    
    The coach manages the iterative training process:
    1. Generate self-play data with current model
    2. Train neural network on aggregated data
    3. Evaluate new model against previous best
    4. Keep new model if it wins enough games
    """
    
    def __init__(self, game, network, args):
        """
        Args:
            game: Game instance
            network: Neural network
            args: Training configuration
        """
        self.game = game
        self.network = network
        self.args = args
        
        # Training history
        self.train_examples_history = deque([], maxlen=args.num_iters_for_train_examples_history)
        
        # Best model
        self.best_network = None
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        
    def execute_episode(self):
        """Execute one episode of self-play"""
        self_play = SelfPlay(self.game, self.network, self.args)
        return self_play.execute_episode()
    
    def learn(self):
        """
        Main training loop
        
        Performs multiple iterations of:
        - Self-play
        - Training
        - Evaluation
        """
        for iteration in range(1, self.args.num_iterations + 1):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{self.args.num_iterations}")
            print(f"{'='*60}")
            
            # Generate self-play data
            iteration_train_examples = self._generate_self_play_data()
            
            # Add to history
            self.train_examples_history.append(iteration_train_examples)
            
            # Save iteration examples
            self._save_iteration_examples(iteration, iteration_train_examples)
            
            # Train network
            if len(self.train_examples_history) > 0:
                # Aggregate training examples from history
                train_examples = []
                for e in self.train_examples_history:
                    train_examples.extend(e)
                
                # Shuffle examples
                np.random.shuffle(train_examples)
                
                # Train
                self._train_network(train_examples)
                
                # Save checkpoint
                self._save_checkpoint(iteration, 'train')
            
            # Evaluate against previous best
            if iteration > 1:
                print("\nEvaluating new model against previous best...")
                self._evaluate_and_update_best(iteration)
            else:
                # First iteration: accept current model as best
                self._accept_new_model(iteration)
        
        print("\nTraining complete!")
    
    def _generate_self_play_data(self):
        """Generate self-play training data"""
        self.network.eval()
        
        self_play = SelfPlay(self.game, self.network, self.args)
        iteration_train_examples = self_play.generate_self_play_data(
            self.args.num_episodes
        )
        
        return iteration_train_examples
    
    def _train_network(self, train_examples):
        """
        Train the neural network
        
        Args:
            train_examples: List of (board, player, pi, v) tuples
        """
        print(f"\nTraining network on {len(train_examples)} examples...")
        
        self.network.train()
        
        # Setup optimizer
        optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        
        # Prepare data
        train_data = [(ex[0], ex[2], ex[3]) for ex in train_examples]
        dataset = TrainDataset(train_data)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Loss function
        criterion = AlphaZeroLoss()
        
        # Training loop
        for epoch in range(self.args.epochs):
            total_loss = 0
            total_policy_loss = 0
            total_value_loss = 0
            batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            for boards, pis, vs in pbar:
                # Move to device
                boards = boards.to(self.device)
                pis = pis.to(self.device)
                vs = vs.to(self.device).squeeze()
                
                # Forward pass
                policy_pred, value_pred = self.network(boards)
                
                # Calculate loss
                loss, policy_loss, value_loss = criterion(
                    policy_pred, pis, value_pred.squeeze(), vs
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                batches += 1
                
                pbar.set_postfix({
                    'loss': total_loss / batches,
                    'pi_loss': total_policy_loss / batches,
                    'v_loss': total_value_loss / batches
                })
            
            avg_loss = total_loss / batches
            avg_policy_loss = total_policy_loss / batches
            avg_value_loss = total_value_loss / batches
            
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                  f"Policy Loss={avg_policy_loss:.4f}, "
                  f"Value Loss={avg_value_loss:.4f}")
    
    def _evaluate_and_update_best(self, iteration):
        """
        Evaluate new model against best model
        
        Args:
            iteration: Current iteration number
        """
        # Load best model
        best_network = self._load_best_model()
        
        if best_network is None:
            print("No previous best model found. Accepting current model.")
            self._accept_new_model(iteration)
            return
        
        # Create player functions
        mcts_args = DummyArgs(
            num_mcts_sims=self.args.arena_mcts_sims,
            cpuct=self.args.cpuct
        )
        
        def new_player(board):
            mcts = MCTS(self.game, self.network, mcts_args)
            pi = mcts.get_action_prob(board, temp=0)
            return np.argmax(pi)
        
        def best_player(board):
            mcts = MCTS(self.game, best_network, mcts_args)
            pi = mcts.get_action_prob(board, temp=0)
            return np.argmax(pi)
        
        # Play arena games
        arena = Arena(new_player, best_player, self.game)
        new_wins, best_wins, draws = arena.play_games(
            self.args.arena_compare_games
        )
        
        print(f"\nArena Results:")
        print(f"New Model: {new_wins} wins")
        print(f"Best Model: {best_wins} wins")
        print(f"Draws: {draws}")
        
        # Accept new model if win rate is high enough
        win_rate = new_wins / (new_wins + best_wins) if (new_wins + best_wins) > 0 else 0
        print(f"Win Rate: {win_rate:.2%}")
        
        if win_rate >= self.args.update_threshold:
            print("New model is better! Accepting as best model.")
            self._accept_new_model(iteration)
        else:
            print("New model not good enough. Keeping previous best model.")
            # Reload best model
            self.network.load_state_dict(best_network.state_dict())
    
    def _accept_new_model(self, iteration):
        """Accept current model as best"""
        self._save_checkpoint(iteration, 'best')
        print(f"Model from iteration {iteration} accepted as best.")
    
    def _load_best_model(self):
        """Load the best model"""
        best_path = os.path.join(self.args.checkpoint_dir, 'best.pth')
        if not os.path.exists(best_path):
            return None
        
        # Create new network instance
        from neural_network import AlphaZeroNetwork
        best_network = AlphaZeroNetwork(
            self.game,
            num_channels=self.args.num_channels,
            num_res_blocks=self.args.num_res_blocks,
            dropout=self.args.dropout
        )
        best_network.to(self.device)
        
        # Load weights
        best_network.load_state_dict(torch.load(best_path, map_location=self.device))
        best_network.eval()
        
        return best_network
    
    def _save_checkpoint(self, iteration, name):
        """Save model checkpoint"""
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        
        filepath = os.path.join(
            self.args.checkpoint_dir,
            f'{name}.pth'
        )
        
        torch.save(self.network.state_dict(), filepath)
        print(f"Model saved: {filepath}")
    
    def _save_iteration_examples(self, iteration, examples):
        """Save training examples from iteration"""
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        
        filepath = os.path.join(
            self.args.checkpoint_dir,
            f'examples_iter_{iteration}.pkl'
        )
        
        with open(filepath, 'wb') as f:
            pickle.dump(examples, f)
        
        print(f"Training examples saved: {filepath}")


class TrainingArgs:
    """Training configuration arguments"""
    
    def __init__(self):
        # Iterations
        self.num_iterations = 100
        self.num_episodes = 100  # Self-play games per iteration
        self.num_iters_for_train_examples_history = 20
        
        # MCTS
        self.num_mcts_sims = 50  # MCTS simulations during self-play
        self.cpuct = 1.0
        self.temp_threshold = 15  # Moves before deterministic play
        
        # Training
        self.epochs = 10
        self.batch_size = 64
        self.lr = 0.001
        self.weight_decay = 1e-4
        
        # Network architecture
        self.num_channels = 256
        self.num_res_blocks = 10
        self.dropout = 0.3
        
        # Arena evaluation
        self.arena_compare_games = 40
        self.arena_mcts_sims = 50
        self.update_threshold = 0.55  # Win rate needed to accept new model
        
        # Checkpoints
        self.checkpoint_dir = './checkpoints'
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

