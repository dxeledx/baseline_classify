"""
Configuration presets for different training scenarios
"""


class QuickTestConfig:
    """Fast configuration for testing the pipeline"""
    
    def __init__(self):
        # Iterations
        self.num_iterations = 5
        self.num_episodes = 10
        self.num_iters_for_train_examples_history = 3
        
        # MCTS
        self.num_mcts_sims = 25
        self.cpuct = 1.0
        self.temp_threshold = 10
        
        # Training
        self.epochs = 5
        self.batch_size = 32
        self.lr = 0.001
        self.weight_decay = 1e-4
        
        # Network
        self.num_channels = 128
        self.num_res_blocks = 5
        self.dropout = 0.3
        
        # Arena
        self.arena_compare_games = 20
        self.arena_mcts_sims = 25
        self.update_threshold = 0.55
        
        # Checkpoints
        self.checkpoint_dir = './checkpoints_test'
        self.device = 'cuda'


class TicTacToeConfig:
    """Optimized configuration for Tic-Tac-Toe"""
    
    def __init__(self):
        # Iterations
        self.num_iterations = 30
        self.num_episodes = 50
        self.num_iters_for_train_examples_history = 10
        
        # MCTS
        self.num_mcts_sims = 50
        self.cpuct = 1.0
        self.temp_threshold = 8
        
        # Training
        self.epochs = 10
        self.batch_size = 64
        self.lr = 0.001
        self.weight_decay = 1e-4
        
        # Network (smaller for simple game)
        self.num_channels = 128
        self.num_res_blocks = 6
        self.dropout = 0.3
        
        # Arena
        self.arena_compare_games = 40
        self.arena_mcts_sims = 50
        self.update_threshold = 0.55
        
        # Checkpoints
        self.checkpoint_dir = './checkpoints_tictactoe'
        self.device = 'cuda'


class Connect4Config:
    """Optimized configuration for Connect4"""
    
    def __init__(self):
        # Iterations
        self.num_iterations = 100
        self.num_episodes = 100
        self.num_iters_for_train_examples_history = 20
        
        # MCTS
        self.num_mcts_sims = 80
        self.cpuct = 1.0
        self.temp_threshold = 15
        
        # Training
        self.epochs = 10
        self.batch_size = 64
        self.lr = 0.001
        self.weight_decay = 1e-4
        
        # Network
        self.num_channels = 256
        self.num_res_blocks = 10
        self.dropout = 0.3
        
        # Arena
        self.arena_compare_games = 40
        self.arena_mcts_sims = 80
        self.update_threshold = 0.55
        
        # Checkpoints
        self.checkpoint_dir = './checkpoints_connect4'
        self.device = 'cuda'


class ProductionConfig:
    """High-quality configuration for production-level training"""
    
    def __init__(self):
        # Iterations
        self.num_iterations = 200
        self.num_episodes = 200
        self.num_iters_for_train_examples_history = 30
        
        # MCTS (more simulations for better play)
        self.num_mcts_sims = 400
        self.cpuct = 1.0
        self.temp_threshold = 20
        
        # Training
        self.epochs = 15
        self.batch_size = 128
        self.lr = 0.0005
        self.weight_decay = 1e-4
        
        # Network (deeper and wider)
        self.num_channels = 512
        self.num_res_blocks = 19
        self.dropout = 0.3
        
        # Arena (more games for reliable evaluation)
        self.arena_compare_games = 100
        self.arena_mcts_sims = 400
        self.update_threshold = 0.55
        
        # Checkpoints
        self.checkpoint_dir = './checkpoints_production'
        self.device = 'cuda'

