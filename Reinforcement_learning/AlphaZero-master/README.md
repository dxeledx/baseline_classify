# AlphaZero Implementation

A comprehensive implementation of DeepMind's AlphaZero algorithm for reinforcement learning through self-play. This implementation supports training agents to play various two-player zero-sum games including Connect4 and Tic-Tac-Toe.

## 🎯 Overview

AlphaZero is a reinforcement learning algorithm that combines:
- **Deep Neural Networks** with policy and value heads
- **Monte Carlo Tree Search (MCTS)** for move selection
- **Self-play** for training data generation
- **Iterative improvement** through competition

This implementation follows the architecture described in the [AlphaZero paper](https://arxiv.org/abs/1712.01815) by DeepMind.

## 🏗️ Architecture

### Key Components

1. **Neural Network** (`neural_network.py`)
   - Convolutional architecture with residual blocks
   - Policy head: outputs move probabilities
   - Value head: evaluates position quality

2. **MCTS** (`mcts.py`)
   - Guided tree search using neural network priors
   - Balances exploration and exploitation
   - UCB-based action selection

3. **Self-Play** (`self_play.py`)
   - Generates training data through self-play
   - Arena for model evaluation
   - Data augmentation through symmetries

4. **Training Pipeline** (`coach.py`)
   - Orchestrates the training loop
   - Manages model checkpoints
   - Evaluates and selects best models

5. **Games** (`game.py`, `connect4.py`)
   - Abstract game interface
   - Connect4 implementation (6x7 board)
   - Tic-Tac-Toe implementation (3x3 board)

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- tqdm

### Setup with UV (Recommended - 10x Faster!)

[UV](https://astral.sh/uv) is a fast Python package installer. Installation takes ~7 seconds vs ~50 seconds with pip.

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup everything automatically
./setup_uv.sh

# Activate virtual environment
source .venv/bin/activate

# Test installation
python test_installation.py
```

### Setup with pip (Traditional Method)

```bash
# Install dependencies
pip install -r requirements.txt

# For GPU support (optional but recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**✅ The implementation has been fully tested with UV and all tests pass!** See `UV_SETUP.md` for details.

## 🚀 Quick Start

### Training a Model

Train on Connect4 (default):

```bash
python main.py --game connect4 --iterations 100 --episodes 100
```

Train on Tic-Tac-Toe:

```bash
python main.py --game tictactoe --iterations 50 --episodes 50
```

### Training Parameters

```bash
python main.py \
    --game connect4 \
    --iterations 100 \        # Number of training iterations
    --episodes 100 \          # Self-play games per iteration
    --mcts_sims 50 \         # MCTS simulations per move
    --epochs 10 \            # Training epochs per iteration
    --batch_size 64 \        # Batch size for training
    --lr 0.001 \             # Learning rate
    --num_channels 256 \     # CNN channels
    --num_res_blocks 10 \    # Number of residual blocks
    --arena_games 40 \       # Games for model evaluation
    --checkpoint_dir ./checkpoints \
    --cuda                    # Use GPU if available
```

### Playing Against the Model

After training, play against your model:

```bash
python play.py \
    --game connect4 \
    --model checkpoints/best.pth \
    --human_first \
    --mcts_sims 100
```

### Evaluating the Model

Test your model against a random player:

```bash
python play.py \
    --game connect4 \
    --model checkpoints/best.pth \
    --evaluate \
    --eval_games 100
```

## 📊 Training Process

The training loop consists of:

1. **Self-Play Generation**
   - The current model plays games against itself
   - Each move uses MCTS to select actions
   - Training examples stored: (state, policy, outcome)

2. **Neural Network Training**
   - Network trained on aggregated self-play data
   - Minimizes combined loss:
     - Policy loss: cross-entropy with MCTS policy
     - Value loss: MSE with game outcome

3. **Model Evaluation**
   - New model plays against previous best
   - New model accepted if win rate > threshold (default 55%)
   - Best model saved for next iteration

4. **Repeat**
   - Process continues for specified iterations
   - Model progressively improves through competition

## 🎮 Game Interface

To add a new game, implement the `Game` abstract class:

```python
from game import Game

class MyGame(Game):
    def get_init_board(self):
        # Return initial board state
        pass
    
    def get_board_size(self):
        # Return (height, width)
        pass
    
    def get_action_size(self):
        # Return number of possible actions
        pass
    
    def get_next_state(self, board, player, action):
        # Return (new_board, next_player)
        pass
    
    # ... implement other required methods
```

See `connect4.py` for a complete example.

## 📁 Project Structure

```
alpha-zero/
├── main.py              # Training script
├── play.py              # Play against trained model
├── neural_network.py    # Neural network architecture
├── mcts.py              # Monte Carlo Tree Search
├── self_play.py         # Self-play and arena
├── coach.py             # Training pipeline
├── game.py              # Abstract game interface
├── connect4.py          # Game implementations
├── utils.py             # Utility functions
├── requirements.txt     # Dependencies
├── README.md           # This file
└── checkpoints/        # Saved models (created during training)
```

## 🔧 Configuration

### MCTS Parameters

- `num_mcts_sims`: Number of simulations per move (higher = stronger but slower)
- `cpuct`: Exploration constant for UCB (typically 1.0)
- `temp_threshold`: Moves before switching to deterministic play

### Network Architecture

- `num_channels`: CNN channels (default 256)
- `num_res_blocks`: Residual blocks (default 10-19)
- `dropout`: Dropout rate for regularization

### Training Hyperparameters

- `lr`: Learning rate (default 0.001)
- `batch_size`: Training batch size
- `epochs`: Epochs per iteration
- `num_iters_for_train_examples_history`: History buffer size

## 💡 Tips for Training

1. **Start Small**: Begin with Tic-Tac-Toe to verify your setup works
2. **GPU Recommended**: Training on CPU is very slow
3. **Adjust Parameters**: 
   - Fewer MCTS sims during training speeds things up
   - More sims during evaluation for better play
4. **Monitor Progress**: Check arena win rates to ensure improvement
5. **Be Patient**: Good models require many iterations (50-100+)

## 📈 Expected Results

### Tic-Tac-Toe
- Training time: ~1-2 hours (CPU), ~10-20 min (GPU)
- After 20-30 iterations: Should play perfectly or near-perfectly

### Connect4
- Training time: Several hours to days depending on hardware
- After 50-100 iterations: Should beat random player >95% of the time
- Strong play requires 100+ iterations with good hyperparameters

## 🐛 Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `num_channels` or `num_res_blocks`
- Reduce `num_episodes`

### Slow Training
- Reduce `num_mcts_sims` during self-play
- Use GPU (`--cuda` flag)
- Reduce `num_res_blocks`

### Model Not Improving
- Increase `num_episodes` for more diverse training data
- Adjust learning rate
- Check arena evaluation threshold
- Ensure enough training iterations

## 📚 References

- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) - AlphaZero paper
- [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270) - AlphaGo Zero paper
- [A Simple Alpha(Go) Zero Tutorial](https://web.stanford.edu/~surag/posts/alphazero.html)

## 🤝 Contributing

This is an educational implementation. Feel free to:
- Add new games
- Improve performance
- Add features (e.g., better visualization, distributed training)
- Fix bugs

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

Based on research by DeepMind and inspired by various open-source implementations including:
- [alpha-zero-general](https://github.com/suragnair/alpha-zero-general)
- [AlphaZero.jl](https://github.com/jonathan-laurent/AlphaZero.jl)
- [AlphaZero-Edu](https://github.com/StarLight1212/AlphaZero_Edu)

---

**Happy Training! 🎯🤖**

