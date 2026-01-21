#!/bin/bash
# Quick Start Guide for Chess Training and ELO Evaluation

echo "=================================="
echo "AlphaZero Chess - Quick Start"
echo "=================================="
echo ""

# Activate environment
source .venv/bin/activate

# Option 1: Quick Test
echo "Option 1: Quick Test (10-15 minutes)"
echo "This trains a tiny model to verify everything works"
echo ""
echo "Command:"
echo "python train_chess.py --iterations 3 --episodes 5 --mcts_sims 25 --epochs 3 --num_channels 64 --num_res_blocks 3 --checkpoint_dir ./checkpoints_chess_quick"
echo ""
read -p "Run quick test? (y/n): " choice
if [ "$choice" = "y" ]; then
    python train_chess.py \
        --iterations 3 \
        --episodes 5 \
        --mcts_sims 25 \
        --epochs 3 \
        --batch_size 16 \
        --num_channels 64 \
        --num_res_blocks 3 \
        --checkpoint_dir ./checkpoints_chess_quick
    
    echo ""
    echo "Training complete! Now evaluating ELO..."
    echo ""
    
    python evaluate_elo.py \
        --model checkpoints_chess_quick/best.pth \
        --num_channels 64 \
        --num_res_blocks 3 \
        --games_per_opponent 10 \
        --quick
    
    echo ""
    echo "✅ Quick test complete!"
    echo "Your engine's ELO rating is shown above."
    exit 0
fi

# Option 2: Serious Training
echo ""
echo "Option 2: Serious Training (Several Hours to Days)"
echo "This trains a real chess engine"
echo ""
echo "Recommended settings:"
echo "  - 100 iterations"
echo "  - 100 episodes per iteration"
echo "  - 100 MCTS simulations"
echo "  - 256 channels, 19 residual blocks"
echo ""
read -p "Start serious training? (y/n): " choice
if [ "$choice" = "y" ]; then
    echo ""
    echo "Starting training... This will take a while."
    echo "You can stop anytime with Ctrl+C and resume later."
    echo ""
    
    python train_chess.py \
        --iterations 100 \
        --episodes 100 \
        --mcts_sims 100 \
        --epochs 10 \
        --batch_size 64 \
        --num_channels 256 \
        --num_res_blocks 19 \
        --checkpoint_dir ./checkpoints_chess \
        --cuda
    
    echo ""
    echo "Training complete! Now evaluating ELO..."
    echo ""
    
    python evaluate_elo.py \
        --model checkpoints_chess/best.pth \
        --mcts_sims 100 \
        --games_per_opponent 20
    
    exit 0
fi

echo ""
echo "No option selected. Here are the commands you can run manually:"
echo ""
echo "Quick test:"
echo "  python train_chess.py --iterations 3 --episodes 5 --mcts_sims 25 --checkpoint_dir ./checkpoints_chess_quick"
echo ""
echo "Full training:"
echo "  python train_chess.py --iterations 100 --episodes 100 --cuda"
echo ""
echo "Evaluate ELO:"
echo "  python evaluate_elo.py --model checkpoints_chess/best.pth"
echo ""
echo "Play against your engine:"
echo "  python play.py --game chess --model checkpoints_chess/best.pth"
echo ""
echo "See CHESS_GUIDE.md for complete documentation."

