#!/bin/bash
# Quick training example for testing the pipeline

echo "Starting quick AlphaZero training on Tic-Tac-Toe"
echo "This is a fast test configuration to verify everything works"
echo ""

python main.py \
    --game tictactoe \
    --iterations 10 \
    --episodes 25 \
    --mcts_sims 25 \
    --epochs 5 \
    --batch_size 32 \
    --num_channels 128 \
    --num_res_blocks 5 \
    --arena_games 20 \
    --checkpoint_dir ./checkpoints_quick_test \
    --cuda

echo ""
echo "Training complete!"
echo "Test the model with:"
echo "python play.py --game tictactoe --model checkpoints_quick_test/best.pth --human_first"

