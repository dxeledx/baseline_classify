#!/bin/bash
# Full training example for Connect4

echo "Starting AlphaZero training on Connect4"
echo "This will take several hours depending on your hardware"
echo ""

python main.py \
    --game connect4 \
    --iterations 100 \
    --episodes 100 \
    --mcts_sims 50 \
    --epochs 10 \
    --batch_size 64 \
    --lr 0.001 \
    --num_channels 256 \
    --num_res_blocks 10 \
    --arena_games 40 \
    --checkpoint_dir ./checkpoints_connect4 \
    --cuda

echo ""
echo "Training complete!"
echo "Play against the model:"
echo "python play.py --game connect4 --model checkpoints_connect4/best.pth --human_first --mcts_sims 100"
echo ""
echo "Evaluate the model:"
echo "python play.py --game connect4 --model checkpoints_connect4/best.pth --evaluate --eval_games 100"

