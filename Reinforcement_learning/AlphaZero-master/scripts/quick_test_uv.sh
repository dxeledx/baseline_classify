#!/bin/bash
# Quick test script using uv

echo "=================================="
echo "AlphaZero Quick Test with UV"
echo "=================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: UV is not installed"
    echo "Please run: ./setup_uv.sh"
    exit 1
fi

# Run installation test
echo "Step 1: Testing installation..."
echo ""
uv run python test_installation.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Installation test failed!"
    echo "Please check the errors above."
    exit 1
fi

echo ""
echo "=================================="
echo "Step 2: Running quick training test..."
echo "Training Tic-Tac-Toe with minimal settings"
echo "This should take about 2-5 minutes..."
echo "=================================="
echo ""

uv run python main.py \
    --game tictactoe \
    --iterations 3 \
    --episodes 10 \
    --mcts_sims 20 \
    --epochs 3 \
    --batch_size 32 \
    --num_channels 64 \
    --num_res_blocks 3 \
    --checkpoint_dir ./checkpoints_uv_test

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✓ Quick test completed successfully!"
    echo "=================================="
    echo ""
    echo "Your AlphaZero implementation is working correctly!"
    echo ""
    echo "Next steps:"
    echo "1. Play against your test model:"
    echo "   uv run python play.py --game tictactoe --model checkpoints_uv_test/best.pth --num_channels 64 --num_res_blocks 3"
    echo ""
    echo "2. Start full training:"
    echo "   uv run python main.py --game connect4 --iterations 100 --episodes 100 --cuda"
    echo ""
    echo "3. See GETTING_STARTED.md for more examples"
else
    echo ""
    echo "Training test failed. Please check the errors above."
    exit 1
fi

