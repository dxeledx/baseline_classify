#!/bin/bash
# Setup script using uv for fast Python package management

echo "=================================="
echo "AlphaZero Setup with UV"
echo "=================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "UV not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo ""
    echo "Please restart your terminal or run: source $HOME/.cargo/env"
    echo "Then run this script again."
    exit 1
fi

echo "✓ UV is installed"
echo "UV version: $(uv --version)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "Installing dependencies..."
uv pip install -e .

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To test the installation:"
echo "  uv run python test_installation.py"
echo ""
echo "To start training:"
echo "  uv run python main.py --game tictactoe --iterations 5 --episodes 10"
echo ""

