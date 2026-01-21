"""
Test script to verify AlphaZero installation and setup

Run this script to ensure all dependencies are installed correctly
and the system is ready for training.
"""

import sys


def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print("✓ tqdm")
    except ImportError as e:
        print(f"✗ tqdm import failed: {e}")
        return False
    
    return True


def test_modules():
    """Test that all project modules can be imported"""
    print("\nTesting project modules...")
    
    modules = [
        'game',
        'connect4',
        'neural_network',
        'mcts',
        'self_play',
        'coach',
        'utils',
        'main',
        'play'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module} import failed: {e}")
            return False
    
    return True


def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from connect4 import Connect4Game, TicTacToeGame
        from neural_network import AlphaZeroNetwork
        import torch
        
        # Test Tic-Tac-Toe
        print("Testing Tic-Tac-Toe game...")
        game = TicTacToeGame()
        board = game.get_init_board()
        assert board.shape == (3, 3), "Board shape incorrect"
        print("✓ Tic-Tac-Toe game initialization")
        
        # Test Connect4
        print("Testing Connect4 game...")
        game = Connect4Game()
        board = game.get_init_board()
        assert board.shape == (6, 7), "Board shape incorrect"
        print("✓ Connect4 game initialization")
        
        # Test neural network
        print("Testing neural network...")
        network = AlphaZeroNetwork(game, num_channels=64, num_res_blocks=2)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network.to(device)
        
        # Test forward pass
        test_board = game.get_board_for_network(board, 1)
        test_input = torch.FloatTensor(test_board).unsqueeze(0).to(device)
        policy, value = network(test_input)
        
        assert policy.shape[1] == game.get_action_size(), "Policy shape incorrect"
        assert value.shape == (1, 1), "Value shape incorrect"
        print("✓ Neural network forward pass")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcts():
    """Test MCTS functionality"""
    print("\nTesting MCTS...")
    
    try:
        from connect4 import TicTacToeGame
        from neural_network import AlphaZeroNetwork
        from mcts import MCTS, DummyArgs
        
        game = TicTacToeGame()
        network = AlphaZeroNetwork(game, num_channels=64, num_res_blocks=2)
        network.eval()
        
        args = DummyArgs(num_mcts_sims=10)
        mcts = MCTS(game, network, args)
        
        board = game.get_init_board()
        pi = mcts.get_action_prob(board, temp=1)
        
        assert len(pi) == game.get_action_size(), "Policy vector size incorrect"
        assert abs(sum(pi) - 1.0) < 1e-5, "Policy doesn't sum to 1"
        print("✓ MCTS search")
        
        return True
        
    except Exception as e:
        print(f"✗ MCTS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("AlphaZero Installation Test")
    print("="*60)
    print()
    
    print(f"Python version: {sys.version}")
    print()
    
    tests = [
        ("Import Test", test_imports),
        ("Module Test", test_modules),
        ("Functionality Test", test_basic_functionality),
        ("MCTS Test", test_mcts),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! You're ready to start training.")
        print("\nNext steps:")
        print("1. Run quick test: ./example_quick_train.sh")
        print("2. Or start training: python main.py --game tictactoe")
        print("3. See GETTING_STARTED.md for more information")
        return 0
    else:
        print("⚠️  Some tests failed. Please check your installation.")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check Python version (3.8+ required)")
        print("3. Verify PyTorch installation")
        return 1


if __name__ == '__main__':
    sys.exit(main())

