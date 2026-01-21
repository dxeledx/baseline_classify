"""
Comprehensive Test Suite for AlphaZero

Tests all components to ensure everything works correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

def test_imports():
    """Test that all modules can be imported"""
    print("\n" + "="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    
    try:
        from src.core import AlphaZeroNetwork, MCTS, Game, SelfPlay
        print("✓ Core modules")
        
        from src.games import ChessGame, Connect4Game, TicTacToeGame
        print("✓ Game modules")
        
        from src.training import Coach, TrainingArgs
        print("✓ Training modules")
        
        from src.evaluation import ELOCalculator, Tournament, load_model
        print("✓ Evaluation modules")
        
        print("\n✅ All imports successful!")
        return True
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_chess_game():
    """Test Chess game implementation"""
    print("\n" + "="*70)
    print("TEST 2: Chess Game")
    print("="*70)
    
    try:
        from src.games import ChessGame
        
        game = ChessGame()
        board = game.get_init_board()
        
        print(f"✓ Board initialized")
        print(f"  Size: {game.get_board_size()}")
        print(f"  Actions: {game.get_action_size()}")
        print(f"  Planes: {game.get_num_planes()}")
        
        # Test valid moves
        valids = game.get_valid_moves(board, 1)
        num_moves = np.sum(valids)
        print(f"✓ Valid moves: {num_moves}")
        
        # Test board encoding
        encoding = game.get_board_for_network(board, 1)
        print(f"✓ Board encoding shape: {encoding.shape}")
        
        # Test a move
        import chess
        move = chess.Move.from_uci("e2e4")
        action = game._move_to_action(move)
        board_new, player = game.get_next_state(board, 1, action)
        print(f"✓ Move executed (e2e4)")
        
        print("\n✅ Chess game tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Chess test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_connect4_game():
    """Test Connect4 game implementation"""
    print("\n" + "="*70)
    print("TEST 3: Connect4 Game")
    print("="*70)
    
    try:
        from src.games import Connect4Game
        
        game = Connect4Game()
        board = game.get_init_board()
        
        print(f"✓ Board initialized: {game.get_board_size()}")
        
        # Test some moves
        board, player = game.get_next_state(board, 1, 3)
        print(f"✓ Move executed")
        
        # Test valid moves
        valids = game.get_valid_moves(board, player)
        print(f"✓ Valid moves: {np.sum(valids)}")
        
        print("\n✅ Connect4 game tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Connect4 test failed: {e}")
        return False


def test_tictactoe_game():
    """Test Tic-Tac-Toe game implementation"""
    print("\n" + "="*70)
    print("TEST 4: Tic-Tac-Toe Game")
    print("="*70)
    
    try:
        from src.games import TicTacToeGame
        
        game = TicTacToeGame()
        board = game.get_init_board()
        
        print(f"✓ Board initialized: {game.get_board_size()}")
        
        # Play a few moves
        board, player = game.get_next_state(board, 1, 4)  # Center
        board, player = game.get_next_state(board, player, 0)  # Top-left
        
        print(f"✓ Moves executed")
        print(f"✓ Game ended: {game.get_game_ended(board, player)}")
        
        print("\n✅ Tic-Tac-Toe game tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Tic-Tac-Toe test failed: {e}")
        return False


def test_neural_network():
    """Test neural network"""
    print("\n" + "="*70)
    print("TEST 5: Neural Network")
    print("="*70)
    
    try:
        from src.core import AlphaZeroNetwork
        from src.games import ChessGame
        
        game = ChessGame()
        network = AlphaZeroNetwork(game, num_channels=64, num_res_blocks=2)
        
        print(f"✓ Network created")
        
        # Test forward pass
        board = game.get_init_board()
        board_input = game.get_board_for_network(board, 1)
        board_tensor = torch.FloatTensor(board_input).unsqueeze(0)
        
        policy, value = network(board_tensor)
        
        print(f"✓ Forward pass successful")
        print(f"  Policy shape: {policy.shape}")
        print(f"  Value shape: {value.shape}")
        
        # Test predict method
        policy_np, value_np = network.predict(board_input)
        print(f"✓ Predict method works")
        
        print("\n✅ Neural network tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Neural network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcts():
    """Test MCTS"""
    print("\n" + "="*70)
    print("TEST 6: MCTS")
    print("="*70)
    
    try:
        from src.core import MCTS, DummyArgs, AlphaZeroNetwork
        from src.games import TicTacToeGame
        
        game = TicTacToeGame()
        network = AlphaZeroNetwork(game, num_channels=32, num_res_blocks=2)
        network.eval()
        
        args = DummyArgs(num_mcts_sims=10)
        mcts = MCTS(game, network, args)
        
        board = game.get_init_board()
        board_input = game.get_board_for_network(board, 1)
        
        pi = mcts.get_action_prob(board_input, temp=1)
        
        print(f"✓ MCTS search completed")
        print(f"  Policy sum: {np.sum(pi):.4f}")
        print(f"  Best move: {np.argmax(pi)}")
        
        print("\n✅ MCTS tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ MCTS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_elo_system():
    """Test ELO rating system"""
    print("\n" + "="*70)
    print("TEST 7: ELO Rating System")
    print("="*70)
    
    try:
        from src.evaluation import ELOCalculator, RandomEngine, MaterialEngine
        
        elo_calc = ELOCalculator()
        
        # Test ELO calculation
        expected = elo_calc.expected_score(1500, 1500)
        print(f"✓ Expected score (equal players): {expected:.2f}")
        
        # Test rating update
        new_rating = elo_calc.update_rating(1500, 0.5, 1.0)
        print(f"✓ Rating update: 1500 → {new_rating:.0f}")
        
        # Test engines
        random_engine = RandomEngine("Test-Random", 800)
        print(f"✓ Random engine created: {random_engine}")
        
        material_engine = MaterialEngine("Test-Material", 1200)
        print(f"✓ Material engine created: {material_engine}")
        
        print("\n✅ ELO system tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ ELO system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """Test that all files are properly organized"""
    print("\n" + "="*70)
    print("TEST 8: File Structure")
    print("="*70)
    
    required_dirs = [
        'src/core',
        'src/games',
        'src/training',
        'src/evaluation',
        'tests',
        'scripts',
        'docs'
    ]
    
    required_files = [
        'pyproject.toml',
        'src/__init__.py',
        'src/core/__init__.py',
        'src/games/__init__.py',
        'src/training/__init__.py',
        'src/evaluation/__init__.py',
    ]
    
    project_root = os.path.join(os.path.dirname(__file__), '..')
    
    all_good = True
    for dir_path in required_dirs:
        full_path = os.path.join(project_root, dir_path)
        if os.path.exists(full_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"❌ Missing: {dir_path}/")
            all_good = False
    
    for file_path in required_files:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_good = False
    
    if all_good:
        print("\n✅ File structure is correct!")
    else:
        print("\n⚠️  Some files/directories are missing")
    
    return all_good


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("ALPHAZERO - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Module Imports", test_imports),
        ("Chess Game", test_chess_game),
        ("Connect4 Game", test_connect4_game),
        ("Tic-Tac-Toe Game", test_tictactoe_game),
        ("Neural Network", test_neural_network),
        ("MCTS", test_mcts),
        ("ELO Rating System", test_elo_system),
        ("File Structure", test_file_structure),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:.<50} {status}")
    
    print("="*70)
    print(f"Result: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your AlphaZero implementation is ready!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

