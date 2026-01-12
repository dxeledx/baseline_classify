"""
Check if all necessary files and dependencies are ready for Graph-CSPNet
"""
import os
import sys

def check_folders():
    """Check if all necessary folders exist"""
    folders = ['dataset', 'model_paras', 'results', 'index/BCIC_index', 'utils']
    print("[Folders] Checking folders...")
    all_exist = True
    for folder in folders:
        if os.path.exists(folder):
            print(f"  [OK] {folder}/")
        else:
            print(f"  [FAIL] {folder}/ - NOT FOUND")
            all_exist = False
    return all_exist

def check_dataset():
    """Check if dataset files exist"""
    print("\n[Dataset] Checking dataset files...")
    all_exist = True
    for sub in range(1, 10):
        train_file = f'dataset/A0{sub}T.mat'
        test_file = f'dataset/A0{sub}E.mat'
        if os.path.exists(train_file) and os.path.exists(test_file):
            print(f"  [OK] Subject {sub}: A0{sub}T.mat, A0{sub}E.mat")
        else:
            print(f"  [FAIL] Subject {sub}: Missing files")
            all_exist = False
    return all_exist

def check_indices():
    """Check if index files exist"""
    print("\n[Indices] Checking index files...")
    all_exist = True
    for sub in range(1, 10):
        train_idx = f'index/BCIC_index/sess1_sub{sub}_train_index.npy'
        test_idx = f'index/BCIC_index/sess1_sub{sub}_test_index.npy'
        if os.path.exists(train_idx) and os.path.exists(test_idx):
            print(f"  [OK] Subject {sub} indices")
        else:
            print(f"  [FAIL] Subject {sub} indices - NOT FOUND")
            all_exist = False
    return all_exist

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n[Dependencies] Checking dependencies...")
    required = ['numpy', 'pandas', 'scipy', 'torch', 'mne', 'pyriemann', 'moabb', 'sklearn']
    all_installed = True
    
    for package in required:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [FAIL] {package} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def main():
    print("=" * 60)
    print("  Graph-CSPNet Setup Checker")
    print("=" * 60)
    print()
    
    folders_ok = check_folders()
    dataset_ok = check_dataset()
    indices_ok = check_indices()
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 60)
    if folders_ok and dataset_ok and indices_ok and deps_ok:
        print("[SUCCESS] All checks passed! You're ready to train Graph-CSPNet!")
        print("\nTo start training:")
        print("  - Cross-validation: python Graph_CSPNet_BCIC_CV.py")
        print("  - Holdout: python Graph_CSPNet_BCIC_Holdout.py")
    else:
        print("[FAILED] Some checks failed. Please fix the issues above.")
        if not deps_ok:
            print("\nTo install missing dependencies:")
            print("  pip install -r requirements.txt")
            print("  pip install scikit-learn torch")
    print("=" * 60)

if __name__ == '__main__':
    main()
