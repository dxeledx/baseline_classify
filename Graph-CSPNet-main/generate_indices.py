"""
Generate cross-validation indices for BCIC-IV-2a dataset
"""
import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
import os

def get_data_info(subject, PATH):
    """Get the number of trials and labels from BCIC-IV-2a dataset"""
    NO_channels = 22
    NO_tests = 6 * 48  # Expected number of trials
    Window_Length = 7 * 250
    
    class_return = np.zeros(NO_tests)
    data_return = np.zeros((NO_tests, NO_channels, Window_Length))
    
    NO_valid_trial = 0
    
    # Load training data
    a = sio.loadmat(PATH + 'A0' + str(subject) + 'T.mat')
    a_data = a['data']
    
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        
        for trial in range(0, a_trial.size):
            class_return[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial += 1
    
    # Return actual number of trials and labels
    return NO_valid_trial, class_return[:NO_valid_trial] - 1


def generate_cv_indices(subject, PATH='dataset/', output_folder='index/BCIC_index/', session_no=1, n_splits=10, random_state=42):
    """
    Generate stratified k-fold cross-validation indices for a subject
    
    Args:
        subject: Subject number (1-9)
        PATH: Path to dataset folder
        output_folder: Path to save indices
        session_no: Session number (default: 1)
        n_splits: Number of folds (default: 10)
        random_state: Random seed for reproducibility
    """
    
    # Get data info
    n_trials, labels = get_data_info(subject, PATH)
    
    print(f"Subject {subject}: {n_trials} trials")
    print(f"  Class distribution: {np.bincount(labels.astype(int))}")
    
    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    train_indices = []
    test_indices = []
    
    # Generate indices for each fold
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n_trials), labels)):
        train_indices.append(train_idx)
        test_indices.append(test_idx)
        print(f"  Fold {fold_idx + 1}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # Convert to numpy arrays
    train_indices = np.array(train_indices, dtype=object)
    test_indices = np.array(test_indices, dtype=object)
    
    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Save indices
    train_file = f'{output_folder}sess{session_no}_sub{subject}_train_index.npy'
    test_file = f'{output_folder}sess{session_no}_sub{subject}_test_index.npy'
    
    np.save(train_file, train_indices)
    np.save(test_file, test_indices)
    
    print(f"  Saved to {train_file} and {test_file}\n")
    
    return train_indices, test_indices


if __name__ == '__main__':
    # Check if dataset exists
    if not os.path.exists('dataset/'):
        print("Error: dataset/ folder not found!")
        exit(1)
    
    # Generate indices for all 9 subjects
    print("=" * 60)
    print("Generating cross-validation indices for BCIC-IV-2a dataset")
    print("=" * 60)
    print()
    
    for subject in range(1, 10):
        try:
            generate_cv_indices(subject)
        except FileNotFoundError:
            print(f"Warning: Data file for subject {subject} not found. Skipping...")
        except Exception as e:
            print(f"Error processing subject {subject}: {e}")
    
    print("=" * 60)
    print("Index generation completed!")
    print("=" * 60)
