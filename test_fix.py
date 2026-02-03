import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

from data_processing.preprocess_edges import preprocess_cloud_to_edges

test_files = list(Path('data/split/train').rglob('*.npy'))
if not test_files:
    print("No test files found!")
    sys.exit(1)

test_file = test_files[0]
print(f"Testing with: {test_file}")

cloud = np.load(test_file)
print(f"Cloud shape: {cloud.shape}")

output_path = Path('test_output.npy')

preprocess_cloud_to_edges(
    test_file,
    output_path,
    radius=[0.4, 1.5],
    use_mp=True,
    verbose=True,
    n_jobs=-1
)

if output_path.exists():
    result = np.load(output_path)
    print(f"\nOutput shape: {result.shape}")
    print(f"Sample edges:\n{result[:5]}")
    
    # Check for zeros
    zero_cols = (result == 0).sum(axis=0)
    print(f"\nZeros per column: {zero_cols}")
    print(f"Total zeros: {(result == 0).sum()} / {result.size} ({100*(result == 0).sum()/result.size:.1f}%)")
    
    output_path.unlink()
else:
    print("No output generated!")
