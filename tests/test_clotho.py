#!/usr/bin/env python
"""Test script for Clotho dataset loading."""
import sys
import os
sys.path.append(os.getcwd())

from datasets.clotho.get_data import get_dataloaders

# Point to your clotho-dataset-master directory
clotho_path = '/home/hejinfeng/datasets/clotho-dataset-master'

print(f"Loading Clotho data from: {clotho_path}")

try:
    print("Creating train dataloader...")
    import time
    start = time.time()
    train_loader, valid_loader = get_dataloaders(
        path_to_clotho=clotho_path,
        batch_size=4,
        num_workers=0,  # WSL friendly
        shuffle_train=True
    )
    print(f"Dataloaders created in {time.time()-start:.1f}s")
    
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(valid_loader)}")
    
    # Test loading one batch
    for batch in train_loader:
        print(f"✓ Batch shapes: {[x.shape if hasattr(x, 'shape') else type(x) for x in batch]}")
        break
    
    print("Clotho dataset loaded successfully!")
    
except Exception as e:
    print(f"Error loading Clotho: {e}")
    import traceback
    traceback.print_exc()

