# Activation Dataset Examples

This directory contains examples demonstrating how to use the PyTorch dataset implementation for loading activation data.

## Background

The original implementation in `exp/activations.py` loaded all activation files into memory at once, which could cause memory issues with large datasets. The new implementation uses a PyTorch `Dataset` to load files on-demand, making it suitable for processing large datasets that don't fit in memory.

## Examples

- `activation_dataset_example.py`: Demonstrates basic usage of the `ActivationDataset` class and `DataLoader`.

## Usage

To run the examples:

```bash
python -m exp.examples.activation_dataset_example
```

## Key Features

1. **Memory Efficiency**: Files are loaded on-demand rather than all at once
2. **DataLoader Integration**: Works with PyTorch's `DataLoader` for parallel loading
3. **Transformation Support**: Custom transforms can be applied to loaded data
4. **Backward Compatibility**: The original API in `exp/activations.py` is maintained

## API Overview

### ActivationDataset

```python
from exp.activation_dataset import ActivationDataset

# Create dataset
dataset = ActivationDataset(
    device="cpu",
    activation_keys=["router_logits"],
    transform=None,
    preload_metadata=True,
)

# Get dataset length
print(f"Dataset contains {len(dataset)} files")

# Get top_k
top_k = dataset.get_top_k()

# Load item
item = dataset[0]
```

### DataLoader

```python
from exp.activation_dataset import create_activation_dataloader

# Create dataloader
dataloader, top_k = create_activation_dataloader(
    batch_size=4,
    device="cpu",
    activation_keys=["router_logits"],
    shuffle=True,
    num_workers=2,
)

# Process batches
for batch in dataloader:
    # Process batch
    router_logits = batch["router_logits"]
    tokens = batch["tokens"]
```

