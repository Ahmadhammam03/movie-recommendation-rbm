# Models Directory

This directory stores trained RBM models and checkpoints.

## Directory Structure

```
models/
├── best_rbm_model.pth      # Best trained model
├── checkpoints/            # Training checkpoints
│   ├── rbm_best.pth       # Best checkpoint during training
│   ├── rbm_epoch_10.pth   # Checkpoint at epoch 10
│   └── ...
└── training_history.png    # Loss plot
```

## Model Files

### best_rbm_model.pth
The final trained RBM model containing:
- Model weights (W, a, b)
- Architecture info (nv, nh)
- Training history
- Dataset dimensions

### Checkpoints
Periodic saves during training for:
- Recovery from interruptions
- Model selection
- Training analysis

## Loading Models

```python
from src.trainer import RBMExperiment

# Load trained model
experiment = RBMExperiment(model_save_path="models/")
experiment.load_model("best_rbm_model.pth")
```

## Note
Model files are not included in the repository due to size. Run `main.py` to train and generate model files.