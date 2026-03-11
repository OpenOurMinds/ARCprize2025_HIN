
## Direct Files Referenced in the Script

### 1. **Script Files**
- `scripts/production_24h_run.sh` - The main script being executed

### 2. **Python Modules Called**
- `src/orchestrate_training.py` - Main orchestrator module (called with `python -m src.orchestrate_training`)

### 3. **Dataset Files**
- `intermediate_data/prepared_dataset_arc_2024.pth`
- `intermediate_data/prepared_dataset_arc_2025.pth` 
- `intermediate_data/prepared_dataset_barc.pth`
- `intermediate_data/prepared_dataset_rearc.pth`

### 4. **Dependencies**
- `requirements.txt` - Python package dependencies

## Files Called by the Orchestrator

### 5. **Training Module**
- `src/train.py` - Called by orchestrator for each training iteration

### 6. **Rollout Generation Module**
- `src/generate_rollout.py` - Called by orchestrator for trajectory generation

### 7. **Core Model and Utilities**
- `src/model.py` - Transformer model definition
- `src/token.py` - Tokenization utilities
- `src/prepare_data.py` - Data loading utilities
- `src/checkpoint_handler.py` - Checkpoint management
- `src/rl_trajectory_generator.py` - Trajectory generation logic
- `src/schedulars.py` - Learning rate schedulers

### 8. **Utility Modules**
- `src/utils/` directory containing:
  - `data_loader.py` - Custom data loading
  - `helper.py` - General utilities
  - `logger_helper.py` - Logging utilities
  - `transformer_helper.py` - Transformer utilities
  - `iterable_helper.py` - Iterator utilities

## Generated Files During Execution

### 9. **Runtime Outputs**
- `runs/` directory - Contains timestamped run folders
- `runs/{run_name}/iter{N}/` - Individual iteration results
- `runs/{run_name}/trajectories/` - Generated trajectory files
- `runs/{run_name}/_orchestration/` - Orchestration state files
- `runs/{run_name}/_config/` - Configuration files

### 10. **Monitoring Files**
- TensorBoard log files in the runs directory (for monitoring with `tensorboard --logdir=runs/`)