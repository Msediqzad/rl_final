# Reinforcement Learning for Architectural Space Planning

This project implements a reinforcement learning-based approach to architectural space planning. The system uses various RL algorithms (Value Iteration, Policy Iteration, and Deep RL) to optimize building layouts based on specified constraints and objectives.

## Project Structure

```
.
├── src/
│   ├── environment.py    # Custom gym-like environment for space planning
│   ├── agent.py         # Implementation of RL agents
│   ├── train.py         # Training script and experiment management
│   └── visualize.py     # Visualization utilities (TODO)
├── results/             # Directory for experiment results
├── main.tex            # LaTeX documentation
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the agents and run experiments:

```bash
python src/train.py
```

This will:
- Initialize the environment and agents
- Run training experiments for all configured algorithms
- Save results and generate visualization plots in the `results/` directory

### Configuration

The training configuration can be modified in `src/train.py`. Key parameters include:

- Environment parameters:
  - Grid size
  - Maximum steps per episode
  - Room size constraints

- Algorithm parameters:
  - Discount factor (gamma)
  - Convergence threshold (theta)
  - Neural network architecture (for Deep RL)
  - Number of training episodes

## Algorithms

The project implements three different approaches:

1. **Value Iteration**
   - Suitable for smaller state spaces
   - Guarantees optimal policy
   - Faster convergence for simple layouts

2. **Policy Iteration**
   - Better performance on larger state spaces
   - More efficient policy updates
   - Good balance of exploration/exploitation

3. **Deep RL (Actor-Critic)**
   - Handles continuous state/action spaces
   - Scales better to complex layouts
   - More flexible reward structures

## Environment

The environment (`ArchitecturalEnvironment`) implements a custom gym-like interface with:

- State space: Grid representation of the building layout
- Action space: Room creation, modification, and removal
- Reward function: Based on space utilization and design constraints

## Results

Training results are saved in the `results/` directory, including:
- Training statistics (JSON format)
- Performance comparison plots
- Training time comparisons
- Final layout visualizations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details
