# Robot Localization Using Search and Learning-Guided Policies

A robot localization project combining A* search, belief-state tracking, convolutional neural networks, and iterative policy improvement to reduce the number of moves needed to determine a robot's position in an unknown-location scenario.

## Project Overview

Imagine a robot navigating a spaceship after losing access to its position sensors. It knows the maze layout but cannot determine which open cell it occupies.

The robot maintains a **belief set** containing every location where it might currently be. As it moves, this set is updated according to the maze's walls and available paths.

The objective is to reduce the belief set to a single location using as few moves as possible.

This project develops a rule-based baseline and progressively improves its decisions using learned estimates of the remaining localization cost.

## Technical Stack

- **Language:** Python
- **Machine learning:** PyTorch, convolutional neural networks
- **Algorithms:** A* search, belief-state updates, heuristic search
- **Numerical computing:** NumPy
- **Visualization:** Matplotlib
- **Optimization:** Adam, mean squared error

## Methodology

### 1. Maze Generation and Belief-State Representation

Generated 30 × 30 grid mazes containing open cells and obstacles.

The robot's uncertainty is represented as a set of possible positions, called the belief set.

For each movement command, the belief set is updated by applying the same action to every possible position. Positions blocked by obstacles remain unchanged.

Localization is complete when the belief set contains exactly one position.

### 2. Baseline Localization Policy (π₀)

Implemented a baseline strategy using A* search and maze dead ends or corners as navigation targets.

The algorithm selects a possible starting location, computes a path toward a target, and applies movement commands while updating the belief set.

The baseline provides localization trajectories and estimates of the number of moves needed from different states.

### 3. Training Data Generation

Generated localization examples using 200 randomly constructed mazes.

The dataset contains **47,570 samples**, combining full and partial belief sets.

Each training example contains:

- A grid representation of the maze and possible robot positions
- The number of remaining moves required under the baseline strategy

The data was divided into 80% training and 20% testing.

### 4. CNN-Based Localization Cost Prediction

Built a convolutional neural network to estimate the number of moves remaining before localization.

**Architecture:**

- Input: 30 × 30 belief-state grid
- Convolutional layer: 16 filters, 3 × 3 kernel
- ReLU activation and max pooling
- Convolutional layer: 32 filters, 3 × 3 kernel
- ReLU activation and max pooling
- Fully connected layer: 64 neurons
- Output: one nonnegative predicted localization cost

**Baseline training configuration:**

- Optimizer: Adam
- Learning rate: 0.0005
- Loss function: Mean Squared Error
- Batch size: 64
- Epochs: 100

The CNN predicts localization cost rather than the robot's exact position.

### 5. Learning-Guided Policy Improvement

Developed three localization policies:

**π₀ — Baseline strategy**

Uses the original A*-guided localization procedure without a learned cost estimate.

**π₁ — One-step learned lookahead**

Simulates each of the four possible first moves, evaluates the resulting belief states using the trained CNN, and selects the move with the lowest predicted remaining cost.

After the selected move, the robot continues with the baseline policy.

**π₂ — Iterative policy improvement**

Uses a second cost-estimation model trained on data generated under π₁.

The updated estimator guides the next action, after which the policy follows the previously improved strategy.

This approach illustrates how learned cost estimates can guide search decisions without replacing the underlying localization algorithm.

## Results

- Generated 47,570 training examples from 200 mazes.
- Trained a CNN to estimate remaining localization moves.
- Observed decreasing training and testing losses during model training.
- Compared the average localization moves required by π₀, π₁, and π₂ across different initial belief-set sizes.
- Found that the learning-guided policies generally reduced localization steps compared with the baseline, although performance varied between trials and belief-set sizes.

The experiments demonstrate the potential of combining traditional search with learned cost estimates for sequential decision-making.

## Repository Structure

| File | Purpose |
|---|---|
| `environment.py` | Generates grid mazes |
| `constants.py` | Defines grid dimensions and cell values |
| `directions.py` | Defines movement directions |
| `L_next.py` | Updates possible robot locations after an action |
| `solution_two.py` | Implements the baseline localization experiment |
| `part_2_belief_sets.py` | A* search and belief-state trajectory generation |
| `data_generation.py` | Generates labeled training data |
| `Training.py` | Defines and trains the baseline CNN |
| `Part_4.py` | Implements CNN cost estimation and π₁ comparisons |
| `PI1_model.py` | Generates π₁ data and trains the updated model |
| `Pi2.py` | Implements π₂ and compares all three policies |
| `requirements.txt` | Python dependency list |
| `Project_3_CS520 BotLocalization.pdf` | Complete project report |

## Running the Project

The project consists of multiple Python scripts that depend on generated data and trained model checkpoints.

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Generate the baseline dataset:**

```bash
python data_generation.py
```

This produces `mixed_train_data.npz`.

**Train the baseline CNN:**

```bash
python Training.py
```

This produces `localization_cnn_model.pt`.

**Train the improved cost estimator:**

```bash
python PI1_model.py
```

This produces `localization_pi1_model.pt`.

**Compare the localization policies:**

```bash
python Pi2.py
```

The scripts are research-oriented implementations rather than a packaged application. The commands above reflect the intended dependency order; execution has not been independently verified in a clean environment.

## Limitations and Future Improvements

- Localization performance depends on maze structure and initial uncertainty.
- Learned cost estimates may not always select the globally optimal action.
- Training and evaluation are based on simulated mazes.
- Future work could explore larger mazes, stronger value-function approximators, improved training-data generation, and more extensive policy evaluation.

## Academic Context

Developed for **CS520 — Introduction to Artificial Intelligence**, Rutgers University, May 2025.

**Project authors:** Zeba Afreen and Sridevi Mudiyanur Ramamurthy.

The complete report is available in `Project_3_CS520 BotLocalization.pdf`.
