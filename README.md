# GCULab

## Overview

**GCU Lab** - Additions to IsaacLab Core functionality for Pack task.

**Pack Task** - Packing environment.

---

### Key Features

#### Pack Task
Packing gym environment for the Amazon task.

**Command:**
```bash
python scripts/test_placement_agent.py --task=Isaac-Pack-NoArm-v0 --num_envs 5
```

---

#### IK Reachability Analysis
Requires **curobo** ([Installation Instructions](https://curobo.org/get_started/1_install_instructions.html)).

**Command:**
```bash
python scripts/ik_reachability_agent.py --task=Isaac-Pack-UR5-v0 --num_envs 1
```

---

## Installation

### Step 1: Install Isaac Lab
Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
We recommend using the **conda installation** as it simplifies calling Python scripts from the terminal.

### Step 2: Clone or Copy This Repository
Ensure this project/repository is separate from the Isaac Lab installation (i.e., outside the `IsaacLab` directory).

### Step 3: Install the Library in Editable Mode
Using a Python interpreter with Isaac Lab installed, run:
```bash
# Use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
python -m pip install -e source/tote_consolidation
python -m pip install -e source/gculab
python -m pip install -e source/gculab_assets
```

---

## Running Tasks with Dummy Agents

These include dummy agents that output zero or random actions. They are useful for verifying environment configurations.

### Zero-Action Agent
**Command:**
```bash
# Use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
python scripts/zero_agent.py --task=<TASK_NAME>
```

---

### Random-Action Agent
**Command:**
```bash
# Use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
python scripts/random_agent.py --task=<TASK_NAME>
```

---

### Test Placement Agent
**Command:**
```bash
# Use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
python scripts/test_placement_agent.py --task=Isaac-Pack-NoArm-v0 --num_envs 5
```

---

## Code Formatting

We use a **pre-commit template** to automatically format your code.

### Installation
Install pre-commit with:
```bash
pip install pre-commit
```

### Running Pre-Commit
Run pre-commit for all files:
```bash
pre-commit run --all-files
```
