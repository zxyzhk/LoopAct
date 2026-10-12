# LoopAct

**Personal robotics project: visual-feedback action control**

LoopAct is a personal robotics engineering project maintained by 卓星晔 (zxyzhk).
It organizes slow action planning, high-frequency visual feedback and action-feature
caching into a single implementation built on GR00T N1.5. The repository includes
training and evaluation entry points, model components and a six-task simulation showcase.


**[Project page and simulation videos](https://zxyzhk.github.io/LoopAct/)**

## Components

| Component | Code | Role |
|---|---|---|
| Action Proposal Planner (APP) | `loopact/model/action_proposal_planner.py` | Slow action proposal generation |
| Visual Refinement Network (VRN) | `loopact/model/visual_refinement_network.py` | Feedback-conditioned last-step action refinement |
| Action Context Encoder (ACE) | `ActionContextEncoder` | Action/timestep/embodiment feature encoding |
| Visual Observation Encoder (VOE) | `loopact/model/visual_observation_encoder.py` | ResNet18 observation features |
| Cross-Modal Fusion Block (CMFB) | `CrossModalFusionBlock` | Gated action/visual fusion and velocity prediction |
| Action Context Cache (ACC) | `LoopAct.action_context_cache` | Cached planner outputs reused by refinement; not a new trainable network |

```python
from loopact.model.policy import LoopAct, LoopActConfig
from loopact.model.action_proposal_planner import ActionProposalPlanner, ActionContextEncoder
from loopact.model.visual_refinement_network import VisualRefinementNetwork, CrossModalFusionBlock
from loopact.model.visual_observation_encoder import VisualObservationEncoder
```


## Overview
This codebase has 2 parts:
- **Baseline**: Foundation model GR00T N1.5 (in `third_party/Isaac-GR00T/` submodule)
- **Feedback**: Modified feedback model (in main directory `loopact/`)

## Setup Instructions

### 1. Prepare the Source and Dependencies

Clone the public repository with its pinned dependencies:

```bash
git clone --recursive https://github.com/zxyzhk/LoopAct.git
cd LoopAct
```

If using a source ZIP, extract it first. GitHub source ZIPs do not contain submodule
code. Clone the dependencies and check out their pinned commits separately:

```bash
git clone https://github.com/jidaxian010/Isaac-GR00T.git third_party/Isaac-GR00T
git -C third_party/Isaac-GR00T checkout 88a3bb37623718a418d7d76c88271d83e478c538
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
```

If you already cloned without `--recursive`, fetch the submodules now:

```bash
git submodule update --init --recursive
```

Both submodules are public: [Isaac-GR00T](https://github.com/jidaxian010/Isaac-GR00T)
(a fork of NVIDIA's, with small changes needed by this repo) and
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).

### 2. Set Up Conda Environment and PyTorch

Use two separate environments depending on which simulator you run.
The instructions are repetitive but kept separate for clarity.

**Install PyTorch first.** Several dependencies build against it, so installing
this repo before PyTorch can fail or pull an unwanted CPU build. The versions
below are what this project was developed and tested with:

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Pick the `--index-url` matching your CUDA version; see
https://pytorch.org/get-started/locally/ for other builds.

> If you are using Blackwell GPUs, set up the correct PyTorch and Flash-Attn using: https://github.com/Zarrac/flashattention-blackwell-wheels-whl-ONLY-5090-5080-5070-5060-flash-attention-/releases/tag/FlashAttention

#### a. RoboSuite environment (`vlarobosuite`)

```bash
conda create -n vlarobosuite python=3.10
conda activate vlarobosuite
pip install --upgrade setuptools
# install PyTorch here (see above)
pip install -e ".[robo]"
pip install -e third_party/Isaac-GR00T --no-deps
```

#### b. LIBERO environment (`vlalibero`)

```bash
conda create -n vlalibero python=3.10
conda activate vlalibero
pip install --upgrade setuptools
# install PyTorch here (see above)
pip install -e ".[libero]"
pip install -e third_party/Isaac-GR00T --no-deps
pip install -e third_party/LIBERO --no-deps
```

Quote the extras (`".[robo]"`) — zsh treats unquoted brackets as a glob pattern
and will fail with `no matches found`.

Both submodules are installed with `--no-deps` on purpose: their own pins
conflict with the ones in `pyproject.toml`, which are the versions this project
is tested against. LIBERO's requirements are already covered by the `libero`
extra — it pins `robosuite==1.4.0`, `bddl` and `robomimic` directly, and MuJoCo
arrives as a robosuite dependency.

### Verify the install

```bash
python -c "import loopact, gr00t; print('ok')"
```

Run this from a directory other than the repo root — importing from the repo
root works even when the install did not.

### 3. Activate Conda Environment

Activate the environment matching the simulator you want to run:

```bash
# For RoboSuite experiments
conda activate vlarobosuite

# For LIBERO experiments
conda activate vlalibero
```

## How to Use

### 1. Collect Data

1. **Build the environment and tasks** under `experiments/robosuite/environment/` and `experiments/robosuite/tasks/`.
2. **Collect demonstrations** — set the target task name in the code, then run:
   ```bash
   python experiments/robosuite/scripts/collect_demos.py
   ```
3. **Teleoperate with a 3Dconnexion SpaceMouse, plus these keys:**

   | Key | Action |
   |-----|--------|
   | `.` | Close gripper |
   | `,` | Open gripper |
   | `k` | Discard the current demo |
   | `l` | Save the current demo |

4. **Convert demos to LeRobot format** using the scripts under `experiments/robosuite/scripts/`:

   | Script | Purpose |
   |--------|---------|
   | `demo2lerobot.py` | Convert a single demo dataset to LeRobot format |
   | `demo2lerobot_mix.py` | Convert multiple demo datasets to LeRobot format |
   | `lerobot_mix.py` | Mix multiple LeRobot datasets into one |
   | `libero2lerobot.py` | Convert a LIBERO dataset to LeRobot format |

### 2. Model Training

1. Write a config under `config/` (e.g. `base_block.yaml` for baseline, `fb_block.yaml` for feedback).
   The YAML files already in `config/` are examples only — they point at datasets and checkpoint
   paths from our own runs. Copy one and edit `dataset_path`, `output_dir`, and `base_model_path`
   to match your own data and checkpoints.
2. Train with:
```bash
# Train baseline GR00T model
python scripts/loopact_train.py --baseline --config config/base_block.yaml

# Train feedback model
python scripts/loopact_train.py --feedback --config config/fb_block.yaml
```
### 3. Inference

Pretrained checkpoints are available at
[jidaxian010/vla-feedback-sim](https://huggingface.co/jidaxian010/vla-feedback-sim):

```bash
pip install -U "huggingface_hub[cli]"
hf download jidaxian010/vla-feedback-sim --local-dir checkpoints
```

Both checkpoints are meant to be used with the `loopact` module — they are
**not** baseline GR00T checkpoints. Select the feedback mode in the inference
code when running them.

For Robosuite tasks, first change mode (baseline/feedback) and task name in the code and then run.

```bash
# Inference on a single task
python experiments/robosuite/scripts/inference_by_task.py

# Inference across multiple tasks (summarizes success rates)
python experiments/robosuite/scripts/inference_all.py
``` 
And for LIBERO tasks, also change mode and sub-dataset name and then run (also don't forget to activate libero environment with conda activate vlalibero): 

```bash
python experiments/libero/libero_runner.py
```



### 4. Checkpoint Evaluation

For debugging, evaluate trained checkpoints with stepped inference (16-step action chunks):

```bash
python scripts/eval_checkpoint.py 
``` 


## Verification

Run the dependency-free interface checks with:

```bash
python -m unittest discover -s tests -v
```

The checks cover syntax, package exports, execution-mode routing and model
serialization contracts. Full training/inference requires PyTorch, the two
submodules, data and checkpoints.

## Attribution

This implementation is based on [VLA-Feedback](https://github.com/jidaxian010/VLA-Feedback-release)
and NVIDIA Isaac-GR00T. Simulation demonstration clips are provided by the
[upstream project](https://vla-feedback.github.io/). Original copyrights,
licenses and attribution are retained in LICENSE, NOTICE and the source files.

## Notes

- **Mode selection**: Always specify `--baseline` or `--feedback` when running scripts
- **Config files**: Can override any parameter via command line
- **Video backend**: Uses `torchvision_av` by default (can change to `decord` in config)
- **Checkpoints**: Saved in `output_dir` specified in config
