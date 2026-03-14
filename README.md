# LM Concealers: Detecting Information Hiding in Language Models

## Overview
This is the official repository for [Seamless Deception: Larger Language Models Are Better Knowledge Concealers
]()

Language models may acquire harmful knowledge while feigning ignorance under audit. We train classifiers to detect active concealment, finding they outperform human evaluators on smaller models — but fail to generalize across unseen architectures and topics, and degrade to random performance on models above 70B parameters.

---

## Environment Setup

```bash
cd setup
pip install --upgrade pip uv
uv sync                      # creates .venv with Python 3.12
source .venv/bin/activate
```

Configure your private settings before running anything:

```bash
cp configs/private_vars.yaml.example configs/private_vars.yaml  # if example exists
# Edit configs/private_vars.yaml with:
#   storage_dir: path to where model checkpoints and data should be stored
#   personal_email: your email (used for Wikipedia API user-agent)
#   namespace: your HuggingFace username (for pushing models/datasets)
source configs/config.env
```

---

## Reproducing Results

### Step 1 — Build the Dataset

Generate Wikipedia-based Q&A pairs for each entity and topic category:

```bash
python data.py setup_data
python data.py setup_entity_contexts      # fetches Wikipedia articles
python data.py setup_probe_inference      # builds probe evaluation data
python data.py create_locking_data        # creates DPO-format datasets for training
python data.py setup_training_data --dataset writingprompts
```

### Step 2 — Run Inference (Prompt-Based Hiding)

Generate model outputs under the three prompting conditions (baseline, hidden, locked):

```bash
bash scripts/run_inference.sh             # baseline + topic-specific hidden outputs
bash scripts/run_locked_inference.sh      # password-locked outputs
bash scripts/run_probe_inference.sh       # probe responses for classifier input
```

### Step 3 — Train DPO / Gradient Ascent Models (Training-Based Hiding)

```bash
bash scripts/locking/dpo.sh              # fine-tunes models with DPO to refuse on target topics
```

For gradient ascent (unlearning), use the corresponding entry point in `llm-utils/train.py` with `--method ga`.

### Step 4 — Train Detection Classifiers

Each script trains a Llama-3.2-1B classifier on a different experimental condition:

| Script | Condition |
|--------|-----------|
| `scripts/clf/same_topic_prompting.sh` | Train and test on same topic, prompt-hidden |
| `scripts/clf/same_topic_dpo.sh` | Train and test on same topic, DPO-hidden |
| `scripts/clf/same_topic_locked.sh` | Compare password-locked vs. hidden |
| `scripts/clf/different_topic_prompting.sh` | Train on one topic, test on others |
| `scripts/clf/different_model_prompting.sh` | Train on one model, test on others |
| `scripts/clf/different_all.sh` | All topic × model combinations |
| `scripts/clf/token_ablation.sh` | Vary probe token count |

### Step 5 — Evaluate and Visualize

```bash
python eval.py --help
```


## Project Structure

```
LMConcealers/
├── data.py               # Data preparation: Wikipedia context, Q&A generation, splits
├── eval.py               # Perplexity evaluation
├── probe.py              # Probe classifier fitting and evaluation
├── prompts.py            # System prompt definitions (baseline, hidden, locked variants)
├── viz.py                # Heatmap and figure generation
├── probes.csv            # 65 detection probe templates
├── configs/
│   ├── project_vars.yaml     # Shared config (WANDB project, seed, results dir)
│   └── private_vars.yaml     # Private config (storage path, HF credentials)
├── scripts/
│   ├── run_inference.sh          # Baseline and prompt-hidden inference
│   ├── run_locked_inference.sh   # Password-locked inference
│   ├── run_probe_inference.sh    # Probe inference
│   ├── locking/dpo.sh            # DPO fine-tuning for training-based hiding
│   └── clf/                      # Classifier training scripts (9 conditions)
├── llm-utils/            # Submodule: inference (HF/vLLM) and training (SFT/DPO/GA)
├── setup/                # uv environment and requirements
└── figures/              # Generated plots and visualizations
```
