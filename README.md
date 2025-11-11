# AIF-ARC-AGI

**AIF-ARC-AGI** is a Active Inference LLM multi-agent system for solving ARC-AGI puzzles. This repository contains the code and experiments from our paper:

> **"How Modality Shapes Perception and Reasoning: A Study of Error Propagation in ARC-AGI"**

## Overview

This repository implements a systematic study of how different input modalities (text vs. image encodings) affect transformer-based models' perception and reasoning on ARC-AGI tasks. Our approach isolates perception from reasoning using a two-stage pipeline:

1. **Perception Task**: Measures how accurately models identify and locate visual features across nine different encoding modalities
2. **Reasoning Task**: Evaluates how modality choice affects instruction generation and execution accuracy

## Key Findings

- Text encodings (JSON, ASCII) excel at precise coordinate identification for sparse features
- Image encodings capture 2D spatial relationships but suffer from patch-size-dependent aliasing effects
- Combining complementary modalities enables cross-validation that improves both perception accuracy (by ~8 points) and execution similarity (by ~0.20 median improvement) without changing the underlying model

## Repository Structure

```
.
├── src/                    # Core source code
│   ├── nodes/             # PocketFlow node implementations
│   │   ├── models.py      # Pydantic models for structured outputs
│   │   └── hypothesis_fast_nodes.py  # Hypothesis generation nodes
│   └── utils/             # Utility functions
│       ├── data_loader.py      # ARC-AGI data loading
│       ├── modality_encoder.py # Multi-modal encoding utilities
│       ├── follow_instructions.py  # Instruction execution engine
│       ├── scoring_engine.py   # Grid similarity scoring
│       └── score_utils.py      # Scoring utilities
├── perception/            # Perception experiments
│   ├── run_modality_vision_experiment.py  # Main perception experiment script
│   ├── calculate_perception_scores.ipynb   # Score calculation notebook
│   └── data/              # Experiment results
├── generate-execute/      # Reasoning experiments
│   ├── run_single_modality_experiment.py   # Single modality experiments
│   ├── run_double_modality_experiment.py   # Cross-order experiments
│   └── data/              # Experiment results
├── data/                  # ARC-AGI challenge datasets
│   ├── arc-prize-2024/   # ARC Prize 2024 challenges
│   └── arc-prize-2025/   # ARC Prize 2025 challenges
└── pyproject.toml         # Project dependencies

```

## Installation

### Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd AIF-ARC-AGI-public
```

2. Install dependencies:
```bash
uv sync
```

3. Set up environment variables:
```bash
cp example.env .env
# Edit .env and add your API keys
```

Required environment variables:
- `GEMINI_API_KEY_2`: Google Gemini API key (get from https://aistudio.google.com/app/apikey)

Optional environment variables:
- `ENABLE_LANGFUSE`: Set to `true` to enable Langfuse tracing (optional)
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`: Langfuse configuration (optional)

**Testing other LLMs/VLMs**: This repository uses [LiteLLM](https://docs.litellm.ai/docs/) for multi-provider LLM integration, which supports many providers including OpenAI, Anthropic Claude, Groq, DeepSeek, Qwen, and open-source models via [vLLM](https://docs.litellm.ai/docs/providers/vllm). To test other models, simply add the corresponding API key environment variables to your `.env` file. See the [LiteLLM provider documentation](https://docs.litellm.ai/docs/providers) for the required environment variable names for each provider.

## Usage

### Perception Experiments

Test how well models perceive visual information across different modalities:

```bash
cd perception
uv run python run_modality_vision_experiment.py \
  --challenge-ids 13e47133 0934a4d8 135a2760 136b0064 142ca369 \
  --rpm 60
```

This tests all modalities (row_only, col_only, ascii, json, image_14x14, image_15x15, image_16x16, image_17x17, image_24x24, image_768x768) and generates detailed descriptions of what the model sees.

See `perception/README.md` for detailed documentation.

### Reasoning Experiments

Test how modality choice affects hypothesis generation and execution. Our reasoning experiments use a held-out validation approach (leave-one-out cross-validation on training examples) and cell-wise similarity scoring, inspired by the methodology in [ARC Lang](https://github.com/arc-lang-public) (see citation in paper):

**Single modality experiment:**
```bash
cd generate-execute
uv run python run_single_modality_experiment.py \
  --challenge-id 13e47133 \
  --modality-type row_col_json_image \
  --rpm 60
```

**Cross-order experiment (tests both ascending and descending example orders):**
```bash
cd generate-execute
uv run python run_double_modality_experiment.py \
  --challenge-id 13e47133 \
  --rpm 60
```

Supported modality types:
- `json_only`: JSON format only
- `row_col_json`: Row-wise + column-wise text + JSON
- `row_col_json_image`: Row-wise + column-wise text + JSON + images
- `image_json`: Images + JSON
- `row_only`, `col_only`, `image_only`, `row_col`, `row_image`, `col_image`, `row_col_image`: Text/image combinations

See `generate-execute/README.md` for detailed documentation.

### Reproducing Paper Results

The paper experiments can be reproduced using the scripts above. Key experiment configurations:

1. **Cross-order experiment** (20251105_1142): Tests all 7 modalities with both ascending and descending orders
2. **JSON-based experiments** (20251108_*): Tests JSON-based modalities with ascending order

See `generate-execute/README.md` for specific commands to reproduce each experiment.

## Data

The repository includes ARC-AGI challenge datasets:
- `data/arc-prize-2024/`: ARC Prize 2024 challenges and solutions
- `data/arc-prize-2025/`: ARC Prize 2025 challenges and solutions

Each dataset includes:
- `arc-agi_training_challenges.json`: Training challenges
- `arc-agi_training_solutions.json`: Training solutions
- `arc-agi_test_challenges.json`: Test challenges
- `arc-agi_evaluation_challenges.json`: Evaluation challenges
- `arc-agi_evaluation_solutions.json`: Evaluation solutions

## Architecture

The system is built on [PocketFlow](https://github.com/the-pocket/PocketFlow), a minimalist LLM framework for multi-agent workflows. Key components:

- **Modality Encoder**: Converts ARC-AGI grids into multiple encoding formats (text, images, JSON)
- **Hypothesis Nodes**: Generate transformation hypotheses using LLMs
- **Execution Engine**: Execute hypotheses to produce output grids
- **Scoring Engine**: Calculate similarity scores between predicted and ground truth grids

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{wen2025modality,
  title={How Modality Shapes Perception and Reasoning: A Study of Error Propagation in ARC-AGI},
  author={Wen, Bo and Wang, Chen and Bilal, Erhan},
  journal={arXiv:},
  year={2025}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

## Contributing

This is a research repository accompanying our paper. For questions or issues, please open a GitHub issue.

## Acknowledgments

- Built on [PocketFlow](https://github.com/the-pocket/PocketFlow) framework
- Uses [LiteLLM](https://github.com/BerriAI/litellm) for multi-provider LLM integration
- ARC-AGI challenges from [ARC Prize](https://www.kaggle.com/competitions/arc-agi-2024)
- Held-out evaluation methodology and scoring approach inspired by [ARC Lang](https://github.com/arc-lang-public) (see citation in paper)
