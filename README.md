# LLM-Working-Memory

This repository contains code for experiments described in the paper:
**"LLMs Do Not Have Human-Like Working Memory"** (Huang et al., 2025)

## Setup

Before running the experiments, create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
TOGETHER_API_KEY=your_together_api_key
```

## Experiments

This repository contains three main experiments:

### 1. `guess_numbers.py`

A task where the model guesses a number from a given range.

* Adjust the number range via the `NUMBER` variable.
* Control the number of repeated runs using the `REPEAT` variable.

### 2. `binary_search.py`

Binary search-style property inference with multiple question types.

* Use `Q_NUMBER` to set the number of questions per property.
* Use `REPEAT` to repeat the experiment multiple times.

### 3. `math_magic.py`

A task testing arithmetic and memory retention.

* Use `REPEAT` to control how many times the experiment runs.

## Citation

If you use this code or find the work helpful, please cite:

```bibtex
@article{huang2025llms,
  title={LLMs Do Not Have Human-Like Working Memory},
  author={Huang, Jen-tse and Sun, Kaiser and Wang, Wenxuan and Dredze, Mark},
  journal={arXiv preprint arXiv:2505.10571},
  year={2025}
}
```
