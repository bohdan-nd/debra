
# DEBRA

Bohdan Naida, Kateryna Nekhomiazh, Mariia Rizhko.

The project report can be found [here](./DEBRA_Final_Report.pdf).

## Overview
DEBRA is a research project focused on analyzing the Decision Transformer (DT) model, a GPT-based architecture developed for solving Reinforcement Learning tasks. The main goal was to critically evaluate the Decision Transformer on various MuJoCo environments, investigate its training stability, and compare it with Conservative Q-Learning (CQL).

## Key Findings
- **Analysis of Decision Transformer:** The original DT paper was scrutinized for inconsistencies in implementation and evaluation. The DT's performance in different versions of gym MuJoCo environments, such as HalfCheetah, Hopper, and Walker2d, was assessed.

- **Training Stability:** The study identified issues with the training stability of the DT, noting that the performance fluctuated significantly during training.
- **Comparison with CQL:** DEBRA compares the DT with CQL, revealing that CQL often outperforms DT in various environments, especially in terms of training stability.

# Limitations and Future Work
- **Performance in Stochastic Environments:** A Significant limitation identified was the DT's inability to perform reliably in stochastic environments.
- **Lack of Explainability:** The DT, like many machine learning models, lacks explainability, making it challenging to understand and trust its decision-making process.

## Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
Then, dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```

## Downloading datasets

Datasets are stored in the `data` directory.
Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.
Then, run the following script in order to download the datasets and save them in our format:

```
python download_d4rl_datasets.py
```

## Example usage

Experiments can be reproduced with the following:

```
python experiment.py --env hopper --dataset medium --env_version 0 --model_type dt
```

Possible options:
```
--env: hopper, halfcheetah, walker2d
--dataset: medium, medium-replay, medium-expert, expert
--env_version: 0, 2
--model_type: dt, debra, rodebra, badebra, debraxl
```


