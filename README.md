
# DEBRA

Bohdan Naida, Kateryna Nekhomiazh, Mariia Rizhko.

Project report can be found [here](./DEBRA_Final_Report.pdf).

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
