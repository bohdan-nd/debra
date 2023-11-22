import json
import torch
import numpy as np
import gym
import argparse

from decision_transformer.evaluate_episodes import evaluate_episode_rtg


def record_video(folder_path, model_file_name):
    with open(f"{folder_path}/config.json", 'r', encoding='utf-8') as file:
        config = json.load(file)

    model = torch.load(f"{folder_path}/{model_file_name}", map_location="cpu")
    env = gym.make(config["env_id"])

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    state_mean, state_std = np.load(
        f"{folder_path}/state_mean.npy"), np.load(f"{folder_path}/state_std.npy")

    target_rew = max(config["env_targets"])
    scale = config["scale"]

    with torch.no_grad():
        evaluate_episode_rtg(
            env=env,
            state_dim=state_dim,
            act_dim=act_dim,
            model=model,
            max_ep_len=100000,
            scale=scale,
            target_return=target_rew/scale,
            mode=config["mode"],
            state_mean=state_mean,
            state_std=state_std,
            device='cpu'
        )

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Making a video')
    parser.add_argument('--folder_path', type=str, default=None)
    parser.add_argument('--model_file_name', type=str, default=None)

    args = parser.parse_args()
    folder_path, model_file_name = args.folder_path, args.model_file_name
    record_video(folder_path, model_file_name)
