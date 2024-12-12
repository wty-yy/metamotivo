# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Example of benchmarking Meta Motivo using humenv[bench].
IMPORTANT: You need to provide paths to the models, motions, and goals
Simple run command:
python -m examples.humenv_evaluation
"""

from metamotivo.fb_cpr import FBcprModel
from metamotivo.wrappers.humenvbench import RewardWrapper, TrackingWrapper, GoalWrapper
from pathlib import Path
from humenv.bench.utils.buffer import load_hdf5, SimpleBuffer
from humenv.bench import (
    RewardEvaluation,
    GoalEvaluation,
    TrackingEvaluation,
)
import humenv
import json
import numpy as np

# IMPORTANT: Update the variables below:
MODELS_ROOT = "" # The path to the root dir of the models
FINAL_MODELS = [
 #Here goes the list of models to be tested (dir names)
]
BUFFER_FILENAME = "buffer_inference_5000000.hdf5" # Buffer filename
# The three paths below point to the default locations where HumEnv data_preparation process stores or creates respective files:
MOTIONS = "<your git root location>/humenv/data_preparation/test_train_split/0-CMU_train_0.1.txt" 
MOTIONS_BASE_PATH = "<your git root location>/humenv/data_preparation/humenv_amass/"
GOAL_POSES_PATH = "<your git root location>/humenv/data_preparation/goal_poses/goals.json"
PROCESS_EXECUTOR = False

def get_tracking_motions(debug: bool = False):
    motions_to_track = []
    if debug:  # Shortlist poses
        motions_to_track = [
            "0-MPI_Limits_03099_op4_poses",
            "0-MPI_Limits_03101_ulr1c_poses",
            "0-CMU_88_88_05_poses",
            "0-CMU_22_23_Rory_22_16_poses",
        ]
        motions_to_track = [
            Path(MOTIONS_BASE_PATH) / f"{el}.hdf5" for el in motions_to_track
        ]
    else:
        motions_to_track = list(Path(MOTIONS_BASE_PATH).glob("*_poses.hdf5"))
    return list(map(str, motions_to_track))


def get_goal_poses(pose_file):
    module_path = Path(humenv.bench.__file__).resolve().parent
    with open(
        module_path / pose_file, "r"
    ) as json_file:  # TODO: test if this path works when package is pip-installed
        goals = json.load(json_file)
    for pose_name, payload in goals.items():
        goals[pose_name] = np.array(payload["observation"])
    return goals


def main(model_path):
    device = "cuda"
    tasks = ["move-ego-0-0"]
    num_samples_per_inference: int = 10_000_000
    inference_function: str = "reward_wr_inference"
    buffer_path = str(model_path) + "/" + BUFFER_FILENAME
    print(f"loading data from {buffer_path}")
    data = load_hdf5(buffer_path)
    buffer = SimpleBuffer(capacity=data["action"].shape[0])
    buffer.extend(data)
    print(f"loading the model from {model_path}")
    model = FBcprModel.load(path=model_path, device=device)
    agent = RewardWrapper(
        model=model,
        inference_dataset=buffer,
        num_samples_per_inference=num_samples_per_inference,
        inference_function=inference_function,
        max_workers=80,
        process_executor=PROCESS_EXECUTOR
    )
    agent = TrackingWrapper(model=agent)
    agent = GoalWrapper(model=agent)

    reward_eval = RewardEvaluation(
        tasks=tasks,
        env_kwargs={
            "state_init": "MoCapAndFall",
            "context": "spawn"
        },
        num_contexts=1,
        num_envs=50,
        num_episodes=100,
        motion_base_path=MOTIONS_BASE_PATH,
        motions=MOTIONS,
    )
    goal_eval = GoalEvaluation(
        goals=get_goal_poses(GOAL_POSES_PATH),
        env_kwargs={
            "state_init": "MoCapAndFall",
            "context": "spawn"
        },
        num_contexts=1,
        num_envs=50,
        num_episodes=100,
        motion_base_path=MOTIONS_BASE_PATH,
        motions=MOTIONS,
    )
    tracking_eval = TrackingEvaluation(
        motions=get_tracking_motions(debug=True),
        env_kwargs={
            "state_init": "Default",
            "context": "spawn"
        },
        num_envs=1, #Tracking spawns its own processes, so setting num_envs to 1.
    )
    import time
    start_t = time.time()
    print(f"Reward started at {time.ctime(start_t)}")
    reward_metrics = reward_eval.run(agent=agent)
    print(f"Reward eval time:{time.time()-start_t}")

    start_t = time.time()
    print(f"Goal started at {time.ctime(start_t)}")
    goal_metrics = goal_eval.run(agent=agent)
    print(f"Goal eval time:{time.time()-start_t}")

    start_t = time.time()
    print(f"Tracking started at {time.ctime(start_t)}")
    tracking_metrics = tracking_eval.run(agent=agent)
    print(f"Tracking eval time:{time.time()-start_t}")

    print("REWARD")
    print(reward_metrics)
    print("GOAL")
    print(goal_metrics)
    print("TRACKING")
    print(tracking_metrics)


if __name__ == "__main__":
    model_path = Path(MODELS_ROOT) / FINAL_MODELS[0]
    main(model_path=model_path)
