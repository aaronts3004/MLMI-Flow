import numpy as np
import torch

import cv2
import os 

from src.visualizations.render_preds import render_rollout
from src.training.objectives import (
    prepare_autoregressive_batch,
    prepare_flow_matching_batch,
)

def rollout_flow_matching():
    raise NotImplementedError(
        "Flow Matching inference not implemented yet."
    )


def rollout_autoregressive(model, trajectory, teacher_forcing=False):
    """
    trajectory["pose"] : (T,3) normalized

    returns
    -------
    predicted : (T,3)
    """

    gt = trajectory["trajectory"]
    gt = gt.clone().float()
    predicted = []
    pose = gt[0].float()
    predicted.append(pose.numpy())
    model.eval()

    T = len(gt)

    with torch.no_grad():
        for frame_idx in range(T-1):
            t = frame_idx / (T-1)
            t_tensor = torch.tensor([[t]], dtype=torch.float32)
            pose = model(pose.unsqueeze(0),t_tensor).squeeze(0)
            
            # append predicted pose to all predictions
            predicted.append(pose.numpy())

            if teacher_forcing:
                pose = torch.from_numpy(gt[frame_idx + 1]).float()
            else:
                pose = pose

    predicted = np.stack(predicted)
    return predicted


def evaluate(model, objective, dataset, teacher_forcing=False, save_videos=True, output_dir="outputs/"):
    """
    Evaluate the model on every active trajectory in the dataset.
    """
    print("Evaluating model performance")
    all_errors = []

    for traj_idx in (dataset.active_trajectory_indices):

        trajectory = dataset.get_trajectory(traj_idx)
        gt = trajectory["trajectory"]
        gt = gt.clone().float()

        if objective == "autoregressive":
            predicted = rollout_autoregressive(model=model,
                trajectory=trajectory,
                teacher_forcing=teacher_forcing,
            )

        elif objective == "flow_matching":
            predicted = rollout_flow_matching()
        else: 
            print("Unknown Training Objective!")
            raise ValueError

        # ----------------------------------------
        # Denormalize
        # ----------------------------------------

        gt = dataset.denormalize_pose(gt)
        predicted = dataset.denormalize_pose(predicted)

        # ----------------------------------------
        # Error
        # ----------------------------------------

        errors = gt - predicted

        primitive = trajectory["primitive"]
        traj_name = trajectory["name"]

        print()
        print("=" * 60)
        print(f"{primitive}")
        print("=" * 60)

        print("Mean error")
        print(errors.mean(axis=0))

        print("Std error")
        print(errors.std(axis=0))

        all_errors.append(errors)

        # ----------------------------------------
        # Render rollout
        # ----------------------------------------

        if save_videos:

            suffix = (
                "teacher_forced"
                if teacher_forcing
                else "autoregressive"
            )

            render_rollout(
                predicted=predicted,
                gt=gt,
                output_dir=output_dir,
                filename=f"{primitive}_{traj_name}.mp4",
            )

    # ----------------------------------------
    # Overall statistics
    # ----------------------------------------

    all_errors = np.concatenate(all_errors, axis=0)

    print()
    print("=" * 60)
    print("Evaluation")
    print("=" * 60)

    print("Mean error")
    print(all_errors.mean(axis=0))

    print("Std error")
    print(all_errors.std(axis=0))

    return {
        "mean_error": all_errors.mean(axis=0),
        "std_error": all_errors.std(axis=0),
    }
