import numpy as np
import torch

import cv2
from src.visualizations.render_preds import render_rollout
import os


def rollout(model,trajectory,teacher_forcing=False):
    """
    trajectory["pose"] : (T,3) normalized

    returns
    -------
    predicted : (T,3)
    """

    gt = trajectory["pose"]
    predicted = []
    pose = torch.from_numpy(gt[0]).float()
    predicted.append(pose.numpy())
    model.eval()

    T = len(gt)

    with torch.no_grad():

        for frame_idx in range(T-1):

            t = frame_idx / (T-1)
            t_tensor = torch.tensor([[t]], dtype=torch.float32)

            pose = model(pose.unsqueeze(0),t_tensor).squeeze(0)
            
            # next_pose_pred = pose + pred_delta
            predicted.append(pose.numpy())

            if teacher_forcing:
                pose = torch.from_numpy(gt[frame_idx + 1]).float()
            else:
                pose = pose

    predicted = np.stack(predicted)
    return predicted


def evaluate(model, dataset, teacher_forcing=False, save_videos=True, output_dir="outputs/"):
    """
    Evaluate the model on every active trajectory in the dataset.
    """
    print("Evaluating model performance")
    all_errors = []

    for traj_idx in (dataset.active_trajectory_indices):
        trajectory = dataset.get_trajectory(traj_idx)
        gt = trajectory["pose"]
        
        predicted = rollout(
            model=model,
            trajectory=trajectory,
            teacher_forcing=teacher_forcing,
        )

        # ----------------------------------------
        # Denormalize
        # ----------------------------------------

        gt = dataset.denormalize_pose(gt)
        predicted = dataset.denormalize_pose(predicted)

        # ----------------------------------------
        # Error
        # ----------------------------------------

        errors = gt - predicted

        primitive = trajectory["name"]
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
                filename=f"{primitive}.mp4",
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
