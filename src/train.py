import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.datasets.rigid_dataset import RigidDataset
from src.models.mlp import PoseMLP, TrajectoryMLP
from src.training.trainer import train
from src.eval import evaluate

# ==========================================================
# Configuration
# ==========================================================

NUM_EPOCHS = 1000
LEARNING_RATE = 5e-4

HDF5_PATH = "data/generated/rigid_dataset_isolated_primitives_100_trajectories.h5"
OBJECTIVE = "flow_matching"    # can be "autoregressive", "flow_matching"

if OBJECTIVE == "flow_matching": 
    MODEL_NAME = "traj_mlp"
    model_constructor = TrajectoryMLP
    BATCH_SIZE = 8
elif OBJECTIVE == "autoregressive": 
    MODEL_NAME = "pose_mlp"
    model_constructor = PoseMLP
    BATCH_SIZE = 1
else: 
    print("Unknown Training Objective !")
    raise ValueError

OUTPUT_DIR = f"outputs/overfit/{MODEL_NAME}/{OBJECTIVE}"
SAVE_CKPT = False

PRIMITIVES = [
    "line",
    "circle",
    "arc",
    "rectangle",
    "zigzag",
    "figure8",
]


def main():
    print("Running training")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)
    print("\n\nLoading Dataset...")

    dataset = RigidDataset(HDF5_PATH, OBJECTIVE)
    print()

    for primitive in PRIMITIVES:
        print(f"\nTraining on primitive: {primitive}")
        dataset.set_active_primitives(primitive)
        
        if OBJECTIVE == "autoregressive": 
            dataset.set_active_trajectory(10)            # try to overfit the PoseMLP to a single trajectory

        print("Samples for this primitive (num_trajectories):", len(dataset))

        train_loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
        model = model_constructor()       # re-instantiate model for every primitive

        optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=200,gamma=0.5)
        criterion = nn.MSELoss()

        losses = train(
            model=model,
            objective=OBJECTIVE,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            num_epochs=NUM_EPOCHS,
            device=device,
        )

        if SAVE_CKPT:
            save_path = os.path.join(OUTPUT_DIR,f"{primitive}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")

        try: 
            evaluate(
                model=model,
                objective=OBJECTIVE,
                dataset=dataset,
                output_dir=OUTPUT_DIR,
            )
        except: 
            print("evaluation failed")

        print()

    print("Finished.")


if __name__ == "__main__":
    main()