import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.datasets.rigid_dataset import RigidDataset
from src.models.mlp import PoseMLP
from src.training.trainer import train
from src.eval import evaluate

# ==========================================================
# Configuration
# ==========================================================

NUM_EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 5e-4

HDF5_PATH = "data/generated/rigid_dataset_isolated_primitives_100_trajectories.h5"
MODEL_NAME = "mlp"
OUTPUT_DIR = f"outputs/overfit/{MODEL_NAME}"
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

    dataset = RigidDataset(HDF5_PATH)

    print("Primitive types:", dataset.trajectory_names)
    print("len(dataset)=:", len(dataset))
    print()

    for primitive in PRIMITIVES:
        print(f"\nTraining on primitive: {primitive}")
        dataset.set_active_primitives(primitive)

        print("Samples:", len(dataset))

        train_loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

        model = PoseMLP()       # re-instantiate model for every primitive

        optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=200,gamma=0.5)
        criterion = nn.MSELoss()

        losses = train(
            model=model,
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

        evaluate(
            model=model,
            dataset=dataset,
            output_dir=OUTPUT_DIR,
        )

        print()

    print("Finished.")


if __name__ == "__main__":
    main()