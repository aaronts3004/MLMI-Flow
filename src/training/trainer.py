import torch

from .objectives import (
    prepare_autoregressive_batch,
    prepare_flow_matching_batch,
)

def train(
    model,
    objective,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    num_epochs,
    device,
):
    """
    Returns

    list[float]
        Average training loss for every epoch.
    """

    model.to(device)

    loss_history = []

    if objective == "autoregressive":
        prepare_batch = prepare_autoregressive_batch
    elif objective == "flow_matching":
        prepare_batch = prepare_flow_matching_batch
    else:
        raise ValueError(f"Unknown objective '{objective}'")

    for epoch in range(num_epochs):

        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            
            # print("---")
            x, t, target = prepare_batch(batch, device)
            pred = model(x, t)
            loss = criterion(pred, target)

            # print(pred.mean(), pred.std())
            # print(target.mean(), target.std())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        epoch_loss /= len(train_loader)
        loss_history.append(epoch_loss)

        if epoch % 25 == 0:
            print(f"Epoch {epoch:03d} | loss={epoch_loss:.8f}")

    return loss_history

