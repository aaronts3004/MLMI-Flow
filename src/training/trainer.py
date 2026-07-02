import torch


def train(
    model,
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

    for epoch in range(num_epochs):

        model.train()
        epoch_loss = 0.0

        for batch in train_loader:

            pose = batch["pose"].to(device)
            t = batch["trajectory_t"].to(device)
            target = batch["next_pose"].to(device)

            pred = model(pose, t)

            loss = criterion(pred, target)

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