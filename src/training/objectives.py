import torch


''' Utils for defining different training objectives using the batch information '''


# fixed_noise = torch.randn([1, 100, 3])
# fixed_tau = torch.full((1,1,1), 0.5)

# RANDOM_NOISE=True  
# RANDOM_TAU = False

def prepare_autoregressive_batch(batch, device):
    x = batch["pose"].to(device)
    t = batch["trajectory_t"].to(device)
    target = batch["next_pose"].to(device)
    return x, t, target

def prepare_flow_matching_batch(batch, device):

    target_trajectory = batch["trajectory"].to(device)                # full, unnoisy trajectory of poses
    noisy_trajectory = torch.randn_like(target_trajectory)     # source trajectory, starts from noise

    tau = torch.rand(
        target_trajectory.size(0),
        1,
        1,
        device=device,
    )

    noisy_trajectory = torch.randn_like(target_trajectory)

    tau = torch.rand(
        target_trajectory.size(0), 1, 1,
        device=device,
    )
    
    xt = (1 - tau) * noisy_trajectory + tau * target_trajectory
    ut = target_trajectory - noisy_trajectory

    # print("------------------------")
    # print(target_trajectory.mean(), target_trajectory.std())
    # print(noisy_trajectory.mean(), noisy_trajectory.std())
    # print(ut.mean(), ut.std())
    # print("xt stats: ", xt.mean(), xt.std())
    # print("ut stats: ", ut.mean(), ut.std())


    return xt, tau, ut