import numpy as np
import pandas as pd
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import model


# Parameter settings ---------------------------------------------------------------------------------------------------
csv_path = "../input_files.csv"
out_dir = "../out_data"

# hyper parameters
n_epoch = 1001
batch_size = 64
z_dim = 500
lr = 0.0001
data_len = 4096
n_critic = 5
beta_1 = 0.5
beta_2 = 0.9
_lambda = 10

seed = 0
seed_numpy = 1234

# ----------------------------------------------------------------------------------------------------------------------
try:
    os.makedirs(out_dir, exist_ok=True)
except OSError as error:
    print(error)
    pass

# other parameters
time = np.linspace(0.0, (data_len - 1) / 100, data_len)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))

# random seed values
rng = np.random.RandomState(seed_numpy)
torch.manual_seed(seed)

# ----------------------------------------------------------------------------------------------------------------------
# Initialize the G and D
netG = model.Generator(nch=z_dim).to(device)
netG.apply(model.weight_init)

netD = model.Discriminator().to(device)
netD.apply(model.weight_init)

# load the data
wave_dataset = model.GroundMotionDatasets(csv_path)
data_loader = DataLoader(wave_dataset, batch_size=batch_size, shuffle=True)

# Optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta_1, beta_2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta_1, beta_2))

# noise for validation
fixed_noise = 2 * torch.rand(30, z_dim, 1, 1, device=device) - 1

# Training -------------------------------------------------------------------------------------------------------------
log_G_losses = []
log_D_losses = []
log_epoch = []
log_iteration = []
log_i = []
# log_Wasserstein_D = []

log_epoch_2 = []
log_D_losses_2 = []
log_G_losses_2 = []
log_Wasserstein_D_2 = []

iteration = 0
batches_done = 0

netG.train()
netD.train()

for epoch in range(n_epoch):
    temp_G_loss = []
    temp_D_loss = []
    temp_Wasserstein_D = []

    for itr, data in enumerate(data_loader):
        real_wave = data[0].to(device)
        sample_size = real_wave.size(0)

        # Generate the noise
        noise = 2 * torch.rand(sample_size, z_dim, 1, 1, device=device) - 1

        # Training Discriminator
        netD.zero_grad()

        fake_wave = netG(noise)
        temp = netD(fake_wave.detach())

        # if epoch == 0 and itr == 0:
        #     print('generated wave size: {}'.format(fake_wave.shape))
        #     print('size of the Discriminator output: {}'.format(temp.shape))

        gradient_penalty = model.gradient_penalty(
            netD, real_data=real_wave, fake_data=fake_wave, device=device, wei=_lambda
        )
        errD_real = torch.mean(netD(real_wave))
        errD_fake = torch.mean(temp)
        errD = -1 * errD_real + errD_fake + gradient_penalty
        errD.backward()
        optimizerD.step()

        Wasserstein_D = errD_real - errD_fake
        temp_D_loss.append(errD.item())
        temp_Wasserstein_D.append(Wasserstein_D.item())

        temp_errG = 0
        if (itr + 1) % n_critic == 0:
            netG.zero_grad()
            errG = -1 * torch.mean(netD(fake_wave))
            errG.backward()

            optimizerG.step()

            log_epoch.append(epoch)
            log_i.append(itr)
            log_iteration.append(iteration)
            log_D_losses.append(errD.item())
            log_G_losses.append(errG.item())

            temp_G_loss.append(errG.item())

            temp_errG = errG.item()

        if (itr + 1) % 60 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]"
                % (
                    epoch,
                    n_epoch,
                    batches_done % len(data_loader),
                    len(data_loader),
                    errD.item(),
                    temp_errG,
                )
            )

        iteration += 1
        batches_done += 1

    log_epoch_2.append(epoch)
    log_G_losses_2.append(np.mean(temp_G_loss))
    log_D_losses_2.append(np.mean(temp_D_loss))
    log_Wasserstein_D_2.append(np.mean(temp_Wasserstein_D))

    if epoch % 10 == 0:
        netG.eval()

        fake_wave_test = netG(fixed_noise)

        out_data = fake_wave_test.to("cpu").detach().numpy()

        out_data = np.squeeze(out_data)
        out_mat = time.reshape(-1, 1)
        col_names = ["time"]

        for num in range(out_data.shape[0]):
            out_mat = np.append(out_mat, out_data[num, :].reshape(-1, 1), axis=1)
            col_names.append("wave_" + str(num))

        out_df = pd.DataFrame(out_mat, columns=col_names)
        out_path = out_dir + "wave_epoch_" + str(epoch) + ".csv"
        out_df.to_csv(out_path, index=False)

        netG.train()

    if epoch % 100 == 0:
        temp_state_dict_G = netG.state_dict()
        temp_out_path_G = out_dir + "model_G_epoch_" + str(epoch) + ".pth"
        torch.save(temp_state_dict_G, temp_out_path_G)

        temp_state_dict_D = netD.state_dict()
        temp_out_path_D = out_dir + "model_D_epoch_" + str(epoch) + ".pth"
        torch.save(temp_state_dict_D, temp_out_path_D)

log_G_losses = np.array(log_G_losses)
log_D_losses = np.array(log_D_losses)
log_epoch = np.array(log_epoch)
log_iteration = np.array(log_iteration)
log_i = np.array(log_i)
log_epoch_2 = np.array(log_epoch_2)
log_D_losses_2 = np.array(log_D_losses_2)
log_G_losses_2 = np.array(log_G_losses_2)
log_Wasserstein_D_2 = np.array(log_Wasserstein_D_2)

out_mat_1 = np.stack(
    [log_epoch, log_i, log_iteration, log_D_losses, log_G_losses], axis=1
)
col_names_1 = ["Epoch", "index", "Iteration", "Loss_D", "Loss_G"]
df1 = pd.DataFrame(out_mat_1, columns=col_names_1)
df1.to_csv(out_dir + "results_all.csv", index=False)

out_mat_2 = np.stack(
    [log_epoch_2, log_D_losses_2, log_G_losses_2, log_Wasserstein_D_2], axis=1
)
col_names_2 = [
    "Epoch",
    "Loss_D: mean of each epochs",
    "Loss_G: the same as Loss_D",
    "Wasserstein_distance",
]
df2 = pd.DataFrame(out_mat_2, columns=col_names_2)
df2.to_csv(out_dir + "results_mean.csv", index=False)
