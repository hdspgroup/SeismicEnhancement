# -----------------------------------------------------------------------------
# Original Source: Generative Adversarial Nets
# Link: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# -----------------------------------------------------------------------------
# Description:
# Generative Adversarial Networks
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
# Original Source: Improved training of Wasserstein GANs
# GitHub Repository: Lornatang/WassersteinGAN_GP-PyTorch
# Link: https://github.com/Lornatang/WassersteinGAN_GP-PyTorch
# -----------------------------------------------------------------------------
# Description:
# Improved training of Wasserstein GANs
# -----------------------------------------------------------------------------

import torch
from libs.adversarial.utils import get_gradient, gradient_penalty, get_crit_loss, get_gen_loss, calculate_gradient_penalty

def train_one_epoch(netD, netG, loader, optimizerD, optimizerG, loss_fn, epoch, device, nz):

    real_label = 1.
    fake_label = 0.

    crit_repeats = 5
    c_lambda = 10

    critic_losses    = []
    generator_losses = []


    for i, data in enumerate(loader, 0):

        mean_iteration_critic_loss = 0
        real_images = data.to(device)
        batch_size = real_images.size(0)

        for _ in range(crit_repeats):
            ### Update critic ###
            noise = torch.randn(batch_size, nz, 1, 1, device=device)

            ##############################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ##############################################
            # Set discriminator gradients to zero.
            optimizerD.zero_grad()

            # Train with real
            real_output = netD(real_images)
            errD_real = torch.mean(real_output)
            D_x = real_output.mean().item()

            # Generate fake image batch with G
            fake_images = netG(noise)

            # Train with fake
            fake_output = netD(fake_images.detach())
            errD_fake = torch.mean(fake_output)
            D_G_z1 = fake_output.mean().item()

            # Calculate W-div gradient penalty
            gradient_penalty = calculate_gradient_penalty(netD, real_images.data, fake_images.data, device)

            # Add the gradients from the all-real and all-fake batches
            errD = -errD_real + errD_fake + gradient_penalty * c_lambda
            errD.backward()
            # Update D
            optimizerD.step()

            mean_iteration_critic_loss += errD.item() / crit_repeats


        critic_losses += [mean_iteration_critic_loss]


        ### Update generator ###
        optimizerG.zero_grad()

        # Generate fake image batch with G
        fake_images = netG(noise)
        fake_output = netD(fake_images)
        errG = -torch.mean(fake_output)
        D_G_z2 = fake_output.mean().item()
        errG.backward()
        optimizerG.step()

        # Keep track of the average generator loss
        generator_losses += [errG.item()]


        ## Train with all-real batch
        # netD.zero_grad()
        # Format batch
        
        # b_size = real_cpu.size(0)
        # label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # # Forward pass real batch through D
        # output = netD(real_cpu).view(-1)
        # # Calculate loss on all-real batch
        # errD_real = loss_fn(output, label)
        # # Calculate gradients for D in backward pass
        # errD_real.backward()
        # D_x = output.mean().item()

        # ## Train with all-fake batch
        # # Generate batch of latent vectors
        # noise = torch.randn(b_size, nz, 1, 1, device=device)
        # # Generate fake image batch with G
        # fake = netG(noise)
        # label.fill_(fake_label)
        # # Classify all fake batch with D
        # output = netD(fake.detach()).view(-1)
        # # Calculate D's loss on the all-fake batch
        # errD_fake = loss_fn(output, label)
        # # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        # errD_fake.backward()
        # D_G_z1 = output.mean().item()
        # # Compute error of D as sum over the fake and the real batches
        # errD = errD_real + errD_fake
        # # Update D
        # optimizerD.step()

        # ############################
        # # (2) Update G network: maximize log(D(G(z)))
        # ###########################
        # netG.zero_grad()
        # label.fill_(real_label)  # fake labels are real for generator cost
        # # Since we just updated D, perform another forward pass of all-fake batch through D
        # output = netD(fake).view(-1)
        # # Calculate G's loss based on this output
        # errG = loss_fn(output, label)
        # # Calculate gradients for G
        # errG.backward()
        # D_G_z2 = output.mean().item()
        # # Update G
        # optimizerG.step()

        # Output training stats
        # if i % 50 == 0:
        #     print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
        #           % (epoch, -1, i, len(loader),
        #              errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # # Save Losses for plotting later
        # G_losses.append(errG.item())
        # D_losses.append(errD.item())

        # # Check how the generator is doing by saving G's output on fixed_noise
        # if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        #     with torch.no_grad():
        #         fake = netG(fixed_noise).detach().cpu()
        #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        # iters += 1

        print_val = f"Epoch: {epoch}/{-1} Steps:{i}/{len(loader)}\t"
        print_val += f"Loss_C : {mean_iteration_critic_loss:.6f}\t"
        print_val += f"Loss_G : {errG:.6f}\t"  
        print(print_val, end='\r',flush = True)

    return generator_losses, critic_losses
