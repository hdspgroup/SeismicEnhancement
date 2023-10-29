# -----------------------------------------------------------------------------
# Original Source: DDPMs from Scratch
# GitHub Repository: spmallick/learnopencv
# Link: https://github.com/spmallick/learnopencv/tree/master/Guide-to-training-DDPMs-from-Scratch
# -----------------------------------------------------------------------------
# Description:
# Train Denoising Diffusion Probabilisitic Models from scratch on the Flowers and MNIST dataset.
# -----------------------------------------------------------------------------

import torch


from torchmetrics import MeanMetric
from torch.cuda import amp
from tqdm import tqdm
from IPython.display import display, HTML, clear_output

from .config import BaseConfig, TrainingConfig
from .ddpm import forward_diffusion
from ..utils import get, frames2vid
from ..dataloader import inverse_transform
from PIL import Image

import torchvision
import torchvision.transforms as TF

import matplotlib.pyplot as plt 
import cv2
import numpy as np

def apply_custom_colormap(image_gray, cmap=plt.get_cmap('seismic')):

    assert image_gray.dtype == np.uint8, 'must be np.uint8 image'
    if image_gray.ndim == 3: image_gray = image_gray.squeeze(-1)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256))[:,0:3]    # color range RGBA => RGB
    color_range = (color_range*255.0).astype(np.uint8)         # [0,1] => [0,255]
    color_range = np.squeeze(np.dstack([color_range[:,2], color_range[:,1], color_range[:,0]]), 0)  # RGB => BGR

    # Apply colormap for each channel individually
    channels = [cv2.LUT(image_gray, color_range[:,i]) for i in range(3)]
    return np.dstack(channels)


# Algorithm 1: Training

def train_one_epoch(model, sd, loader, optimizer, scaler, loss_fn, epoch=800, 
                   base_config=BaseConfig(), training_config=TrainingConfig(), condition=None):
    
    loss_record = MeanMetric()
    model.train()

    with tqdm(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{training_config.NUM_EPOCHS}")

        for x0s in loader:

            if condition is not None:
                x0s, b_label = x0s
                b_label = b_label.to(base_config.DEVICE) - 1

            tq.update(1)            
            ts = torch.randint(low=1, high=training_config.TIMESTEPS, size=(x0s.shape[0],), device=base_config.DEVICE)
            xts, gt_noise = forward_diffusion(sd, x0s, ts)

            with amp.autocast():

                if condition is not None:
                    pred_noise = model(xts, ts, b_label)
                else:
                    pred_noise = model(xts, ts)

                loss = loss_fn(gt_noise, pred_noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.detach().item()
            loss_record.update(loss_value)

            tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")

        mean_loss = loss_record.compute().item()
    
        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")
    
    return mean_loss 



# Algorithm 2: Sampling
    
@torch.inference_mode()
def reverse_diffusion(model, sd, timesteps=1000, img_shape=(3, 64, 64), 
                      num_images=5, nrow=8, device="cpu", **kwargs):

    x = torch.randn((num_images, *img_shape), device=device)
    model.eval()

    if kwargs.get("generate_video", False):
        outs = []

    for time_step in tqdm(iterable=reversed(range(1, timesteps)), 
                          total=timesteps-1, dynamic_ncols=False, 
                          desc="Sampling :: ", position=0):

        ts = torch.ones(num_images, dtype=torch.long, device=device) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

        predicted_noise = model(x, ts)

        beta_t                            = get(sd.beta, ts)
        one_by_sqrt_alpha_t               = get(sd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(sd.sqrt_one_minus_alpha_cumulative, ts) 

        x = (
            one_by_sqrt_alpha_t
            * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )

        if kwargs.get("generate_video", False):
            x_inv = inverse_transform(x).type(torch.uint8)
            grid = torchvision.utils.make_grid(x_inv, nrow=nrow, pad_value=255.0).to("cpu")
            grid = grid.permute(1, 2, 0)
            grid = grid[..., 0, None].numpy()
            ndarr = apply_custom_colormap(grid)[:, :, ::-1]
            outs.append(ndarr)

    if kwargs.get("generate_video", False): # Generate and save video of the entire reverse process. 
        frames2vid(outs, kwargs['save_path'])
        display(Image.fromarray(outs[-1][:, :, ::-1])) # Display the image at the final timestep of the reverse process.

        
        return x

    else: # Display and save the image at the final timestep of the reverse process. 
        x = inverse_transform(x).type(torch.uint8)
        grid = torchvision.utils.make_grid(x, nrow=nrow, pad_value=255.0).to("cpu")
        grid = grid.permute(1, 2, 0)
        grid = grid[..., 0, None].numpy()
        grid = apply_custom_colormap(grid)
        pil_image = Image.fromarray(grid)
        pil_image.save(kwargs['save_path'], format= kwargs['save_path'][-3:].upper())
        display(pil_image)
        return x
    

@torch.inference_mode()
def cond_reverse_diffusion(model, sd, timesteps=1000, img_shape=(3, 64, 64), 
                      num_images=5, nrow=6, device="cpu", complexity=1, **kwargs):

    x = torch.randn((num_images, *img_shape), device=device)
    model.eval()

    # b_label = torch.randint(low=0, high=5, size=(nrow,), device=device)
    # b_label = torch.arange(5, device=device).int()
    
    # fact = num_images // nrow
    # b_label = b_label.repeat_interleave(fact)
    b_label = torch.Tensor([complexity-1]).int().repeat(num_images).to(device)

    if kwargs.get("generate_video", False):
        outs = []

    for time_step in tqdm(iterable=reversed(range(1, timesteps)), 
                          total=timesteps-1, dynamic_ncols=False, 
                          desc="Sampling :: ", position=0):

        ts = torch.ones(num_images, dtype=torch.long, device=device) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

        predicted_noise = model(x, ts, b_label)

        beta_t                            = get(sd.beta, ts)
        one_by_sqrt_alpha_t               = get(sd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(sd.sqrt_one_minus_alpha_cumulative, ts) 

        x = (
            one_by_sqrt_alpha_t
            * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )

        if kwargs.get("generate_video", False):
            x_inv = inverse_transform(x).type(torch.uint8)
            grid = torchvision.utils.make_grid(x_inv, nrow=nrow, pad_value=255.0).to("cpu")
            grid = grid.permute(1, 2, 0)
            grid = grid[..., 0, None].numpy()
            ndarr = apply_custom_colormap(grid)[:, :, ::-1]
            outs.append(ndarr)

    if kwargs.get("generate_video", False): # Generate and save video of the entire reverse process. 
        frames2vid(outs, kwargs['save_path'])
        display(Image.fromarray(outs[-1][:, :, ::-1])) # Display the image at the final timestep of the reverse process.

        
        return x

    else: # Display and save the image at the final timestep of the reverse process. 
        x = inverse_transform(x).type(torch.uint8)
        grid = torchvision.utils.make_grid(x, nrow=nrow, pad_value=255.0).to("cpu")
        grid = grid.permute(1, 2, 0)
        grid = grid[..., 0, None].numpy()
        grid = apply_custom_colormap(grid)
        pil_image = Image.fromarray(grid)
        pil_image.save(kwargs['save_path'], format= kwargs['save_path'][-3:].upper())
        display(pil_image)
        return x


@torch.inference_mode()
def affine_reverse_diffusion(model, sd, timesteps=1000, img_shape=(3, 64, 64), 
                      num_images=5, nrow=6, device="cpu", **kwargs):

    x = torch.randn((2, *img_shape), device=device)

    affine_val = np.linspace(0, 1, num=num_images)
    affine_val = np.stack([affine_val, 1 - affine_val]).T
    affine_val = np.sqrt(affine_val)
    affine_val = torch.tensor(affine_val, dtype=x.dtype, device=x.device)
    
    model.eval()

    if kwargs.get("generate_video", False):
        outs = []

    num_images_t = 2
    for time_step in tqdm(iterable=reversed(range(1, timesteps)), 
                          total=timesteps-1, dynamic_ncols=False, 
                          desc="Sampling :: ", position=0):
        
        if time_step == 150:
            print("affined")
            x = x.reshape( (2, -1) )
            x = affine_val @ x 
            x = x.reshape((num_images, *img_shape))
            num_images_t = num_images

        ts = torch.ones(num_images_t, dtype=torch.long, device=device) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)
        # z = torch.zeros_like(x)

        if time_step > 1:
            # z = torch.zeros_like(x)
            z = torch.randn((1, *img_shape), device=device)
        else:
            z = torch.zeros_like(x)
        
        

        predicted_noise = model(x, ts)

        beta_t                            = get(sd.beta, ts)
        one_by_sqrt_alpha_t               = get(sd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(sd.sqrt_one_minus_alpha_cumulative, ts) 

        x = (
            one_by_sqrt_alpha_t
            * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )

        if kwargs.get("generate_video", False):
            x_inv = inverse_transform(x).type(torch.uint8)
            grid = torchvision.utils.make_grid(x_inv, nrow=nrow, pad_value=255.0).to("cpu")
            grid = grid.permute(1, 2, 0)
            grid = grid[..., 0, None].numpy()
            ndarr = apply_custom_colormap(grid)[:, :, ::-1]
            outs.append(ndarr)

    if kwargs.get("generate_video", False): # Generate and save video of the entire reverse process. 
        # frames2vid(outs, kwargs['save_path'])
        display(Image.fromarray(outs[-1][:, :, ::-1])) # Display the image at the final timestep of the reverse process.

        
        return None

    else: # Display and save the image at the final timestep of the reverse process. 
        x = inverse_transform(x).type(torch.uint8)
        grid = torchvision.utils.make_grid(x, nrow=nrow, pad_value=255.0).to("cpu")
        grid = grid.permute(1, 2, 0)
        grid = grid[..., 0, None].numpy()
        grid = apply_custom_colormap(grid)
        pil_image = Image.fromarray(grid)
        pil_image.save(kwargs['save_path'], format= kwargs['save_path'][-3:].upper())
        display(pil_image)
        return None
