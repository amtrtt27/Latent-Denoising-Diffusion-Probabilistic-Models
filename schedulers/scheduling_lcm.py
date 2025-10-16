from typing import List, Optional, Tuple, Union

import torch 
import torch.nn as nn 
import numpy as np

from utils import randn_tensor

from.scheduling_ddim import DDIMScheduler

# LCM Code is modified based on the original paper https://arxiv.org/pdf/2310.04378 and
# Huggingface LCM schduler https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lcm.py


class LCMScheduler(DDIMScheduler):    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.num_inference_steps is not None, "Please set `num_inference_steps` before running inference using DDIM."
        self.set_timesteps(self.num_inference_steps)


    def set_lcm_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).
        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """

        original_steps = self.num_inference_steps

        # LCM Timesteps Setting
        # The skipping step parameter k from the paper.
        k = self.num_train_timesteps // original_steps
        # LCM Training/Distillation Steps Schedule
        # Currently, only a linearly-spaced schedule is supported (same as in the LCM distillation scripts).
        lcm_origin_timesteps = np.asarray(list(range(1, int(original_steps) + 1))) * k - 1

        skipping_step = len(lcm_origin_timesteps) // num_inference_steps

        if skipping_step < 1:
            raise ValueError(
                f"The original_steps {original_steps}  is smaller than `num_inference_steps`: {num_inference_steps}."
            )

        self.num_inference_steps = num_inference_steps

        # LCM Inference Steps Schedule
        lcm_origin_timesteps = lcm_origin_timesteps[::-1].copy()
        # Select (approximately) evenly spaced indices from lcm_origin_timesteps.
        inference_indices = np.linspace(0, len(lcm_origin_timesteps), num=num_inference_steps, endpoint=False)
        inference_indices = np.floor(inference_indices).astype(np.int64)
        timesteps = lcm_origin_timesteps[inference_indices]

        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.long)


    def get_scalings_for_boundary_condition_discrete(self, t):
        self.sigma_data = 0.5  # Default: 0.5

        # By dividing 0.1: This is almost a delta function at t=0.
        c_skip = self.sigma_data**2 / ((t / 0.1) ** 2 + self.sigma_data**2)
        c_out = (t / 0.1) / ((t / 0.1) ** 2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out

    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        return_dict = True,
    ) -> torch.Tensor:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of the noise to add to the variance.
            generator (`torch.Generator`, *optional*):
                A random number generator.

        Returns:
            pred_prev_sample (`torch.Tensor`):
                The predicted previous sample.
        """
        
        t = timestep
        prev_t = self.previous_timestep(t)
        
        # TODO: 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)
        
        # TODO: 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.prediction_type == 'epsilon':
            pred_original_sample = (sample - (beta_prod_t ** 0.5) * model_output) / alpha_prod_t ** (0.5) 
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented.")

        # Denoise model output using boundary conditions
        denoised = c_out * pred_original_sample + c_skip * sample

        # Sample z ~ N(0, I), For MultiStep Inference
        # Noise is not used for one-step sampling.
        if len(self.timesteps) > 1:
            noise = torch.randn(model_output.shape).to(model_output.device)
            prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
        else:
            prev_sample = denoised

        if not return_dict:
            return (prev_sample, denoised)
        
        return prev_sample
