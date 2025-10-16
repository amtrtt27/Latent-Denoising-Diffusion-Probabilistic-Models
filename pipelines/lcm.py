
from typing import List, Optional, Tuple, Union
from PIL import Image
from tqdm import tqdm
import torch 
import torch.nn as nn
from utils import randn_tensor


# LCM Code is modified based on the original paper https://arxiv.org/pdf/2310.04378 and
# Huggingface LCM schduler https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/latent_consistency_models/pipeline_latent_consistency_text2img.py

class LCMPipeline:
    def __init__(self, unet, scheduler, vae, class_embedder):
        self.unet = unet
        self.scheduler = scheduler
        
        # NOTE: this is for latent DDPM
        self.vae = vae
            
        # NOTE: this is for CFG
        self.class_embedder = class_embedder

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")



    
    def get_w_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        see https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
        Args:
        timesteps: torch.Tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        dtype: data type of the generated embeddings
        Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb




    @torch.no_grad()
    def __call__(
        self, 
        batch_size: int = 1,
        num_inference_steps: int = 1000,
        classes: Optional[Union[int, List[int]]] = None,
        guidance_scale : Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        device = None,
    ):
        image_shape = (batch_size, self.unet.input_ch, self.unet.input_size, self.unet.input_size)
        if device is None:
            device = next(self.unet.parameters()).device
        
        
        # convert classes to tensor
        if isinstance(classes, int):
            classes = [classes] * batch_size
        elif isinstance(classes, list):
            assert len(classes) == batch_size, "Length of classes must be equal to batch_size"
            classes = torch.tensor(classes, device=device)
            
        # TODO: get class embeddings from classes
        class_embeds = self.class_embedder(classes)

        
        # TODO: starts with random noise
        image = randn_tensor(image_shape, generator=generator, device=device)

        # TODO: set step values using set_timesteps of scheduler
        self.scheduler.set_lcm_timesteps(num_inference_steps)
        
        # Get Guidance Scale Embedding
        w = torch.tensor(guidance_scale).repeat(batch_size)
        w_embedding = self.get_w_embedding(w, embedding_dim=self.unet.tdim).to(device=device, dtype=image.dtype)

        # TODO: inverse diffusion process with for loop
        for t in self.progress_bar(self.scheduler.timesteps):
            
            # Predict noise using the model
            model_output = self.unet(image, t, c=class_embeds, t_c=w_embedding)
    
            # TODO: 2. compute previous image: x_t -> x_t-1 using scheduler
            image, denoised = self.scheduler.step(model_output, t, image, generator=generator, return_dict=False)
            
        
        # NOTE: this is for latent DDPM
        # TODO: use VQVAE to get final image

        denoised = denoised / 0.1845
        image = self.vae.decode(denoised) 
        # TODO: clamp your images values
        image = image.clamp(-1, 1)
        
        # TODO: return final image, re-scale to [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)
        
        # convert to PIL images
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = self.numpy_to_pil(image)
        
        return image
        


