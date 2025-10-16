import os 
import sys 
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb 
import logging 
from logging import getLogger as get_logger
from tqdm import tqdm 
from PIL import Image

import torchvision
from torchvision import transforms


from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint

from train import parse_args

logger = get_logger(__name__)


def main():
    # parse arguments
    args = parse_args()
    
    # seed everything
    seed_everything(args.seed)
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # vae 
    vae = None
    if args.latent_ddpm:        
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
    # cfg
    class_embedder = None
    if args.use_cfg:
        # TODO: class embeder
        class_embedder = ClassEmbedder(n_classes= args.num_classes, embed_dim=args.unet_ch)
        
    # send to device
    unet = unet.to(device)
    unet.eval()
    # scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)
        
    # scheduler
    if args.use_ddim:
        shceduler_class = DDIMScheduler
    else:
        shceduler_class = DDPMScheduler
    # TOOD: scheduler
    scheduler = shceduler_class(
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range
    ).to(device)

    # load checkpoint
    load_checkpoint(unet, scheduler=None, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)
    # TODO: pipeline
    pipeline = DDPMPipeline(unet, scheduler, vae=vae, class_embedder=class_embedder)

    
    logger.info("***** Running Infrence *****")
    
    # TODO: we run inference to generation 5000 images
    # TODO: with cfg, we generate 50 images per class 
    all_images = []
    batch_size=args.batch_size

    if args.use_cfg:
        # generate 50 images per class
        for i in tqdm(range(args.num_classes)):
            logger.info(f"Generating 50 imageså for class {i}")
            batch_size = 50
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)

            gen_images = pipeline(batch_size=batch_size,
                                    num_inference_steps=10,
                                    generator=generator,
                                    classes=classes,
                                    guidance_scale=args.cfg_guidance_scale,
                                    device=device ) 
            all_images.extend(gen_images)

    else:
        # generate 5000 images
        num_gen_image =250
        progress_bar = tqdm(range(num_gen_image // batch_size), dynamic_ncols=True)
        for _ in range(0, num_gen_image, batch_size):
            cfg_guidance_scale = args.cfg_guidance_scale if args.cfg_guidance_scale > 0 else None
            gen_images = pipeline(batch_size=batch_size,
                                    num_inference_steps=args.num_inference_steps,
                                    generator=generator,
                                    classes=args.num_classes,
                                        guidance_scale=cfg_guidance_scale,
                                        device=device )
            all_images.extend(gen_images)
            progress_bar.update()


    import random

    selected_images = random.sample(all_images, 5)

    # Save the selected images as PNG files
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    for i, img in enumerate(selected_images):
        output_path = f"{output_dir}/image_{i+1}.png"
        img.save(output_path, format="PNG")
        print(f"Saved: {output_path}")


    # TODO: load validation images as reference batch
    logger.info("Loading validation images for FID/IS computation")
    val_dir = os.path.join(args.data_dir, 'validation')


    # Custom transform to bypass normalization
    def to_uint8_tensor(image):
        # Convert to tensor without normalization
        tensor = torch.tensor(np.array(image), dtype=torch.uint8)
        # Rearrange dimensions if image is RGB
        if len(tensor.shape) == 3:  # (H, W, C)
            tensor = tensor.permute(2, 0, 1)  # Convert to (C, H, W)
        return tensor

    # Transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Lambda(to_uint8_tensor),  # Convert to uint8 tensor
    ])


    validation_dataset = torchvision.datasets.ImageFolder(val_dir,transform=transform)
    # Define metric objects
    logger.info("Computing FID and IS")
    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception_score = InceptionScore().to(device)


    # Convert PIL images to torch.Tensor with dtype=torch.uint8
    tensor_images = []
    for img in all_images:
        img = transforms.functional.pil_to_tensor(img).numpy()
        img = torch.tensor(img,  dtype=torch.uint8)
        tensor_images.append(img)

    # Stack the tensors into a single batch
    tensor_images = torch.stack(tensor_images)

    # Combine all images (generated and reference) into a single loader for batch processing
    generated_loader = torch.utils.data.DataLoader(tensor_images, batch_size=batch_size, shuffle=False)
    reference_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # Process generated and reference images in a single loop
    for gen_batch, [ref_batch, _] in tqdm(zip(generated_loader, reference_loader), total=len(generated_loader)):
        # Update metrics for generated images
        gen_batch = gen_batch.to(device)
        ref_batch = ref_batch.to(device)

        fid.update(gen_batch, real=False)
        inception_score.update(gen_batch)

        # Update metrics for reference images
        fid.update(ref_batch, real=True)

    # Compute and log metrics
    fid_score = fid.compute()
    is_score = inception_score.compute()
    logger.info(f"FID Score: {fid_score}")
    logger.info(f"Inception Score: {is_score[0].item():.4f}, {is_score[1].item():.4f}")
    

if __name__ == '__main__':
    main()
