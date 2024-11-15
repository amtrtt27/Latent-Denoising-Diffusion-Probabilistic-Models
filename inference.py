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
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from torchvision.utils  import make_grid

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
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # # TODO: ddpm shceduler
    # scheduler = DDPMScheduler(
    #     num_train_timesteps=args.num_train_timesteps,
    #     beta_start=args.beta_start,
    #     beta_end=args.beta_end,
    #     beta_schedule=args.beta_schedule,
    #     clip_sample=args.clip_sample,
    #     clip_sample_range=args.clip_sample_range
    # )
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
        class_embedder = ClassEmbedder(None)
        
    # send to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
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
    )

    # load checkpoint
    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)
    
    # TODO: pipeline
    pipeline = DDPMPipeline(unet, scheduler, vae=vae, class_embedder=class_embedder)

    
    logger.info("***** Running Infrence *****")
    
    # TODO: we run inference to generation 5000 images
    # TODO: with cfg, we generate 50 images per class 
    all_images = []
    if args.use_cfg:
        # generate 50 images per class
        for i in tqdm(range(args.num_classes)):
            logger.info(f"Generating 50 images for class {i}")
            batch_size = 50
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images = None 
            all_images.append(gen_images)
    else:
        # generate 5000 images
        for _ in tqdm(range(0, 5000, batch_size)):
            gen_images = pipeline.generate(num_inference_steps=args.num_inference_steps,
                                           generator=generator,
                                            classes=args.num_classes,
                                             guidnace_scale=args.guidnace_scale,
                                              device=device )
            all_images.append(gen_images)
    
    # TODO: load validation images as reference batch
    logger.info("Loading validation images for FID/IS computation")
    val_dir = os.path.join(args.data_dir, 'dev')

    validation_dataset = torchvision.datasets.ImageFolder(val_dir,
                                              transform=transforms.Compose([
                                                  transforms.Resize((args.image_size, args.image_size)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                              ]))
    
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    reference_images = []
    for images, _ in validation_loader:
        reference_images.append(images)
    reference_images = torch.cat(reference_images, dim=0).to(device)

    # TODO: using torchmetrics for evaluation, check the documents of torchmetrics
    import torchmetrics 
    from torchmetrics.image.fid import FrechetInceptionDistance, InceptionScore

    # Compute metrics
    logger.info("Computing FID and IS")
    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception_score = InceptionScore().to(device)

    # Add generated images to metrics
    for img_batch in torch.utils.data.DataLoader(torch.stack(all_images), batch_size=batch_size):
        fid.update(img_batch, real=False)
        inception_score.update(img_batch)

    # Add reference images to metrics
    for img_batch in torch.utils.data.DataLoader(reference_images, batch_size=batch_size):
        fid.update(img_batch, real=True)

    # Compute and log metrics
    fid_score = fid.compute()
    is_score = inception_score.compute()
    logger.info(f"FID Score: {fid_score}")
    logger.info(f"Inception Score: {is_score}")
    wandb.log({"FID Score": fid_score, "Inception Score": is_score})
        
    

if __name__ == '__main__':
    main()