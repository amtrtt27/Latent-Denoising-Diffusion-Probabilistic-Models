

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
from utils import randn_tensor
import uuid
from train import parse_args

from torchvision import datasets, transforms
from torchvision.utils  import make_grid

from models import UNet, VAE, ClassEmbedder
from schedulers import LCMScheduler
from pipelines import LCMPipeline
from utils import seed_everything, init_distributed_device, is_primary, AverageMeter, str2bool, save_checkpoint, load_checkpoint


# LCM Code is modified based on the original paper https://arxiv.org/pdf/2310.04378 and
# Huggingface LCM schduler https://github.com/luosiallen/latent-consistency-model/blob/main/LCM_Training_Script/consistency_distillation/train_lcm_distill_sd_wds.py

def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.Tensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
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


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out


# Compare LCMScheduler.step, Step 4
def get_predicted_original_sample(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0



@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)



def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]




class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = torch.arange(1, ddim_timesteps + 1, device=alpha_cumprods.device) * step_ratio - 1
        self.ddim_timesteps = self.ddim_timesteps.long()

        # Ensure alpha_cumprods is a tensor
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]

        # For previous alpha cumprods, concatenate the first element with selected indices
        first_element = alpha_cumprods[0].unsqueeze(0)
        prev_indices = self.ddim_timesteps[:-1]
        self.ddim_alpha_cumprods_prev = torch.cat([first_element, alpha_cumprods[prev_indices]], dim=0)


    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev





def main(args):

    # parse arguments
    args = parse_args()

    logger = get_logger(__name__)
    seed_everything(args.seed)


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


    # setup distributed initialize and device
    device = init_distributed_device(args) 
    if args.distributed:
        logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0
    
    # setup dataset
    logger.info("Creating dataset")
    # TODO: use transform to normalize your images to [-1, 1]
    # TODO: you can also use horizontal flip
    transform = transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
    # TOOD: use image folder for your train dataset
    train_dir = os.path.join(args.data_dir, 'train')
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    

    # TODO dataloader
    train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, sampler=None, pin_memory=True
                    )
    
    # calculate total batch_size
    total_batch_size = args.batch_size * args.world_size 
    args.total_batch_size = total_batch_size
    
    # setup experiment folder
    if args.run_name is None:
        args.run_name = f'exp-{len(os.listdir(args.output_dir))}'
    else:
        args.run_name = f'exp-{len(os.listdir(args.output_dir))}-{args.run_name}'
    output_dir = os.path.join(args.output_dir, args.run_name)
    save_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(args.output_dir, exist_ok=True)
    if is_primary(args):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)


    logger.info(f"Creating model")
    # unet
    teacher_unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)


    scheduler = LCMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range
    ).to(device)
    

    vae = VAE()
    # NOTE: do not change this
    vae.init_from_ckpt('pretrained/model.ckpt')
    vae.eval()
    

    class_embedder = ClassEmbedder(n_classes=args.num_classes, embed_dim=args.unet_ch)
    class_embedder = class_embedder.to(device)

    vae = vae.to(device)
    # send to device
    teacher_unet = teacher_unet.to(device)
    scheduler = scheduler.to(device)

    load_checkpoint(unet=teacher_unet, scheduler=scheduler, class_embedder=class_embedder, checkpoint_path=args.ckpt)

    class_embedder.eval()
    teacher_unet.eval()


    # 7. Create online student U-Net. This will be updated by the optimizer (e.g. via backpropagation.)
    # Add `time_cond_proj_dim` to the student U-Net if `teacher_unet.config.time_cond_proj_dim` is None
    time_cond_proj_dim = args.unet_ch * 4
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    unet.to(device)
    # load teacher_unet weights into unet
    unet.load_state_dict(teacher_unet.state_dict(), strict=False)
    unet.train()

    # 8. Create target student U-Net. This will be updated via EMA updates (polyak averaging).
    # Initialize from (online) unet
    target_unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    target_unet = target_unet.to(device)

    target_unet.load_state_dict(unet.state_dict())
    target_unet.train()
    target_unet.requires_grad_(False)


    # Move teacher_unet to device, optionally cast to weight_dtype
    target_unet.to(device)
    teacher_unet.to(device)


    solver = DDIMSolver(
        scheduler.alphas_cumprod,
        timesteps=scheduler.num_train_timesteps,
        ddim_timesteps=scheduler.num_inference_steps,
    )

    # DDPMScheduler calculates the alpha and sigma noise schedules (based on the alpha bars) for us
    alpha_schedule = torch.sqrt(scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - scheduler.alphas_cumprod)

    # Also move the alpha and sigma noise schedules to accelerator.device.
    alpha_schedule = alpha_schedule.to(device)
    sigma_schedule = sigma_schedule.to(device)
    solver = solver.to(device)

    # 12. Optimizer creation
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


    # max train steps
    num_update_steps_per_epoch = len(train_loader)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # dump config file
    if is_primary(args):
        experiment_config = vars(args)
        with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            # Use the round_trip_dump method to preserve the order and style
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)
    
    # start tracker
    if is_primary(args):
        wandb_logger = wandb.init(
            project='ddpm', 
            name=f"{args.run_name}", 
            config=vars(args))
    
    # Start training    
    if is_primary(args):
        logger.info("***** Training arguments *****")
        logger.info(args)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Total optimization steps per epoch {num_update_steps_per_epoch}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

    if args.test_imple:
        args.num_epochs = 1

    pipeline = LCMPipeline(unet, scheduler, vae=vae, class_embedder=class_embedder)
    target_pipeline = LCMPipeline(target_unet, scheduler, vae=vae, class_embedder=class_embedder)
    scaler = torch.amp.GradScaler()
    # progress_bar = tqdm(range(args.max_train_steps), disable=not is_primary(args))
    # training
    for epoch in range(args.num_epochs):
        
        progress_bar = tqdm(range(len(train_loader)), dynamic_ncols=True, disable=not is_primary(args))
        # args.epoch = epoch
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Learning Rate: {current_lr:.7f}")
        
        loss_m = AverageMeter()
        
        # TODO: set unet and scheduelr to train
        unet.train()
        
        generator = torch.Generator(device=device)
        generator.manual_seed(epoch + args.seed)
        
        # TODO: finish this
        for step, (images, labels) in enumerate(train_loader):
            
            batch_size = images.size(0)
            
            # TODO: send to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True) 

            images = vae.encode(images) 
            # NOTE: do not change  this line, this is to ensure the latent has unit std
            latents = images * 0.1845
            
            # TODO: zero grad optimizer
            optimizer.zero_grad()
            class_emb = class_embedder(labels)
            
            # TODO: get uncond classes
            uncond_classes = torch.full((batch_size,), class_embedder.num_classes, device=device) 

            # TODO: get uncon class embeddings
            uncond_embeds = class_embedder(uncond_classes)

            
            with torch.amp.autocast(device_type='cuda', enabled=True):
                # TODO: sample noise 
                noise = randn_tensor(
                    images.shape, generator=generator, device=device, dtype=images.dtype
                ) 
                
                # TODO: sample timestep t
                # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
                # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
                # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
                topk = scheduler.num_train_timesteps // scheduler.num_inference_steps
                index = torch.randint(0, scheduler.num_inference_steps, (batch_size,), device=latents.device).long()
                start_timesteps = solver.ddim_timesteps[index].to(device)
                timesteps = start_timesteps - topk
                timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

                # 3. Get boundary scalings for start_timesteps and (end) timesteps.
                c_skip_start, c_out_start = scalings_for_boundary_conditions(
                    start_timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
                c_skip, c_out = scalings_for_boundary_conditions(
                    timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

                # 4. Sample noise from the prior and add it to the latents according to the noise magnitude at each
                # timestep (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]

                noisy_model_input = scheduler.add_noise(latents, noise, start_timesteps)

                # 5. Sample a random guidance scale w from U[w_min, w_max] and embed it
                w = (args.w_max - args.w_min) * torch.rand((batch_size,)) + args.w_min
                w_embedding = guidance_scale_embedding(w, embedding_dim=time_cond_proj_dim)
                w = w.reshape(batch_size, 1, 1, 1)
                # Move to U-Net device and dtype
                w = w.to(device=latents.device, dtype=latents.dtype)
                w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)

                # 7. Get online LCM prediction on z_{t_{n + k}} (noisy_model_input), w, c, t_{n + k} (start_timesteps)
                noise_pred = unet(
                    noisy_model_input,
                    start_timesteps,
                    t_c=w_embedding,
                    c=class_emb,
                )

                pred_x_0 = get_predicted_original_sample(
                    noise_pred,
                    start_timesteps,
                    noisy_model_input,
                    args.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )

                model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

                # 8. Compute the conditional and unconditional teacher model predictions to get CFG estimates of the
                # predicted noise eps_0 and predicted original sample x_0, then run the ODE solver using these
                # estimates to predict the data point in the augmented PF-ODE trajectory corresponding to the next ODE
                # solver timestep.
                with torch.no_grad():
                    # 1. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and conditional embedding c
                    cond_teacher_output = teacher_unet(
                        noisy_model_input,
                        start_timesteps,
                        c=class_emb,
                    )

                    cond_pred_x0 = get_predicted_original_sample(
                        cond_teacher_output,
                        start_timesteps,
                        noisy_model_input,
                        args.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )

                    if args.prediction_type == "epsilon":
                        cond_pred_noise = cond_teacher_output
                    else:
                        raise NotImplementedError("This functionality is not implemented yet.")

                    # 2. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and unconditional embedding 0
                    uncond_teacher_output = teacher_unet(
                        noisy_model_input,
                        start_timesteps,
                        c=uncond_embeds,
                    )

                    uncond_pred_x0 = get_predicted_original_sample(
                        uncond_teacher_output,
                        start_timesteps,
                        noisy_model_input,
                        scheduler.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )

                    if args.prediction_type == "epsilon":
                        uncond_pred_noise = uncond_teacher_output
                    else:
                        raise NotImplementedError("This functionality is not implemented yet.")

                    # 3. Calculate the CFG estimate of x_0 (pred_x0) and eps_0 (pred_noise)
                    # Note that this uses the LCM paper's CFG formulation rather than the Imagen CFG formulation
                    pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                    pred_noise = cond_pred_noise + w * (cond_pred_noise - uncond_pred_noise)
                    # 4. Run one step of the ODE solver to estimate the next point x_prev on the
                    # augmented PF-ODE trajectory (solving backward in time)
                    # Note that the DDIM step depends on both the predicted x_0 and source noise eps_0.
                    x_prev = solver.ddim_step(pred_x0, pred_noise, index)

                # 9. Get target LCM prediction on x_prev, w, c, t_n (timesteps)
                with torch.no_grad():
                    target_noise_pred = target_unet(
                        x_prev,
                        timesteps,
                        t_c=w_embedding,
                        c=class_emb,
                    )

                    pred_x_0 = get_predicted_original_sample(
                        target_noise_pred,
                        timesteps,
                        x_prev,
                        scheduler.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    target = c_skip * x_prev + c_out * pred_x_0

                # Calculate loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # record loss
            loss_m.update(loss.item())
            
            # backward and step 
            scaler.scale(loss).backward() # This is a replacement for loss.backward()

            # TODO: step your optimizer
            scaler.step(optimizer) # This is a replacement for optimizer.step()
            scaler.update()
            progress_bar.set_postfix(loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg))
            progress_bar.update()

            if args.test_imple:
                break
                    # logger
            if step % 200 == 0 and is_primary(args):
                wandb_logger.log({'loss': loss_m.avg, 
                                  'lr': current_lr})
        
        progress_bar.close()
        # validation
        # send unet to evaluation mode
        lr_scheduler.step()
        unet.eval()
        torch.cuda.empty_cache()        
        update_ema(target_unet.parameters(), unet.parameters(), args.ema_decay)

           # random sample 4 classes
        classes = torch.randint(0, args.num_classes, (4,), device=device)
        # TODO: fill pipeline

        def gen_image(pipeline1, pipeline2):
            gen_images1 = pipeline1(batch_size=len(classes),
                            num_inference_steps=10,
                            classes=classes,
                            guidance_scale=args.cfg_guidance_scale,
                            generator=generator,
                            device = device,
                        )  

            # create a blank canvas for the grid
            grid_image1 = Image.new('RGB', (4 * args.image_size, 1 * args.image_size))
            # paste images into the grid
            for i, image in enumerate(gen_images1):
                x = (i % 4) * args.image_size
                y = 0
                grid_image1.paste(image, (x, y))


            # Construct the file name using run_name and epoch
            img_file_name = f"{args.run_name}_epoch_{epoch}_1.png"
            img_file_path = os.path.join(output_dir, img_file_name)

            # Save the grid image
            grid_image1.save(img_file_path)
            print(f"Grid image saved as '{img_file_path}'")
            
            # Send to wandb
            if is_primary(args):
                wandb_logger.log({'gen_images': wandb.Image(grid_image1)})


            gen_images2 = pipeline2(batch_size=len(classes),
                    num_inference_steps=10,
                    classes=classes,
                    guidance_scale=args.cfg_guidance_scale,
                    generator=generator,
                    device = device,
                )  

            # create a blank canvas for the grid
            grid_image2 = Image.new('RGB', (4 * args.image_size, 1 * args.image_size))
            # paste images into the grid
            for i, image in enumerate(gen_images2):
                x = (i % 4) * args.image_size
                y = 0
                grid_image2.paste(image, (x, y))

            # Ensure the output directory exists
            img_output_dir = "output_images2"
            os.makedirs(img_output_dir, exist_ok=True)

            # Construct the file name using run_name and epoch
            img_file_name = f"{args.run_name}_epoch_{epoch}_2.png"
            img_file_path = os.path.join(output_dir, img_file_name)

            # Save the grid image
            grid_image2.save(img_file_path)
            print(f"Grid image saved as '{img_file_path}'")

        gen_image(pipeline1=pipeline, pipeline2=target_pipeline)

        # save checkpoint
        if is_primary(args) and epoch % 3 == 0:
            save_checkpoint(unet=[unet, target_unet], optimizer=optimizer, epoch=epoch, save_dir=save_dir)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    main(args)
