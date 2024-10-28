# IDL24Fall-HW5

# Starter Code Usage

**Training**

```
python train.py --config configs/ddpm.yaml
```

**Inference and Evaluating**

```
python inference.py inference.py
```

# 1. Download the data

Please first download the data from here: https://drive.google.com/drive/u/0/folders/1Hr8LU7HHPEad8ALmMo5cvisazsm6zE8Z

After download please unzip the data with

```
tar -xvf imagenet100_128x128.tar.gz
```

# 2.Implementing DDPM from Scratch

This homework will start from implementing DDPM from scratch.

We provide the basic code structure for you and you will be implementing the following modules (by filling all TODOs)):

```
1. pipelines/ddpm.py
2. schedulers/scheduling_ddpm.py
3. train.py
4. configs/ddpm.yaml
```

A very basic U-Net architecture is provided to you, and you will need to improve the architecture for better performacne.

# 3. Implementing DDIM

Implement the DDIM from scratch:

```
1. schedulers/scheduling_ddpm.py
2. create a config with ddim by setting use_ddim to True
```

**NOTE: you need to set use_ddim to TRUE**

# 4. Implementing Latent DDPM

Implement the Latent DDPM.

The pre-trained weights of VAE and basic modules are provided. 

Download the pretrained weight here: and put it under a folder named 'pretrained' (create one if it doesn't exsit)

You need to implement:

```
1. models/vae.py
2. train.py with vae related stuff
3. pipeline/ddpm.py with vae related stuff
```

**NOTE: you need to set use_vae to TRUE**

# 5. Implementing CFG

Implement CFG

```
1. models/class_embedder.py
2. train.py with cfg related stuff
3. pipeline/ddpm.py with cfg related stuff
```

**NOTE: you need to set use_cfg to TRUE**

# 6. Evaluation

```
inference.py
```
