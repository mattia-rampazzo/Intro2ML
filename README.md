# Basic fine-tuning
For now just replace original output layer and performs fine tune on this

## Main requirements
- pip install torch
- pip install torchvision
- pip install torchinfo
- pip install timm
- pip install wandb

## How to run
Change config setting in YAML file (e.g. backbone):

- Try different datasets ('datset_name': num_classes)
    - 'aircraft': 102,
    - 'cars': 196,
    - 'cub2011': 200,
    - 'dogs': 120,
    - 'food': 101,
    - 'flowers': 102

- Try different backbones:
    - convnext_base.fb_in22k_ft_in1k
    - convnextv2_base.fcmae_ft_in22k_in1k
    - timm/vit_base_patch16_224.augreg2_in21k_ft_in1k
    - resnet50.a1_in1k
    - regnety_002.pycls_in1k
    - efficientnet_b3.ra2_in1k
- Run on terminal
`py main.py --config config.yaml -- run_name first`

## Work to do
- transformers
- logger
- checkpoints + saving model
- improving training (more layers?, optimizer, ...?)
- data augmentation and domain adaptation
- other datasets
- our net
- automatize (model selection)
- practice