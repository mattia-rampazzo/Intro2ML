"# Intro2ML" 


Vanilla training version: for now just replace original output layer and performs fine tune on this, with fixed lr, momentum, optimizer

Change backbone in config.yaml. Try with this:
- convnext_base.fb_in22k_ft_in1k
- convnextv2_base.fcmae_ft_in22k_in1k
- resnet50.a1_in1k
- regnety_002.pycls_in1k
- efficientnet_b3.ra2_in1k

To run
python main.py --config config.yaml

Before run
pip install timm
pip install wandb
pip install torch
pip install torchvision
...