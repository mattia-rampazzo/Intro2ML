import torch
import timm
import torch.nn as nn

torch.manual_seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k", pretrained=True, features_only=True)
print(encoder)

