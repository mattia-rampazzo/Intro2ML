import torch
import timm
import torch.nn as nn

torch.manual_seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k", pretrained=True, features_only=True)
dummy_input = torch.randn(1, 3, 224, 224)

# Get the output by passing the dummy input through the model
encoder = encoder(dummy_input)

last_feature_shape = encoder[-1].shape[1:]

# Print the shape of the last feature map without the batch size
print(f"Last feature map shape (without batch size): {last_feature_shape}")