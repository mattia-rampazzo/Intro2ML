import torch
import timm
import torch.nn as nn

torch.manual_seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k", pretrained=True)
encoder.head = nn.Identity()
encoder = encoder.to(device)
dummy_input = torch.randn(1, 3, 224, 224)  # Example shape for an image input (batch_size, channels, height, width)

#Pass the dummy input through the model to get the feature outputs
with torch.no_grad():
    features = encoder(dummy_input)

last_feature_shape = features[-1].shape
print(f'Last feature shape: {last_feature_shape}')
output = encoder(dummy_input)
print(f"\nOutput shape: {type(output)}") 