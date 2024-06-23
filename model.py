import torch.nn as nn

class CustomClassifier(nn.Module):
    def __init__(self, backbone, num_in_features, num_classes):
        super(CustomClassifier, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_in_features),
            nn.Dropout(0.5),
            nn.Linear(in_features=num_in_features, out_features=512, bias=True),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# # simple classifier
# class CustomClassifier(nn.Module):
#     def __init__(self, backbone, num_in_features, num_classes):
#         super(CustomClassifier, self).__init__()
#         self.backbone = backbone
#         self.classifier = nn.Linear(in_features=num_in_features, out_features=num_classes, bias=True)

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.classifier(x)
#         return x



# # brothernet
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class GRN(nn.Module):
#     """
#     GRN (Global Response Normalization) layer
#     """
#     def __init__(self, dim):
#         super(GRN, self).__init__()
#         self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
#         self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

#     def forward(self, x):
#         Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
#         Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
#         return self.gamma * (x * Nx) + self.beta + x

# class CustomClassifier(nn.Module):
#     def __init__(self, backbone, num_in_features, num_classes):
#         super(CustomClassifier, self).__init__()
#         self.backbone = backbone
#         self.classifier = nn.Sequential(
#             nn.BatchNorm1d(num_in_features),
#             nn.Dropout(0.5),
#             nn.Linear(in_features=num_in_features, out_features=512, bias=True),
#             nn.Linear(512, 512),  # Linear regression replacing the first GELU
#             nn.BatchNorm1d(512),
#             nn.Dropout(0.5),
#             nn.Linear(in_features=512, out_features=512, bias=True),
#             GRN(512),            # GRN layer before the second GELU
#             nn.GELU(),
#             nn.BatchNorm1d(512),
#             nn.Dropout(0.5),
#             nn.Linear(in_features=512, out_features=num_classes, bias=True)
#         )

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.classifier(x)
#         return x