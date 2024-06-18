import torch.nn as nn

# simple classifier
class CustomClassifier(nn.Module):
    def __init__(self, backbone, num_in_features, num_classes):
        super(CustomClassifier, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(in_features=num_in_features, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# class CustomClassifier(nn.Module):
#     def __init__(self, backbone, num_in_features, num_classes):
#         super(CustomClassifier, self).__init__()
#         self.backbone = backbone
#         self.classifier = nn.Sequential(
#             nn.BatchNorm1d(num_in_features),
#             nn.Dropout(0.5),
#             nn.Linear(in_features=num_in_features, out_features=512, bias=True),
#             nn.GELU(),
#             nn.BatchNorm1d(512),
#             nn.Dropout(0.5),
#             nn.Linear(in_features=512, out_features=512, bias=True),
#             nn.GELU(),
#             nn.BatchNorm1d(512),
#             nn.Dropout(0.5),
#             nn.Linear(in_features=512, out_features=num_classes, bias=True)
#         )

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.classifier(x)
#         return x