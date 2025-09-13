import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, AutoProcessor
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToPILImage


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        c_in = 768
        reduction = 4
        self.adapter = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

        # Freeze CLIP
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x):
        if x.shape[-1] == 768:
            pass
        else:
            x = self.CLIP.get_image_features(pixel_values=x)
        adapter_output = self.adapter(x)
        return adapter_output


class TimeEncoderCapsule(nn.Module):
    def __init__(self, num_classes=24):
        super(TimeEncoderCapsule, self).__init__()
        self.capsule = nn.Sequential(
                                    nn.Linear(num_classes, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(512, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.ReLU(),
                                    nn.Dropout(0.3),
                                    nn.Linear(1024, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.ReLU(),
                                    nn.Dropout(0.3),
                                    nn.Linear(1024, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    )
        self.head = nn.Sequential(nn.Linear(1024, 768))
    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x
    
class TimeEncoder(nn.Module):
    def __init__(self, from_pretrained=True, num_classes=24):
        super(TimeEncoder, self).__init__()
        self.add_module('TimeEnc', TimeEncoderCapsule(num_classes=num_classes))

    def forward(self, one_hot_time):
        time_features = torch.zeros(one_hot_time.shape[0], 768).to(one_hot_time.device)
        #print('location_features:',location_features.shape)
        time_features += self._modules['TimeEnc'](one_hot_time)
        return time_features

class TICL(nn.Module):
    def __init__(self, precomputed_image_feature=False, num_classes=24):
        super().__init__()
        self.logit = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_encoder = ImageEncoder()
        self.time_encoder = TimeEncoder(num_classes=num_classes)
        self.device = "cpu"
        # precomputed features is stored in a h5 file's feature dataset
        self.precomputed_image_feature = precomputed_image_feature

    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.time_encoder.to(device)
        self.logit.data = self.logit.data.to(device)
        return super().to(device)

    def forward(self, image, location):
        """ TimeCLIP's forward pass

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224)
            location (torch.Tensor): GPS location tensor of shape (m, 2)

        Returns:
            logits_per_image (torch.Tensor): Logits per image of shape (n, m)
        """

        # Compute Features
        if self.precomputed_image_feature:
            # load precomputed image features and convert to tensor
            image_features = torch.tensor(image).to(self.device)
            image_features = self.image_encoder(image_features)
        else:
            image_features = self.image_encoder(image)
        time_features = self.time_encoder(location)
        logit = self.logit.exp()
        if len(image_features.shape) == 1:
            image_features = image_features.unsqueeze(0)
        image_features = F.normalize(image_features, dim=1)
        time_features = F.normalize(time_features, dim=1)
        logits_per_image = logit * (image_features @ time_features.t())

        return logits_per_image