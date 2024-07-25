# model.py
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import io
import os
import torch.nn as nn

from blocks import CNNBlock, ResidualBlock


# Define Generator class 
class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                CNNBlock(
                    num_features, 
                    num_features * 2, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1
                ),
                CNNBlock(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                CNNBlock(
                    num_features * 4,
                    num_features * 2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                CNNBlock(
                    num_features * 2,
                    num_features * 1,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        self.last = nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))

# Define your image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'models/serving_model_v0.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    G_Photo2Monet = Generator(img_channels=3).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    G_Photo2Monet.load_state_dict(checkpoint['G_Photo2Monet'])
    G_Photo2Monet.eval()  # Set model to evaluation mode
    
    return G_Photo2Monet, device

def process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = ImageOps.exif_transpose(image)  # Correct the orientation
    return transform(image).unsqueeze(0)  # Add batch dimension

def denormalize(tensor):
    # tensor = tensor * torch.tensor([0.5, 0.5, 0.5]) + torch.tensor([0.5, 0.5, 0.5])
    tensor = tensor * torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1) + torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    tensor = tensor.clamp(0, 1)
    return transforms.ToPILImage()(tensor.squeeze(0))  # Remove batch dimension
