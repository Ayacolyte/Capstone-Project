from torchvision.transforms import v2




# Custom transformation: Convert an RGB tensor to grayscale and drop the channel dimension
import torch
class ToGrayScale(torch.nn.Module):
    def __init__(self):
        super(ToGrayScale, self).__init__()

    def forward(self, img):
        # Use the formula to convert RGB image to grayscale
        r, g, b = img[0], img[1], img[2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray  # Return 2D tensor without channel dimension

normalise = v2.Lambda(lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x)))
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    ToGrayScale(),
    normalise # Normalize the images between 0 and 1
])

# Mirrored and rotated (90 degrees) transformation
mirrored_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True), 
    v2.RandomVerticalFlip(p=1.0), 
    ToGrayScale(),
    normalise
])
rotation_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),  
    v2.RandomRotation((90,90)),  
    ToGrayScale(), 
    normalise
])
