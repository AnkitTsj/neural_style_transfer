# Neural style transfer script
import torch
import torch.nn.functional as F
from torchvision import models
from PIL import Image
# from google.colab import drive # Not using Colab for now
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load all the images we'll need
content_1 = Image.open("images/tree.jpg")
content_2 = Image.open("images/trees.jpg")
style_image = Image.open("images/style.jpg")
bro_c = Image.open("images/bro_1.jpg")
bro_cps = Image.open("images/bro_31.jpeg")
bro_s = Image.open("images/bro_2.jpg")

# Class to extract features from specific layers
class Hooker:
   def __init__(self, model, layers):
       self.model = model
       self.layers = layers
       self.outputs = {}
       self._register_hooks()

   # Set up hooks for each layer we want
   def _register_hooks(self):
       for layer_id in self.layers:
           layer = self.model[layer_id]
           layer.register_forward_hook(self.get_hook(layer_id))

   # Hook function to save outputs
   def get_hook(self, layer_id):
       def hook(module, input, output):
           self.outputs[layer_id] = output
       return hook

   # Run image through model and get features
   def extract_feats(self, image):
       self.outputs = {}  # Clear previous outputs
       image = image.to(device)
       self.model(image)
       return self.outputs

# Get image ready for the model
def preprocess_image(image, imsize=256):
   # image = Image.open(image_path).convert('RGB')
   image = image.resize((imsize, imsize))
   image = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])(image)
   return image.unsqueeze(0)

# Convert model output back to viewable image
def postprocess_image(tensor):
   tensor = tensor.squeeze(0)
   tensor = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                 std=[1/0.229, 1/0.224, 1/0.225])(tensor)
   tensor = tensor.clamp(0, 1)
   image = transforms.ToPILImage()(tensor)
   return image