import time
import cv2
import torch
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vits' # 'vits' or 'vitb', 'vitl'
dataset = 'vkitti' # 'hypersim' or 'vkitti'
max_depth = 80  # or 80

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(f"Using device: {device}")

# Load model and move to device
model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=device))
model.to(device)
model.eval()

# Load and infer image
raw_img = cv2.imread('/home/kevin-zhou/Desktop/UMich/WeilandLab/Adaptive-Visual-Aid-CV/src/images/Chicago3.JPG')
start = time.time()
depth = model.infer_image(raw_img)  # Returns a numpy array
end = time.time()
print(f"Inference time: {end - start:.2f} seconds")

# Display the depth map
plt.figure(figsize=(10, 8))
plt.imshow(depth, cmap='plasma')
plt.colorbar(label='Depth (meters)')
plt.title("Estimated Depth Map")
plt.axis('off')
plt.show()
