import torch
import onnx
import onnxruntime as ort
from onnxsim import simplify
from depth_anything_v2.dpt import DepthAnythingV2

# Configuration
encoder = 'vitb'   # Choose from 'vits', 'vitb', 'vitl'
dataset = 'vkitti'   # Choose from 'hypersim', 'vkitti'
max_depth = 80   # Choose 20 or 80
onnx_output_path = f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.onnx'

# Model config dictionary
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
print("Loading model...")
model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
checkpoint_path = f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# Dummy input for export
dummy_input = torch.randn(1, 3, 448, 672).to(device)

# Export to ONNX
print(f"Exporting model to {onnx_output_path}...")
torch.onnx.export(
    model,
    dummy_input,
    onnx_output_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    }
)
print("Export complete.")

# Simplify ONNX
print("Simplifying ONNX model...")
onnx_model = onnx.load(onnx_output_path)
model_simp, check = simplify(onnx_model)
if not check:
    raise RuntimeError("Simplified ONNX model could not be validated.")
onnx.save(model_simp, onnx_output_path)
print("Simplification complete.")

# Validate ONNX with ONNX Runtime
print("Validating ONNX model...")
ort_session = ort.InferenceSession(onnx_output_path)
outputs = ort_session.run(None, {"input": dummy_input.cpu().numpy()})
print("ONNX inference successful, output shape:", outputs[0].shape)
