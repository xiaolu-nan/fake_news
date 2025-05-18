# gradcam_infer.py

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from model import DeepfakeDetector

# Step 1: 加载模型与权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeDetector().to(device)
model.load_state_dict(torch.load("deepfake_detector.pth", map_location=device))
model.eval()

# Step 2: 设置 Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

target_layer = model.backbone._blocks[-1]  # 最后一层特征层

cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=None)

# Step 3: 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def run_gradcam(image_path):
    raw_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(raw_image).unsqueeze(0).to(device)
    rgb_img = np.array(raw_image.resize((224, 224))) / 255.0

    # 推理与可视化
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 保存图像代替显示
    cv2.imwrite("gradcam_result.jpg", visualization)
    print("Grad-CAM 可视化结果已保存为 gradcam_result.jpg")


# Step 4: 测试
run_gradcam("deepfake.png")
