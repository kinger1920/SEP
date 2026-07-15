import torch
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms

from src.model import TransferLearningNetwork

class CAM:
    def __init__(self, model, target_layer, fc_layer):
        self.model = model
        self.target_layer = target_layer
        self.fc_layer = fc_layer
        self.activations = None
        
        self.hook_handle = self.target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def remove_hook(self):
        self.hook_handle.remove()

    def generate_cam(self, input_tensor, target_class):
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        act = self.activations.squeeze(0)
        channels, h, w = act.shape
        
        linear_layer = None
        for module in self.fc_layer.children():
            if isinstance(module, torch.nn.Linear):
                linear_layer = module
                break
                
        if linear_layer is None:
            raise ValueError("Couldnt find the linear layer")
            
        class_weights = linear_layer.weight[target_class].detach()
        
        cam = torch.zeros((h, w), dtype=torch.float32)
        for i in range(channels):
            cam += class_weights[i] * act[i]
            
        cam = torch.clamp(cam, min=0)
        
        if cam.max() > 0:
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-7)
            
        return cam.cpu().numpy()

def save_visual_result(heatmap, original_img_path, output_path):
    img = cv2.imread(original_img_path)
    height, width, _ = img.shape

    # Upsampling
    heatmap_resized = cv2.resize(heatmap, (width, height))

    heatmap_color = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
    
    # 60% Original, 40% Heatmap
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(output_path, superimposed_img)


if __name__ == '__main__':

    from src.dataset import CLASSES

    model = TransferLearningNetwork()
    model.load_state_dict(torch.load('src/best_model_transfer.pth', map_location=torch.device('cpu')))
    
    target_layer = model.backbone.layer4[-1]
    fc_layer = model.backbone.fc
    
    test_image_path = "images/2.jpg"
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if os.path.exists(test_image_path):
        pil_img = Image.open(test_image_path).convert('RGB')
        input_tensor = preprocess(pil_img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item() * 100
            predicted_name = CLASSES[predicted_idx]
        
        print(f"Prediction: {predicted_name} ({confidence:.2f}%)")
        
        cam_extractor = CAM(model, target_layer, fc_layer)
        heatmap = cam_extractor.generate_cam(input_tensor, predicted_idx)
        
        output_filename = f"cam_{predicted_name.lower()}_result.jpg"
        save_visual_result(heatmap, test_image_path, output_filename)
        print(f"saved as: {output_filename}")
        
        cam_extractor.remove_hook()