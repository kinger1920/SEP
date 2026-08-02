import torch
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.model import ConvolutionalNetwork, TransferLearningNetwork

class OcclusionSensitivity:
    def __init__(self, model, patch_size=32, stride=16):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride

    def generate_heatmap(self, input_tensor, target_class):
        self.model.eval()
        
        _, _, h, w = input_tensor.shape
        
        # Baseline
        with torch.no_grad():
            base_output = self.model(input_tensor)
            base_prob = torch.softmax(base_output, dim=1)[0, target_class].item()

        h_out = int((h - self.patch_size) / self.stride) + 1
        w_out = int((w - self.patch_size) / self.stride) + 1
        heatmap = np.zeros((h_out, w_out))

        
        with torch.no_grad():
            for i, y in enumerate(tqdm(range(0, h - self.patch_size + 1, self.stride))):
                for j, x in enumerate(range(0, w - self.patch_size + 1, self.stride)):
                    
                    occluded_img = input_tensor.clone()
                    occluded_img[:, :, y:y+self.patch_size, x:x+self.patch_size] = 0.0
                    
                    out = self.model(occluded_img)
                    prob = torch.softmax(out, dim=1)[0, target_class].item()
                    
                    drop_in_prob = base_prob - prob
                    heatmap[i, j] = drop_in_prob

        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            
        return heatmap

def save_visual_result(heatmap, original_img_path, output_path):
    img = cv2.imread(original_img_path)
    height, width, _ = img.shape

    # Upsampling
    heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)

    heatmap_color = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
    
    # 60% Original, 40% Heatmap
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(output_path, superimposed_img)

if __name__ == '__main__':

    from src.dataset import CLASSES

    model = ConvolutionalNetwork()
    model.load_state_dict(torch.load('src/model_v4.pth', map_location=torch.device('cpu')))
    model.eval()
    
    test_image_path = "images/1500.jpg"
    
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
        
        print(f"Prediction: {predicted_name} (class {predicted_idx})")
        print(f"Confidence: {confidence:.2f}%")
        
        occlusion_extractor = OcclusionSensitivity(model, patch_size=32, stride=16)
        heatmap = occlusion_extractor.generate_heatmap(input_tensor, predicted_idx)
        
        output_filename = f"occlusion_{predicted_name.lower()}_result.jpg"
        save_visual_result(heatmap, test_image_path, output_filename)
        print(f" saves as: {output_filename}")