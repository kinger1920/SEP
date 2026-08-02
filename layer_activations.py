import torch
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms

from src.Scanner.cat_dog_filter1 import crop_cat_or_dog_segmented
from src.model import ConvolutionalNetwork

class LayerActivations:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        
        self.hook_handle = self.target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def remove_hook(self):
        self.hook_handle.remove()

    def get_activations(self, input_tensor):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        mean_activations = torch.mean(self.activations, dim=1).squeeze(0)
        
        act_max = mean_activations.max()
        act_min = mean_activations.min()
        if act_max > act_min:
            mean_activations = (mean_activations - act_min) / (act_max - act_min)
            
        return mean_activations.cpu().numpy()

def save_visual_result(activation_map, base_img_pil, output_path):
    if not isinstance(base_img_pil, Image.Image):
        if isinstance(base_img_pil, np.ndarray):
             base_img_pil = Image.fromarray(base_img_pil.astype('uint8'), 'RGB')
        else:
             raise TypeError("base_img_pil needs to be pil or numpy")


    img_np = np.array(base_img_pil, dtype=np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    height, width, _ = img_bgr.shape

    # Upsampling
    activation_resized = cv2.resize(activation_map, (width, height))

    activation_color = np.uint8(255 * activation_resized)
    activation_color = cv2.applyColorMap(activation_color, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(img_bgr, 0.5, activation_color, 0.5, 0)
    cv2.imwrite(output_path, superimposed_img)


if __name__ == '__main__':

    from src.dataset import CLASSES

    model = ConvolutionalNetwork()
    model.load_state_dict(torch.load('src/model_v4.pth', map_location=torch.device('cpu')))
    model.eval()
    
    target_layers = {
        "layer2": model.conv2_2,
        "layer4": model.conv4_2,
        "layer6": model.conv6_2
    }
    
    test_image_path = "images/1500.jpg"
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if os.path.exists(test_image_path):
        pil_img = Image.open(test_image_path).convert('RGB')

        cropped_img = crop_cat_or_dog_segmented(pil_img)
        
        if cropped_img is None:
            print("Fehler: YOLO konnte kein Tier finden.")
        else:
            input_tensor = preprocess(cropped_img).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                predicted_idx = torch.argmax(output, dim=1).item()
                predicted_name = CLASSES[predicted_idx]
        
        print(f"Prediction: {predicted_name}")
        
        for layer_name, layer_object in target_layers.items():
            try:
                
                extractor = LayerActivations(model, layer_object)
                activation_map = extractor.get_activations(input_tensor)
                
                output_filename = f"activation_{predicted_name.lower()}_{layer_name}_result.jpg"
                save_visual_result(activation_map, cropped_img, output_filename)
                print(f" saved as: {output_filename}")
                
                extractor.remove_hook()
                
            except Exception as e:
                print(f"mistake {layer_name}: {e}")