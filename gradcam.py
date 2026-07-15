import torch
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms

from src.model import ConvolutionalNetwork, TransferLearningNetwork

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.forward_handle = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate_heatmap(self, input_tensor, target_class):
        # forward pass
        model_output = self.model(input_tensor)

        self.model.zero_grad()

        #backward pass
        loss = model_output[0][target_class]
        loss.backward()

        #global average pooling
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        #relu
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = torch.clamp(cam, min=0)

        #normalising
        if cam.max() > 0:
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-7)

            return cam.cpu().numpy()
        
def save_visual_result(heatmap, original_img_path, output_path):
    img = cv2.imread(original_img_path)
    height, width, _ = img.shape

    #upsampling
    heatmap_resized = cv2.resize(heatmap, (width, height))

    heatmap_color = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
    
    # 60% original, 40% heatmap
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(output_path, superimposed_img)

if __name__ == '__main__':

    from src.dataset import CLASSES

    model = ConvolutionalNetwork()
    model.load_state_dict(torch.load('src/model_v4.pth', map_location=torch.device('cpu')))
    model.eval()

    print("\n--- Verfügbare Schichten (Named Modules) ---")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"Schicht-Name: model.{name}")
    print("-------------------------------------------\n")
    
    target_layers = {
        "layer1_2": model.conv1_2,
        "layer4_2": model.conv4_2,
        "layer6_2": model.conv6_2
    }
    
    test_image_path = "images/1500.jpg"
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if os.path.exists(test_image_path):
        pil_img = Image.open(test_image_path).convert('RGB')
        input_tensor = preprocess(pil_img).unsqueeze(0)
        input_tensor.requires_grad_()


        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item() * 100
            predicted_name = CLASSES[predicted_idx]
        
        print(f" prediction: {predicted_name} (class {predicted_idx})")
        print(f" confidence: {confidence:.2f}%")
        
        target_class_idx = predicted_idx 
        
        for layer_name, layer_object in target_layers.items():
            try:
                
                cam_extractor = GradCAM(model, layer_object)
                
                heatmap = cam_extractor.generate_heatmap(input_tensor, target_class_idx)
                
                if heatmap is not None:
                    output_filename = f"gradcam_{predicted_name.lower()}_{layer_name}_result.jpg"
                    save_visual_result(heatmap, test_image_path, output_filename)
                    print(f"saved as: {output_filename}")
                
                cam_extractor.remove_hooks()

            except Exception as e:
                print(f"mistake {layer_name}: {e}")