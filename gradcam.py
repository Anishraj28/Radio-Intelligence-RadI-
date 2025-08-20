import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# To use mac gpu
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

IMG_SIZE = 900

class CenterCropPercent:
    def __init__(self, percent=0.2):
        self.percent = percent

    def __call__(self, img):
        w, h = img.size
        dx, dy = int(w * self.percent), int(h * self.percent)
        # Crop box: (left, top, right, bottom)
        return img.crop((dx, dy, w - dx, h - dy))

# Model Class
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Adaptive Pooling so we can have heatmap work properly
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Output size: 4x4
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = F.max_pool2d(X, 2, 2)
        X = F.dropout(X, 0.2, training=self.training)

        X = F.relu(self.bn2(self.conv2(X)))
        X = F.max_pool2d(X, 2, 2)

        X = F.relu(self.bn3(self.conv3(X)))
        X = F.max_pool2d(X, 2, 2)

        X = F.relu(self.bn4(self.conv4(X)))
        
        # This preserves some spatial structure (4x4 instead of 1x1)
        X = self.adaptive_pool(X)
        
        X = X.view(X.size(0), -1)  # Flatten: [batch, 256*4*4]

        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = F.relu(self.fc2(X))
        X = self.dropout(X)
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return X


class ImprovedGradCAM:
    """
    Improved Grad-CAM implementation with better debugging and multiple layer support
    """
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.target_layer = None
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Find the target layer
        for name, module in model.named_modules():
            if name == target_layer_name:
                self.target_layer = module
                break
        
        if self.target_layer is None:
            raise ValueError(f"Layer '{target_layer_name}' not found in model")
        
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach().clone()  # Add .clone()
            print(f"Captured activations shape: {output.shape}")  # Debug

        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach().clone()  # Add .clone()
                print(f"Captured gradients shape: {grad_output[0].shape}")  # Debug
        
        # Register hooks
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        
        self.hooks.extend([forward_handle, backward_handle])
    
    def generate_heatmap(self, input_tensor, target_class=None, use_guided=False):
        
        #Generate Grad-CAM heatmap
        
        
        self.gradients = None
        self.activations = None
   
        # Ensure input requires gradients
        input_tensor = input_tensor.clone()
        input_tensor.requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_score = output[0, target_class]
        class_score.backward()
        
        # Check if we captured gradients and activations
        if self.gradients is None:
            print("Warning: No gradients captured!")
            return None
        
        if self.activations is None:
            print("Warning: No activations captured!")
            return None
        
        # Generate heatmap using Grad-CAM formula
        grads = self.gradients  # Shape: [1, channels, H, W]
        acts = self.activations  # Shape: [1, channels, H, W]
        
        # Global Average Pooling of gradients (importance weights)
        weights = torch.mean(grads, dim=(2, 3))  # Shape: [1, channels]
        
        # Weighted combination of activation maps
        heatmap = torch.zeros(acts.shape[2:], device=acts.device)  # Shape: [H, W]
        
        for i in range(weights.shape[1]):
            heatmap += weights[0, i] * acts[0, i]
        
        # Apply ReLU to remove negative influences
        heatmap = F.relu(heatmap)
        
        # Normalize heatmap to [0, 1]
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap.cpu().numpy()
    
    def cleanup(self):
        """Remove hooks to prevent memory leaks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def predict_with_gradcam(img_path, model, device, target_layers=None, save_results=False):
    """
    Complete prediction function with Grad-CAM visualization
    
    Args:
        img_path: Path to the input image
        model: Trained PyTorch model
        device: Device to run inference on
        target_layers: List of layer names to visualize (if None, uses default layers)
        save_results: Whether to save the visualization
    
    Returns:
        prediction: Predicted class name
        confidence: Prediction confidence
        results: Dictionary containing heatmaps for different layers
    """
    
    # Default layers to visualize
    if target_layers is None:
        target_layers = ['conv3', 'conv2', 'conv1']
    
    # Test transform (same as training)
    test_transform = transforms.Compose([
        CenterCropPercent(0.2),  # remove 10% border
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    # Load and preprocess image
    try:
        original_image = Image.open(img_path)
        if original_image.mode != 'L':
            original_image = original_image.convert('L')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None
    
    input_tensor = test_transform(original_image).unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1)
        confidence = probabilities[0][predicted_class].item() * 100
    
    # Class names
    classes = ['NORMAL', 'PNEUMONIA']
    prediction = classes[predicted_class.item()]
    
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Raw output: {output[0].detach().cpu().numpy()}")
    
    # Generate heatmaps for different layers
    results = {}
    heatmaps = {}
    
    for layer_name in target_layers:
        print(f"\nGenerating Grad-CAM for layer: {layer_name}")
        
        try:
            # Create Grad-CAM instance
            cam = ImprovedGradCAM(model, layer_name)
            
            # Generate heatmap
            heatmap = cam.generate_heatmap(input_tensor.clone(), predicted_class.item())
            
            if heatmap is not None:
                heatmaps[layer_name] = heatmap
                print(f"✓ Successfully generated heatmap for {layer_name}")
                print(f"  Heatmap shape: {heatmap.shape}")
                print(f"  Heatmap range: {heatmap.min():.4f} to {heatmap.max():.4f}")
            else:
                print(f"✗ Failed to generate heatmap for {layer_name}")
            
            # Clean up hooks
            cam.cleanup()
            
        except Exception as e:
            print(f"✗ Error with layer {layer_name}: {e}")
    
    # Visualization
    if heatmaps:
        num_layers = len(heatmaps)
        fig, axes = plt.subplots(2, num_layers + 1, figsize=(5 * (num_layers + 1), 10))
        
        if num_layers == 0:  # Handle case with no successful heatmaps
            return prediction, confidence, results
        
        # Original image in both rows
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original X-ray')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(original_image, cmap='gray')
        axes[1, 0].set_title('Original X-ray')
        axes[1, 0].axis('off')
        
        # Heatmaps and overlays
        for idx, (layer_name, heatmap) in enumerate(heatmaps.items()):
            col_idx = idx + 1
            
            # Top row: Pure heatmaps
            im1 = axes[0, col_idx].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
            axes[0, col_idx].set_title(f'{layer_name.upper()}\nGrad-CAM Heatmap')
            axes[0, col_idx].axis('off')
            plt.colorbar(im1, ax=axes[0, col_idx], fraction=0.046, pad=0.04)
            
            # Bottom row: Overlays
            original_np = np.array(original_image)
            heatmap_resized = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
            
            axes[1, col_idx].imshow(original_np, cmap='gray', alpha=0.7)
            axes[1, col_idx].imshow(heatmap_resized, cmap='jet', alpha=0.4, vmin=0, vmax=1)
            axes[1, col_idx].set_title(f'{layer_name.upper()}\nOverlay')
            axes[1, col_idx].axis('off')
        
        # Add overall title
        fig.suptitle(f'Prediction: {prediction} (Confidence: {confidence:.1f}%)', 
                    fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        if save_results:
            plt.savefig(f'gradcam_analysis_{prediction.lower()}_{confidence:.1f}pct.png', 
                       dpi=300, bbox_inches='tight')
            print(f"Results saved to gradcam_analysis_{prediction.lower()}_{confidence:.1f}pct.png")
        
        plt.show()
    
    else:
        print("No heatmaps were successfully generated.")
        
        # Try a simple debug visualization
        print("\nTrying simple gradient visualization...")
        try:
            input_tensor_debug = input_tensor.clone()
            input_tensor_debug.requires_grad_(True)
            
            output_debug = model(input_tensor_debug)
            model.zero_grad()
            output_debug[0, predicted_class].backward()
            
            if input_tensor_debug.grad is not None:
                input_grad = torch.abs(input_tensor_debug.grad[0, 0]).cpu().numpy()
                
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.imshow(original_image, cmap='gray')
                plt.title('Original X-ray')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(input_grad, cmap='hot')
                plt.title('Input Gradients\n(Simple Saliency)')
                plt.colorbar()
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(original_image, cmap='gray', alpha=0.7)
                plt.imshow(input_grad, cmap='hot', alpha=0.3)
                plt.title(f'Overlay\n{prediction} ({confidence:.1f}%)')
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
                
                print("✓ Simple gradient visualization completed")
            else:
                print("✗ No gradients found even for input")
                
        except Exception as e:
            print(f"✗ Debug visualization failed: {e}")
    
    results = {
        'prediction': prediction,
        'confidence': confidence,
        'heatmaps': heatmaps,
        'predicted_class_idx': predicted_class.item()
    }
    
    return prediction, confidence, results

def debug_model_layers(model):
    """Print all layers in the model for debugging"""
    print("Available layers in the model:")
    print("-" * 50)
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            print(f"{name}: {module}")
    print("-" * 50)

def analyze_batch_gradcam(model, data_loader, device, num_samples=3, target_layers=None):
    """
    Analyze multiple samples from a data loader with Grad-CAM
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader containing test samples
        device: Device to run inference on
        num_samples: Number of samples to analyze
        target_layers: List of layer names to visualize
    """
    if target_layers is None:
        target_layers = ['conv3']
    
    model.eval()
    classes = ['NORMAL', 'PNEUMONIA']
    
    samples_processed = 0
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        if samples_processed >= num_samples:
            break
            
        images = images.to(device)
        labels = labels.to(device)
        
        for i in range(min(len(images), num_samples - samples_processed)):
            print(f"\n{'='*60}")
            print(f"Sample {samples_processed + 1}")
            print(f"True label: {classes[labels[i].item()]}")
            
            # Single image tensor
            single_image = images[i:i+1]
            
            # Get prediction
            with torch.no_grad():
                output = model(single_image)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1)
                confidence = probabilities[0][predicted_class].item() * 100
            
            prediction = classes[predicted_class.item()]
            print(f"Predicted: {prediction} (Confidence: {confidence:.2f}%)")
            
            # Generate heatmaps
            heatmaps = {}
            
            for layer_name in target_layers:
                try:
                    cam = ImprovedGradCAM(model, layer_name)
                    heatmap = cam.generate_heatmap(single_image.clone(), predicted_class.item())
                    
                    if heatmap is not None:
                        heatmaps[layer_name] = heatmap
                    
                    cam.cleanup()
                    
                except Exception as e:
                    print(f"Error generating heatmap for {layer_name}: {e}")
            
            # Visualize
            if heatmaps:
                fig, axes = plt.subplots(1, len(heatmaps) + 1, figsize=(5 * (len(heatmaps) + 1), 4))
                if len(heatmaps) == 0:
                    axes = [axes]
                elif len(heatmaps) == 1:
                    axes = list(axes)
                
                # Original image
                original_np = single_image[0, 0].cpu().numpy()
                axes[0].imshow(original_np, cmap='gray')
                axes[0].set_title(f'Original\nTrue: {classes[labels[i].item()]}')
                axes[0].axis('off')
                
                # Heatmaps
                for idx, (layer_name, heatmap) in enumerate(heatmaps.items()):
                    axes[idx + 1].imshow(original_np, cmap='gray', alpha=0.7)
                    heatmap_resized = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
                    axes[idx + 1].imshow(heatmap_resized, cmap='jet', alpha=0.4)
                    axes[idx + 1].set_title(f'{layer_name}\n{prediction} ({confidence:.1f}%)')
                    axes[idx + 1].axis('off')
                
                plt.tight_layout()
                plt.show()
            
            samples_processed += 1
            
            if samples_processed >= num_samples:
                break

# Simple usage for testing a single image:
def test_single_image(img_path, model, device):
    """
    Simple function to test a single image with Grad-CAM
    """
    prediction, confidence, results = predict_with_gradcam(
        img_path=img_path,
        model=model,
        device=device,
        target_layers=['conv3'],  # Just use the best layer
        save_results=False
    )
    return prediction, confidence

# Usage examples:

# Load your model
model = ConvolutionalNetwork().to(device)
model.load_state_dict(torch.load('trained_xray_model.pth', weights_only=True))

# Test a single image - SIMPLEST WAY:
prediction, confidence = test_single_image(
    "'/Users/anishrajumapathy/Downloads/chest_xray/test/PNEUMONIA/Pneumonia-Viral (1060).jpg'",
    model, 
    device
)
"""
# OR if you want more detailed analysis with multiple layers:
prediction, confidence, results = predict_with_gradcam(
    img_path="/Users/anishrajumapathy/Downloads/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg",
    model=model,
    device=device,
    target_layers=['conv4', 'conv3'],  # Compare multiple layers
    save_results=True  # Save the visualization
)
"""