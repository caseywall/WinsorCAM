import io
import os
import random
import numpy as np
from PIL import Image
import ipywidgets as widgets
from ipywidgets import VBox, Layout, HTML, Image as WImage, IntSlider, IntRangeSlider, ToggleButtons, FloatSlider, Dropdown
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torch_models
from matplotlib import pyplot as plt
from skimage import filters
# get display module from IPython
from torchvision.models import resnet50, densenet121, vgg16, inception_v3, efficientnet_b0, convnext_tiny
from winsorcam import make_winsorcam_model

seed = 42
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDNN_DETERMINISTIC"] = "1"
os.environ["PYTHONHASHSEED"] = str(seed)


# make a function to denormalize the image
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = tensor * std + mean
    return tensor

def create_colored_heatmap(matrix, size=(224, 224), colormap='Reds', op_multiplier=1.0, interpolation_mode='nearest-exact'):
    """
    Create a colored heatmap from a 2D matrix or torch tensor.

    Args:
        matrix (np.ndarray or torch.Tensor): 2D array or tensor to visualize.
        size (tuple): Output size (height, width).
        colormap (str): Name of matplotlib colormap.
        op_multiplier (float): Opacity multiplier for the alpha channel.
        interpolation_mode (str): Interpolation mode for resizing (if torch.Tensor).

    Returns:
        np.ndarray: RGBA heatmap image of shape (H, W, 4).
    """
    # Interpolate if needed
    if isinstance(matrix, torch.Tensor) and matrix.shape != size:
        matrix = F.interpolate(matrix[None, None], size=size, mode=interpolation_mode)
        matrix = matrix.squeeze().cpu().numpy()

    # Normalize the matrix
    normalized_matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min()) if matrix.max() > matrix.min() else matrix

    # Apply colormap
    cmap = plt.colormaps.get_cmap(colormap)
    colored_matrix = cmap(normalized_matrix)[..., :3]

    # Reduce the opacity of the heatmap using the op_multiplier
    alpha_matrix = normalized_matrix * op_multiplier

    # Add alpha channel
    return np.concatenate([colored_matrix, alpha_matrix[..., None]], axis=-1)

def generate_masked_image(image, mask):
    """
    Generate a masked image by applying Otsu thresholding to the mask and resizing it to match the image size.
    Args:
        image (np.ndarray): The input image of shape (H, W, 3).
        mask (torch.Tensor or np.ndarray): The mask to apply, shape (h, w).
    Returns:
        np.ndarray: The masked image of shape (H, W, 3).
    """
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    # Apply Otsu thresholding to get binary mask
    threshold = filters.threshold_otsu(mask)
    binary_mask = mask > threshold

    # Resize binary mask to match image size using nearest-exact interpolation 
    binary_mask = F.interpolate(torch.tensor(binary_mask)[None, None].float(), 
                            size=image.shape[:2], 
                            mode="nearest-exact").squeeze().numpy()

    # Apply mask to image
    masked_image = image.copy()
    for i in range(3):
        masked_image[..., i] = masked_image[..., i] * binary_mask
        
    return masked_image

def generate_masked_image_set(image, mask, threshold):
    """
    Generate a masked image by applying a threshold to the mask and resizing it to match the image size.
    Args:
        image (np.ndarray): The input image of shape (H, W, 3).
        mask (torch.Tensor or np.ndarray): The mask to apply, shape (h, w).
        threshold (float): The threshold value to apply.
    Returns:
        np.ndarray: The masked image of shape (H, W, 3).
    """
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    # Apply threshold to get binary mask
    binary_mask = mask > threshold

    # Resize binary mask to match image size using nearest-exact interpolation 
    binary_mask = F.interpolate(torch.tensor(binary_mask)[None, None].float(), 
                            size=image.shape[:2], 
                            mode="nearest-exact").squeeze().numpy()

    # Apply mask to image
    masked_image = image.copy()
    for i in range(3):
        masked_image[..., i] = masked_image[..., i] * binary_mask
        
    return masked_image

def load_model_weights(model, model_name, device='cpu'):
    """
    Load pretrained weights for a model.
    
    Args:
        model: The model to load weights into
        model_name (str): Model architecture name
        device (str): Device to load weights on
    
    Returns:
        model: Model with loaded weights
    """
    # Model weight paths
    weight_paths = {
        'resnet50': './model_files/best_resnet50_model.pth',
        'densenet121': './model_files/best_dense121_model.pth',
        'vgg16': './model_files/best_vgg16_model.pth',
        'inception_v3': './model_files/best_inception_v3_model.pth',
        'efficientnet_b0': './model_files/best_efficientnet_b0_higher_lr.pth',
        'convnext_tiny': './model_files/best_convnext_tiny_norm_lr.pth'
    }
    
    weight_path = weight_paths.get(model_name)
    if weight_path and os.path.exists(weight_path):
        if device == 'cpu':
            model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(weight_path))
            # print(f"Loaded {model_name} model onto GPU")
    else:
        print(f"Warning: Weight file not found for {model_name} at {weight_path}")
    
    # For Inception v3, replace AuxLogits with Identity AFTER loading weights
    # This prevents AttributeError in FullGrad while still loading the trained weights
    # We use Identity instead of deleting to avoid breaking any code that expects the attribute
    if model_name == 'inception_v3' and hasattr(model, 'AuxLogits'):
        model.AuxLogits = nn.Identity()
        # print("Replaced AuxLogits module with Identity (after loading weights)")
    
    return model

def create_model_with_custom_head(model_name, num_classes=20, weights='IMAGENET1K_V1'):
    """
    Create a model with custom classification head for Pascal VOC.
    
    Args:
        model_name (str): One of ['resnet50', 'densenet121', 'vgg16', 'inception_v3', 'efficientnet_b0', 'convnext_tiny']
        num_classes (int): Number of output classes (default: 20 for Pascal VOC)
        weights (str): Pretrained weights to use
    
    Returns:
        model: Model with custom head
    """
    if model_name == 'resnet50':
        model = resnet50(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, num_classes)
        )
        
    elif model_name == 'densenet121':
        model = densenet121(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )
        
    elif model_name == 'vgg16':
        model = vgg16(weights=weights)
        model.classifier[6] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(4096, num_classes)
        )
        
    elif model_name == 'inception_v3':
        # Load with aux_logits=True (required by pretrained weights)
        # Keep AuxLogits for now - will be removed after loading weights
        model = inception_v3(weights=weights, aux_logits=True)
        model.aux_logits = False  # Disable aux outputs during forward pass
        # Modify both main classifier and auxiliary classifier to match training (20 classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'convnext_tiny':
        model = convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(in_features=768, out_features=num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def find_heatmap_centroid(heatmap):
    """
    Find the centroid of a heatmap based on pixel intensity values.
    
    Args:
        heatmap (torch.Tensor or np.ndarray): 2D array of intensity values between 0 and 1
        
    Returns:
        tuple: (x, y) coordinates of the centroid
    """
    
    # Convert to numpy array if it's a torch tensor
    if torch.is_tensor(heatmap):
        heatmap = heatmap.cpu().numpy()
    
    # normalize the heatmap between 0 and 1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    
    # Create coordinate grids
    h, w = heatmap.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Calculate total weight (sum of all intensity values)
    total_weight = np.sum(heatmap)
    
    # Avoid division by zero
    if total_weight == 0:
        return (w // 2, h // 2)  # Return center of image if all values are zero
    
    # Calculate weighted averages
    x_centroid = np.sum(x_coords * heatmap) / total_weight
    y_centroid = np.sum(y_coords * heatmap) / total_weight
    
    return int(x_centroid), int(y_centroid)

def generate_masked_image(image, mask):
    """
    Generate a masked image by applying Otsu thresholding to the mask and resizing it to match the image size.
    Args:
        image (np.ndarray): The input image of shape (H, W, 3).
        mask (torch.Tensor or np.ndarray): The mask to apply, shape (h, w).
    Returns:
        np.ndarray: The masked image of shape (H, W, 3).
    """
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    # Apply Otsu thresholding to get binary mask
    threshold = filters.threshold_otsu(mask)
    binary_mask = mask > threshold

    # Resize binary mask to match image size using nearest-exact interpolation 
    binary_mask = F.interpolate(torch.tensor(binary_mask)[None, None].float(), 
                            size=image.shape[:2], 
                            mode="nearest-exact").squeeze().numpy()

    # Apply mask to image
    masked_image = image.copy()
    for i in range(3):
        masked_image[..., i] = masked_image[..., i] * binary_mask
        
    return masked_image

def generate_masked_image_set(image, mask, threshold):
    """
    Generate a masked image by applying Otsu thresholding to the mask and resizing it to match the image size.
    Args:
        image (np.ndarray): The input image of shape (H, W, 3).
        mask (torch.Tensor or np.ndarray): The mask to apply, shape (h, w).
    Returns:
        np.ndarray: The masked image of shape (H, W, 3).
    """
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    # Apply Otsu thresholding to get binary mask
    binary_mask = mask > threshold

    # Resize binary mask to match image size using nearest-exact interpolation 
    binary_mask = F.interpolate(torch.tensor(binary_mask)[None, None].float(), 
                            size=image.shape[:2], 
                            mode="nearest-exact").squeeze().numpy()

    # Apply mask to image
    masked_image = image.copy()
    for i in range(3):
        masked_image[..., i] = masked_image[..., i] * binary_mask
        
    return masked_image

def make_binary_mask(mask):
    """
    Convert a mask to a binary mask using Otsu thresholding.

    Args:
        mask (torch.Tensor or np.ndarray): The mask to binarize.

    Returns:
        np.ndarray: Binary mask of the same shape as input, dtype uint8.
    """
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    # Apply Otsu thresholding to get binary mask
    threshold = filters.threshold_otsu(mask)
    binary_mask = mask > threshold
    return binary_mask.astype(np.uint8)

def calculate_iou(winsor_gradcam, label, input_tensor, interpolation_mode):
    """
    Calculate the Intersection over Union (IoU) between a predicted mask and a ground truth mask.

    Args:
        winsor_gradcam (torch.Tensor): The predicted mask (2D tensor).
        label (torch.Tensor): The ground truth label mask (C, H, W) or (H, W, C).
        input_tensor (torch.Tensor): The input image tensor (B, C, H, W) or (C, H, W).
        interpolation_mode (str): Interpolation mode for resizing.

    Returns:
        float: IoU score.
    """
    thing_to_mask = winsor_gradcam
    
    # Determine target size from input_tensor
    if input_tensor.ndim == 4:  # (B, C, H, W)
        target_size = input_tensor.shape[2:]
    else:  # (C, H, W)
        target_size = input_tensor.shape[1:]
    
    # Resize mask if needed
    if thing_to_mask.shape != target_size:
        thing_to_mask = F.interpolate(thing_to_mask.unsqueeze(0).unsqueeze(0), 
                                     size=target_size, 
                                     mode=interpolation_mode).squeeze()
    
    binary_mask = make_binary_mask(thing_to_mask.squeeze())

    # Process target to create proper binary mask
    # Check if label is (H, W, C) or (C, H, W)
    if label.shape[-1] == 3 or label.shape[-1] == 1:
        # Label is already (H, W, C)
        target = label
    else:
        # Label is (C, H, W), permute to (H, W, C)
        target = label.permute(1, 2, 0)

    # Check if there are any non-zero pixels in the target
    # This will create a binary mask where any non-black pixel becomes 1
    if target.shape[-1] == 3:
        target_binary = torch.where(
            (target[:,:,0] > 0) | (target[:,:,1] > 0) | (target[:,:,2] > 0),
            torch.tensor(1, dtype=torch.uint8),
            torch.tensor(0, dtype=torch.uint8)
        )
    else:
        target_binary = (target.squeeze() > 0).to(torch.uint8)
    
    # target_binary is already (H, W), no need to squeeze
    narrowed_target = target_binary.to(torch.uint8)
    binary_mask = torch.from_numpy(binary_mask).to(torch.uint8)

    # Calculate intersection over union
    intersection = torch.logical_and(binary_mask, narrowed_target)
    union = torch.logical_or(binary_mask, narrowed_target)
    iou = intersection.sum() / union.sum() if union.sum() > 0 else 0
    return iou.item()

def calculate_centroid_distance(mask1, centroid2):
    """
    Calculate the Euclidean distance between the centroid of a mask and a given centroid.

    Args:
        mask1 (torch.Tensor or np.ndarray): The mask to compute centroid from.
        centroid2 (tuple or np.ndarray): The (x, y) coordinates of the second centroid.

    Returns:
        float: Euclidean distance between centroids.
    """
    if torch.is_tensor(mask1):
        mask1 = mask1.cpu().numpy()
    centroid1 = find_heatmap_centroid(mask1)
    return np.linalg.norm(np.array(centroid1) - np.array(centroid2))

def visualize_mask_iou(mask, gt_mask):
    """
    Visualize the Intersection over Union (IoU) between a predicted mask and a ground truth mask.
    Args:
        mask (torch.Tensor or np.ndarray): The predicted mask (2D tensor).
        gt_mask (torch.Tensor or np.ndarray): The ground truth mask (2D tensor).
    Returns:
        np.ndarray: Colored image showing IoU visualization.
    """
    mask = torch.as_tensor(mask).bool().cpu()
    gt_mask = torch.as_tensor(gt_mask).bool().cpu()
    intersection = mask & gt_mask
    false_positive = mask & ~gt_mask
    false_negative = ~mask & gt_mask
    colored_image = np.zeros((*mask.shape, 3), dtype=np.float32)
    colored_image[intersection] = [0, 1, 0]      # Green
    colored_image[false_positive] = [1, 0, 0]    # Red
    colored_image[false_negative] = [0, 0, 1]    # Blue
    return colored_image

def set_deterministic(cpu_only=False, enable=True):
    """
    Set or unset deterministic algorithms for reproducibility.
    
    Args:
        cpu_only: If True, only set CPU-related deterministic settings
        enable: If True, enable deterministic algorithms. If False, disable them.
    """
    # Set all seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Control deterministic algorithms
    torch.use_deterministic_algorithms(enable)
    
    # Control CPU threading (important for CPU determinism)
    if enable:
        torch.set_num_threads(1)  # Single thread is most deterministic
    
    # GPU-specific settings (skip if CPU only)
    if not cpu_only:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if enable:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["CUDNN_DETERMINISTIC"] = "1"
            os.environ["PYTHONHASHSEED"] = str(seed)
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

def create_winsorgradcam_model(model_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_classes=20, cpu_only_flag=True):
    """
    Create a WinsorGradCAM-wrapped model for the specified architecture using universal_gradcam.
    
    This function:
    1. Gets Conv2d layer names from the base architecture (BEFORE modifying)
    2. Creates a model with the correct architecture and custom head for Pascal VOC
    3. Loads pretrained weights specific to each architecture
    4. Wraps with UniversalGradCAM using make_winsorcam_model() with the specific layers
    
    Args:
        model_name (str): Model architecture name
        device (str): Device to load model on ('cpu' or 'cuda')
        num_classes (int): Number of output classes (default: 20 for Pascal VOC)
        cpu_only_flag (bool): Whether running in CPU-only mode
    
    Returns:
        wrapped_model: Model wrapped with UniversalGradCAM functionality
        layer_names: List of Conv2d layer names for GradCAM (prefixed with 'model.')
        inception_flag: Whether this is an inception model (needs 299x299 input)
    """
    # Get the correct layers BEFORE modifying the model
    # This matches the approach from imagenet_example.ipynb
    inception_flag = False
    
    if model_name == 'resnet50':
        # Enable deterministic mode for ResNet50
        set_deterministic(cpu_only=cpu_only_flag, enable=True)
        base_model_temp = resnet50(weights='IMAGENET1K_V1')
        all_used_conv_layers = [name for name, module in base_model_temp.named_modules() if isinstance(module, nn.Conv2d)]
        
    elif model_name == 'densenet121':
        # Enable deterministic mode for DenseNet121
        set_deterministic(cpu_only=cpu_only_flag, enable=True)
        base_model_temp = densenet121(weights='IMAGENET1K_V1')
        all_used_conv_layers = [name for name, module in base_model_temp.named_modules() if isinstance(module, nn.Conv2d)]
        
    elif model_name == 'vgg16':
        # VGG16 requires deterministic mode to be OFF
        set_deterministic(cpu_only=cpu_only_flag, enable=False)
        base_model_temp = vgg16(weights='IMAGENET1K_V1')
        all_used_conv_layers = [name for name, module in base_model_temp.named_modules() if isinstance(module, nn.Conv2d)]
        
    elif model_name == 'inception_v3':
        # Enable deterministic mode for Inception v3
        set_deterministic(cpu_only=cpu_only_flag, enable=True)
        inception_flag = True  # Requires 299x299 input
        base_model_temp = inception_v3(weights='IMAGENET1K_V1')
        all_used_conv_layers = [name for name, module in base_model_temp.named_modules() if isinstance(module, nn.Conv2d) and "AuxLogits" not in name]
        
    elif model_name == 'efficientnet_b0':
        # Enable deterministic mode for EfficientNet-B0
        set_deterministic(cpu_only=cpu_only_flag, enable=True)
        base_model_temp = efficientnet_b0(weights='IMAGENET1K_V1')
        # Special filtering for EfficientNet to remove fc1, fc2, and certain block layers
        all_used_conv_layers = [name for name, module in base_model_temp.named_modules()
               if isinstance(module, nn.Conv2d) 
               and "fc1" not in name
               and "fc2" not in name
               and "block.0.0" not in name  # Remove expansion layers
               and "block.3.0" not in name  # Remove projection layers
               ]
        
    elif model_name == 'convnext_tiny':
        # Enable deterministic mode for ConvNeXt Tiny
        set_deterministic(cpu_only=cpu_only_flag, enable=True)
        base_model_temp = convnext_tiny(weights='IMAGENET1K_V1')
        all_used_conv_layers = [name for name, module in base_model_temp.named_modules() if isinstance(module, nn.Conv2d)]
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    
    if num_classes is not None:
        # Now create the model with custom head
        base_model = create_model_with_custom_head(model_name, num_classes)
        
        # Load pretrained weights
        base_model = load_model_weights(base_model, model_name, device)
    else:
        base_model = base_model_temp
    # Clean up temporary model
    del base_model_temp
    
    # Wrap with UniversalGradCAM using the specific layers we identified
    # This ensures we only hook the layers that existed in the original architecture
    wrapped_model = make_winsorcam_model(base_model, target_layers=all_used_conv_layers)
    wrapped_model = wrapped_model.to(device)
    wrapped_model.eval()
    
    # Get available layer names and add 'model.' prefix
    # The UniversalGradCAM wrapper stores layers with 'model.' prefix internally
    layer_names = [f"model.{name}" for name in all_used_conv_layers]
    
    return wrapped_model, layer_names, inception_flag

