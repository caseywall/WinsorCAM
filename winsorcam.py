from functools import partial

import torch.nn.functional as F

from torch import nn
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM, AblationCAM, RandomCAM, ScoreCAM, XGradCAM, LayerCAM, FullGrad, ShapleyCAM
from matplotlib import pyplot as plt


def resize_gradcams_grouped(gradcams, mode='nearest'):
    """
    Efficiently resize a list of gradcam tensors to the same target spatial size.
    The target size is automatically set to the largest height and width among all gradcams.
    Groups gradcams by shape to minimize the number of interpolation calls.

    Args:
        gradcams (list of torch.Tensor): List of 2D (H, W) or 3D (1, H, W) gradcam tensors (float).
        mode (str): Interpolation mode (default: 'nearest').

    Returns:
        list of torch.Tensor: List of gradcams resized to the largest spatial size (each tensor H x W, float).
    """
    # Find the largest height and width
    max_h = max(g.shape[-2] for g in gradcams)
    max_w = max(g.shape[-1] for g in gradcams)
    target_size = (max_h, max_w)

    # Group gradcams by their shape
    shape_groups = {}
    for i, gradcam in enumerate(gradcams):
        shape = gradcam.shape[-2:]  # Only spatial dims
        if shape not in shape_groups:
            shape_groups[shape] = []
        shape_groups[shape].append((i, gradcam))

    resized_gradcams = [None] * len(gradcams)
    for shape, group in shape_groups.items():
        indices, tensors = zip(*group)
        batch = torch.stack([g if g.dim() == 2 else g.squeeze(0) for g in tensors])  # (N, H, W)
        # Add channel dim if missing
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)  # (N, 1, H, W)
        resized = F.interpolate(batch, size=target_size, mode=mode).squeeze(1)  # (N, H, W)
        for i, idx in enumerate(indices):
            resized_gradcams[idx] = resized[i]
    return resized_gradcams

def normalize_gradcams_grouped(gradcams):
    """Normalize a list of gradcam tensors, grouping by shape to minimize stacking ops."""
    if not gradcams:
        return gradcams

    shape_groups = {}
    for i, gradcam in enumerate(gradcams):
        shape = gradcam.shape
        shape_groups.setdefault(shape, []).append((i, gradcam))

    normalized_gradcams = [None] * len(gradcams)
    for shape, group in shape_groups.items():
        indices, tensors = zip(*group)
        batch = torch.stack(tensors)  # (N, H, W) or (N, ...)
        flat = batch.reshape(batch.shape[0], -1)
        batch_min = torch.min(flat, dim=1).values.view(batch.shape[0], 1, 1)
        batch_max = torch.max(flat, dim=1).values.view(batch.shape[0], 1, 1)

        normalized_batch = torch.zeros_like(batch)
        for k in range(batch.shape[0]):
            if batch_max[k] != batch_min[k]:
                normalized_batch[k] = (batch[k] - batch_min[k]) / (batch_max[k] - batch_min[k])

        for k, idx in enumerate(indices):
            normalized_gradcams[idx] = normalized_batch[k]

    return normalized_gradcams

class ActivationGradientStorageGPU:
    def __init__(self):
        self._storage = {}
        self._handles = []  # Track gradient hook handles

    def store_activation(self, layer_name, output):
        """Store activation and register gradient hook on the same device as output."""
        if layer_name not in self._storage:
            self._storage[layer_name] = {'activations': None, 'gradients': None}
            
        # Store activation on its original device
        self._storage[layer_name]['activations'] = output.detach()
        
        # Register gradient hook
        if output.requires_grad:
            def _store_grad(grad):
                self._storage[layer_name]['gradients'] = grad.detach()
            handle = output.register_hook(_store_grad)
            self._handles.append(handle)

    def clear(self):
        """Clear storage before each forward pass"""
        self._storage.clear()
        # Remove all gradient hooks to prevent memory leaks
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        # clear all things from the gpu
        torch.cuda.empty_cache()


class WinsorcamClass(nn.Module):
    """
    A universal wrapper that adds GradCAM functionality to any PyTorch model.
    
    Args:
        model (nn.Module): Any PyTorch model
        target_layers (list, optional): List of specific layer names to hook. If provided, only these layers will be hooked.
        target_layer_types (tuple): Tuple of layer types to hook if target_layers not specified (default: (nn.Conv2d,))
        auto_register_hooks (bool): Whether to automatically register hooks on initialization
    
    Example:
        >>> model = torchvision.models.resnet50(pretrained=True)
        >>> # Option 1: Auto-hook all Conv2d layers
        >>> gradcam_model = UniversalGradCAM(model)
        >>> 
        >>> # Option 2: Hook specific layers only
        >>> gradcam_model = UniversalGradCAM(model, target_layers=['layer4.2.conv3', 'layer4.1.conv3'])
        >>> 
        >>> gradcam_model.eval()
        >>> output = gradcam_model(input_tensor)
        >>> stacked_gradcam, gradcams, importance = gradcam_model.get_gradcams_and_importance(...)
    """
    
    def __init__(self, model, target_layers=None, target_layer_types=(nn.Conv2d,), auto_register_hooks=True):
        super().__init__()
        self.model = model
        self.target_layers = target_layers  # Specific layer names
        self.target_layer_types = target_layer_types
        self.storage = ActivationGradientStorageGPU()
        self.hooks = []
        
        if auto_register_hooks:
            self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks on all layers of the specified types or specific layers."""
        def forward_hook(module, input, output, name):
            self.storage.store_activation(name, output)
        
        if self.target_layers is not None:
            # Hook specific layers by name
            for name, module in self.model.named_modules():
                if name in self.target_layers:
                    full_name = f"model.{name}"
                    self.hooks.append(
                        module.register_forward_hook(
                            partial(forward_hook, name=full_name)
                        )
                    )
            # print(f"Registered {len(self.hooks)} hooks on specified layers: {self.target_layers}")
        else:
            # Hook all layers of specified types
            for name, module in self.model.named_modules():
                if isinstance(module, self.target_layer_types):
                    full_name = f"model.{name}"
                    self.hooks.append(
                        module.register_forward_hook(
                            partial(forward_hook, name=full_name)
                        )
                    )
            # print(f"Registered {len(self.hooks)} hooks on {self.target_layer_types}")
                
    def _unregister_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_available_layers(self):
        """Get a list of all layer names that have hooks registered."""
        if self.target_layers is not None:
            # Return the specified layers
            return self.target_layers
        else:
            # Return all layers of the target types
            return [name for name, module in self.model.named_modules() 
                    if isinstance(module, self.target_layer_types)]
    
    def __call__(self, x):
        self.storage.clear()  # Clear before forward pass
        return super().__call__(x)

    def forward(self, x):
        return self.model(x)

    def generate_gradcam(self, filter_importances, activations):
        """Generate individual GradCAM heatmaps for each layer."""
        gradcams = []
        for filter_importance, activation in zip(filter_importances, activations):
            gradcam = torch.sum(filter_importance[:, None, None] * activation.squeeze(), dim=0)
            gradcam = torch.relu(gradcam)
            gradcams.append(gradcam)
        return gradcams, filter_importances
    
    @staticmethod
    def winsorize_preserve_zeros(tensor, percentile=99):
        """Winsorize tensor while preserving zero values."""
        # Identify nonzero values
        nonzero_mask = tensor > 0  
        
        # Compute percentile threshold only for nonzero values
        nonzero_values = tensor[nonzero_mask]
        threshold = torch.quantile(nonzero_values, percentile / 100) if nonzero_values.numel() > 0 else tensor.max()
        
        # Apply winsorization only to nonzero values
        winsorized_tensor = torch.where(nonzero_mask, tensor.clamp(max=threshold), tensor)
        
        return winsorized_tensor
    
    @staticmethod
    def normalize_nonzero(tensor, high=1, low=.1):
        """Normalize only non-zero values in the tensor."""
        normalized = tensor.clone()
        nonzero_mask = normalized > 0
        
        if not nonzero_mask.any():
            return normalized
        
        nonzero_values = normalized[nonzero_mask]
        
        if nonzero_values.max() == nonzero_values.min():
            nonzero_values = torch.ones_like(nonzero_values) * high
        else:
            nonzero_values = low + (nonzero_values - nonzero_values.min()) / \
                            (nonzero_values.max() - nonzero_values.min()) * (high - low)
        
        normalized[nonzero_mask] = nonzero_values
        
        return normalized
    
    @staticmethod
    def normalize_tensor(tensor, high=1, low=-1):
        """Normalize tensor to [low, high] range."""
        if ((tensor.max() - tensor.min()) * (high - low)) == 0:
            return torch.zeros_like(tensor)
        return low + (tensor - tensor.min()) / (tensor.max() - tensor.min()) * (high - low)
    
    def get_gradcams_and_importance(self, input_tensor, target_class, layers,
                                gradient_aggregation_method,
                                layer_aggregation_method, stack_relu,
                                interpolation_mode='nearest'):
        """
        Generate GradCAM heatmaps for specified layers.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index
            layers: List of layer names to compute GradCAM for
            gradient_aggregation_method: How to aggregate gradients ('mean', 'max', 'absmax', etc.)
            layer_aggregation_method: How to aggregate layer importances ('mean', 'max', 'L2norm', etc.)
            stack_relu: Whether to apply ReLU to importance tensor
            interpolation_mode: Mode for resizing ('nearest', 'bilinear', etc.)
            
        Returns:
            stacked_gradcam: Tensor of all gradcams stacked (num_layers, H, W)
            gradcams: List of individual gradcam tensors
            importance_tensor: Tensor of layer importances
        """

        
        device = input_tensor.device
        
        # Forward and backward passes
        with torch.amp.autocast(device_type='cuda', enabled=False):
            if not input_tensor.is_contiguous():
                input_tensor = input_tensor.contiguous()
                
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            output = self(input_tensor)
            self.zero_grad()
            
            target = output[0, target_class]
            target.backward(retain_graph=False, create_graph=False)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()

        # Get activations and gradients
        activations = []
        gradients = []
        for layer_name in layers:
            layer_data = self.storage._storage[layer_name]
            act = layer_data['activations'].detach().cpu()
            grad = layer_data['gradients'].detach().cpu()

            activations.append(act)
            gradients.append(grad)
        self.storage.clear()
        
        # Check for NaN/Inf
        has_nan_inf = False
        for grad in gradients:
            if (~torch.isfinite(grad)).any():
                has_nan_inf = True
                break

        if has_nan_inf:
            print('Gradients contain NaN or Inf values')
        
        # Process filter importances
        filter_importances = self.generate_filter_importances(layers, gradients, gradient_aggregation_method)
        
        # Generate gradcams
        gradcams, importance_lists = self.generate_gradcam(filter_importances, activations)


        # Normalize all gradcams
        gradcams = normalize_gradcams_grouped(gradcams)

        
        # Process layer importances
        importance_lists = self.generate_layer_importances(importance_lists, layer_aggregation_method)
        importance_tensor = torch.stack(importance_lists)
        
        if stack_relu:
            importance_tensor = torch.relu(importance_tensor)
        
        # Resize gradcams
        resized_gradcams = resize_gradcams_grouped(gradcams, interpolation_mode)
        
        # Stack and final resize
        stacked_gradcam = torch.stack(resized_gradcams)
        
        if stacked_gradcam.shape[-2:] != input_tensor.shape[2:]:
            stacked_gradcam = F.interpolate(
                stacked_gradcam.unsqueeze(1),
                size=input_tensor.shape[2:], 
                mode=interpolation_mode
            ).squeeze(1)
        
        
        return stacked_gradcam, gradcams, importance_tensor
    
    def generate_filter_importances(self, layers, gradients, gradient_aggregation_method):
        """Generate filter importance scores based on gradients."""
        filter_importances = []
        for layer_name, gradient in zip(layers, gradients):
            match gradient_aggregation_method:
                case 'mean':
                    filter_importances.append(torch.mean(gradient, dim=(2, 3))[0, :])
                case 'max':
                    filter_importances.append(torch.amax(gradient, dim=[0, 2, 3]))
                case 'min': 
                    filter_importances.append(torch.amin(gradient, dim=[0, 2, 3]))
                case 'absmax':
                    filter_importances.append(torch.amax(torch.abs(gradient), dim=[0, 2, 3]))
                case 'sum':
                    filter_importances.append(torch.sum(gradient, dim=[0, 2, 3]))
                case 'l2norm':
                    filter_importances.append(torch.norm(gradient, p=2, dim=[0, 2, 3]))
                case "absmean":
                    filter_importances.append(torch.mean(torch.abs(gradient), dim=[0, 2, 3]))
                case "variance":
                    filter_importances.append(torch.var(gradient, dim=[0, 2, 3]))
                case "std":
                    filter_importances.append(torch.std(gradient, dim=[0, 2, 3]))
                case "kurtosis":
                    mean_gradients = torch.mean(gradient, dim=[0, 2, 3], keepdim=True)
                    deviations = gradient - mean_gradients
                    fourth_moment = torch.mean(deviations ** 4, dim=[0, 2, 3])
                    second_moment = torch.mean(deviations ** 2, dim=[0, 2, 3])
                    filter_importance = fourth_moment / (second_moment ** 2)
                    filter_importance[torch.isnan(filter_importance)] = 0
                    filter_importances.append(filter_importance)
                case "huber":
                    huber_loss = torch.where(
                        torch.abs(gradient) <= gradient,
                        0.5 * gradient**2,
                        gradient * (torch.abs(gradient) - 0.5 * gradient)
                    )
                    nonzero_mask = gradient != 0
                    huber_loss = huber_loss * nonzero_mask
                    filter_importances.append(torch.mean(huber_loss, dim=[0, 2, 3]))
                case "singlemax":
                    mean_gradients = torch.mean(gradient, dim=[0, 2, 3])
                    max_index = torch.argmax(mean_gradients)
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[max_index] = 1
                    filter_importances.append(filter_importance)
                case "singlemin":
                    mean_gradients = torch.mean(gradient, dim=[0, 2, 3])
                    min_index = torch.argmin(mean_gradients)
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[min_index] = 1
                    filter_importances.append(filter_importance)
                case _:
                    raise ValueError(f"Invalid gradient_aggregation_method: {gradient_aggregation_method}")
        return filter_importances

    def generate_layer_importances(self, importance_lists, layer_aggregation_method):
        """Aggregate filter importances into layer-level importances."""
        match layer_aggregation_method:
            case 'mean':
                importance_lists = [torch.mean(importance_list) for importance_list in importance_lists]
            case 'max':
                importance_lists = [torch.max(importance_list) for importance_list in importance_lists]
            case 'min':
                importance_lists = [torch.min(importance_list) for importance_list in importance_lists]
            case 'L2norm':
                importance_lists = [torch.norm(importance_list, p=2) for importance_list in importance_lists]
            case "std":
                importance_lists = [torch.std(importance_list) for importance_list in importance_lists]
            case "entropy":
                importance_lists = [torch.distributions.Categorical(
                    probs=torch.nn.functional.softmax(importance_list, dim=0)
                ).entropy() for importance_list in importance_lists]
            case _:
                raise ValueError(f"Invalid layer_aggregation_method: {layer_aggregation_method}")
        return importance_lists

    def winsorize_stacked_gradcam(self, input_tensor, stacked_gradcam, importance_tensor, 
                                  interpolation_mode='nearest', winsor_percentile=99):
        """Apply winsorization to stacked gradcam."""
        importance_tensor = self.winsorize_preserve_zeros(importance_tensor, percentile=winsor_percentile)
        normalized_importance = self.normalize_nonzero(importance_tensor, high=1, low=.1)
        stacked_gradcam = torch.sum(stacked_gradcam * normalized_importance[:, None, None], dim=0)
        stacked_gradcam = F.interpolate(stacked_gradcam.unsqueeze(0).unsqueeze(0), 
                                       size=input_tensor.shape[2:], 
                                       mode=interpolation_mode).squeeze()
        self.storage.clear()
        return stacked_gradcam, importance_tensor

    def generate_saliency_map(self, input_tensor, target_class):
        """Generate a saliency map based on input gradients."""
        self.eval()
        input_tensor.requires_grad_()
        self._unregister_hooks()
        
        output = self(input_tensor)
        self.zero_grad()
        self.storage.clear()
        
        target = output[0, target_class]
        target.backward()
        
        saliency_map = input_tensor.grad.data
        saliency_map = torch.abs(saliency_map)
        saliency_map = self.normalize_tensor(saliency_map, high=1, low=0)
        saliency_map = saliency_map.squeeze()
        
        self._register_hooks()
        saliency_map = torch.mean(saliency_map, dim=0)

        return saliency_map
    
    def get_cam_comparative(self, image_tensor, method):
        """Compare with other CAM methods from pytorch_grad_cam library."""
        match method:
            case 'gradcam':
                method_func = GradCAM
            case 'gradcampp':
                method_func = GradCAMPlusPlus
            case 'scorecam':
                method_func = ScoreCAM
            case 'xgradcam':
                method_func = XGradCAM
            case 'fullgrad':
                method_func = FullGrad
                # FullGrad requires Conv2d layers with biases, excluding auxiliary classifiers for InceptionV3 compatibility
                target_layers = [module for name, module in self.model.named_modules() 
                                if isinstance(module, nn.Conv2d) and hasattr(module, 'bias') and module.bias is not None 
                                and 'AuxLogits' not in name]
            case 'ablation':
                method_func = AblationCAM
            case 'layercam':
                method_func = LayerCAM
            case 'shapleycam':
                method_func = ShapleyCAM
            case _:
                raise ValueError(f"Unknown CAM method: {method}")

        # For non-FullGrad methods, use only the last Conv2d layer
        if method != 'fullgrad':
            target_layers = [module for name, module in self.model.named_modules() 
                            if isinstance(module, nn.Conv2d)][-1:]
        
        cam = method_func(self.model, target_layers=target_layers)
        
        if not image_tensor.requires_grad:
            image_tensor = image_tensor.detach()
        
        grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0))
        
        if torch.is_tensor(grayscale_cam):
            grayscale_cam = grayscale_cam.cpu().numpy()
        
        del cam
        torch.cuda.empty_cache()
        
        return grayscale_cam
    
    def __del__(self):
        self._unregister_hooks()
    
    def remove_hooks(self):
        self._unregister_hooks()


# Wrapper function
def make_winsorcam_model(model, target_layers=None, target_layer_types=(nn.Conv2d,)):
    """
    Convenience function to wrap any PyTorch model with GradCAM functionality.

    Example:
        >>> import torchvision.models as models
        >>> resnet = models.resnet50(pretrained=True)
        >>> gradcam_resnet = make_winsorcam_model(resnet)
        >>> output = gradcam_resnet(input_tensor)
    """
    return WinsorcamClass(model, target_layers=target_layers, target_layer_types=target_layer_types)
