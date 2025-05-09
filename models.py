from functools import partial

import torch.nn.functional as F

from torch import nn
import torch
from torchvision.models import resnet50, densenet121, inception_v3, vgg16


class ActivationGradientStorage:
    def __init__(self):
        self._storage = {}
        
    def store_activation(self, layer_name, output):
        """Store activation matching pytorch-grad-cam"""
        if not layer_name in self._storage:
            self._storage[layer_name] = {'activations': None, 'gradients': None}
            
        # Match their CPU storage
        self._storage[layer_name]['activations'] = output.cpu().detach()
        
        # Register gradient hook like pytorch-grad-cam
        if output.requires_grad:
            def _store_grad(grad):
                self._storage[layer_name]['gradients'] = grad.cpu().detach()
            output.register_hook(_store_grad)

    def clear(self):
        """Clear storage before each forward pass"""
        self._storage.clear()

class ResNet50Modified(nn.Module):
    def __init__(self, model_path=None, dataset=None):
        super().__init__()
        if model_path is None:
            model = resnet50(weights='IMAGENET1K_V1')
        else:
            model = resnet50(weights=None)
        # change the last layer to be the number of classes
        if dataset == "pascal_voc":
            num_classes = 20
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.fc.in_features, num_classes)
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load the model
        if model_path is not None:
            if device.type == "cpu":
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
            else:
                model.load_state_dict(torch.load(model_path))
        self.base_model = model
        self.storage = ActivationGradientStorage()
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output, name):
            self.storage.store_activation(name, output)
            
        # Remove backward hooks - use output.register_hook instead
        # for name, module in self.resnet50.named_modules():
        #     if isinstance(module, nn.Conv2d):
        #         full_name = f"resnet50.{name}"
        #         self.hooks.append(
        #             module.register_forward_hook(
        #                 partial(forward_hook, name=full_name)
        #             )
        #         )
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Conv2d):
                full_name = f"base_model.{name}"
                self.hooks.append(
                    module.register_forward_hook(
                        partial(forward_hook, name=full_name)
                    )
                )
        
                
    def _unregister_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def __call__(self, x):
        self.storage.clear()  # Clear before forward pass
        return super().__call__(x)  # Use parent's __call__ to avoid recursion


    def generate_gradcam(self, filter_importances, activations):
        # print(f"activations at max location: {activations[0, torch.argmax(filter_importance)]}")
        gradcams = []
        for filter_importance, activations in zip(filter_importances, activations):
            gradcam = torch.sum(filter_importance[:, None, None] * activations.squeeze(), dim=0)
            # print(f"gradcam min:{gradcam.min()}, max:{gradcam.max()}")
            # Apply ReLU
            # the reason you apply ReLU is to remove the negative values
            gradcam = torch.relu(gradcam)
            gradcams.append(gradcam)
        return gradcams, filter_importances
    
    @staticmethod
    def winsorize_preserve_zeros(tensor, percentile=99):
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
        """
        Normalize only non-zero values in the importance tensor to ensure all maps contribute.
        
        Args:
            importance_tensor: Tensor of importance values
            min_weight: Minimum weight for non-zero values after normalization
            max_weight: Maximum weight for non-zero values after normalization
            
        Returns:
            Normalized tensor where zeros stay zero, and all other values are
            normalized between min_weight and max_weight
        """
        # Create a copy to avoid modifying the original
        normalized = tensor.clone()
        
        # Create a mask of non-zero values
        nonzero_mask = normalized > 0
        
        # If there are no non-zero values, return the original tensor
        if not nonzero_mask.any():
            return normalized
        
        # Extract just the non-zero values
        nonzero_values = normalized[nonzero_mask]
        
        # Normalize the non-zero values between min_weight and max_weight
        if nonzero_values.max() == nonzero_values.min():
            # If all non-zero values are the same, set them all to max_weight
            nonzero_values = torch.ones_like(nonzero_values) * high
        else:
            # Otherwise, normalize them
            nonzero_values = low + (nonzero_values - nonzero_values.min()) / \
                            (nonzero_values.max() - nonzero_values.min()) * (high - low)
        
        # Put the normalized values back in the right places
        normalized[nonzero_mask] = nonzero_values
        
        return normalized
    
    # make a function that takes a list of layers and a list of aggregation methods and returns the stacked and averaged gradcam
    def generate_stacked_gradcam(self, input_tensor, target_class, layers,
                                 gradient_aggregation_method, stack_aggregation_method,
                                layer_aggregation_method, stack_relu, winsor_percentile=99,
                                interpolation_mode='nearest'):

        # Forward pass
        self.eval()
        output = self(input_tensor)

        # Zero gradients
        self.zero_grad()

        # Backward pass
        target = output[0, target_class]
        target.backward()

        # Get the activations and gradients for the target layer
        activations = [self.storage._storage[layer_name]['activations'] for layer_name in layers]
        gradients = [self.storage._storage[layer_name]['gradients'] for layer_name in layers]
        # print(f"activation shapes: {[activation.shape for activation in activations]}") #activation shapes: [torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 64, 56, 56])...
        # print(f"gradient shapes: {[gradient.shape for gradient in gradients]}") # gradient shapes: [torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 64, 56, 56])...

        # raise ValueError("stop here so I can see the gradients")


        if any(torch.isnan(grad).any() or torch.isinf(grad).any() for grad in gradients):
            print(f'Gradients for layer {layer_name} contain NaN or Inf values')

        largest_map = max([layer["activations"].shape[2:] for layer in self.storage._storage.values()])

        filter_importances = []
        for layer_name, gradients in zip(layers, gradients):
            match gradient_aggregation_method:
                case 'mean':
                    # Global average pooling
                    # This will show the average importance of each filter
                    filter_importances.append(torch.mean(gradients, dim=(2, 3))[0, :])
                case 'max':
                    # Global max pooling
                    # This will show the filter with the most importance
                    filter_importances.append(torch.amax(gradients, dim=[0, 2, 3]))
                case 'min': 
                    # Global min pooling
                    # This will show the filter with the least importance
                    filter_importances.append(torch.amin(gradients, dim=[0, 2, 3]))
                case 'absmax':
                    # This will show the filter with the most importance regardless of the sign
                    # can be less sensitive to the noise or outliers
                    filter_importances.append(torch.amax(torch.abs(gradients), dim=[0, 2, 3]))
                case 'sum':
                    # This will show the cumulative importance of each filter
                    filter_importances.append(torch.sum(gradients, dim=[0, 2, 3]))
                case 'l2norm':
                    # similar to the absmax but it show magnitude of the importance
                    # can be less sensitive to the noise or outliers
                    filter_importances.append(torch.norm(gradients, p=2, dim=[0, 2, 3]))
                case "absmean":

                    filter_importances.append(torch.mean(torch.abs(gradients), dim=[0, 2, 3]))
                case "variance":
                    # Global variance pooling
                    # Spread of influence within a feature map
                    filter_importances.append(torch.var(gradients, dim=[0, 2, 3]))
                case "std":
                    filter_importances.append(torch.std(gradients, dim=[0, 2, 3]))

                case "kurtosis":
                    # Global kurtosis pooling
                    # Sharpness of the importance distribution
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
                    deviations = gradients - mean_gradients
                    fourth_moment = torch.mean(deviations ** 4, dim=[0, 2, 3])
                    second_moment = torch.mean(deviations ** 2, dim=[0, 2, 3])
                    filter_importance = fourth_moment / (second_moment ** 2)
                    # as kurtosis divides it is possible to get a division by zero I need to handle this
                    filter_importance[torch.isnan(filter_importance)] = 0
                    filter_importances.append(filter_importance)
                case "huber":
                    # Huber loss pooling
                    # This is a robust loss function that is less sensitive to outliers
                    # Step 1: Apply the Huber loss function element-wise
                    huber_loss = torch.where(
                        torch.abs(gradients) <= gradients,
                        0.5 * gradients**2,
                        gradients * (torch.abs(gradients) - 0.5 * gradients)
                    )
                    
                    # Step 2: Mask out zero values
                    nonzero_mask = gradients != 0
                    huber_loss = huber_loss * nonzero_mask  # Exclude zeros from the loss calculation
                    
                    # Step 3: Pool the values along the specified dimensions (dim=[0, 2, 3])
                    # Using mean pooling but with the Huber loss instead of raw values
                    filter_importances.append(torch.mean(huber_loss, dim=[0, 2, 3]))
                # make one that only shows the single most important filter
                case "singlemax":
                    # this would need to put a 1 where the max is and 0 everywhere else
                    # find the sum of each layers gradients
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
                    # find the max of the sum
                    max_index = torch.argmax(mean_gradients)
                    # create a tensor with 1 where the max is and 0 everywhere else
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[max_index] = 1
                    filter_importances.append(filter_importance)
                case "singlemin":
                    
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
                    min_index = torch.argmin(mean_gradients)
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[min_index] = 1
                    filter_importances.append(filter_importance)
                case _:
                    raise ValueError(f"Invalid gradient_aggregation_method: {gradient_aggregation_method}")

        # for each layer produce a gradcam
        gradcams, importance_lists = self.generate_gradcam(filter_importances, activations)
        
        gradcams = [self.normalize_tensor(gradcam, high=1, low=0) for gradcam in gradcams]
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
            

        importance_tensor = torch.stack(importance_lists)
        importance_tensor = torch.relu(importance_tensor) if stack_relu else importance_tensor
        # print(f"importance_tensor min:{importance_tensor.min()}, max:{importance_tensor.max()}")
        # # Clamp the importance tensor based on the standard deviation threshold
        importance_tensor = self.winsorize_preserve_zeros(importance_tensor, percentile=winsor_percentile)


        gradcams = [torch.nn.functional.interpolate(gradcam.unsqueeze(0).unsqueeze(0), size=largest_map, mode=interpolation_mode).squeeze() for gradcam in gradcams]
        # stack the gradcams with a relu to prevent negative values from cancelling out positive values
        stacked_gradcam = torch.stack(gradcams)


        if len(gradcams) == 1:
            stacked_gradcam = F.interpolate(stacked_gradcam.unsqueeze(0), size=input_tensor.shape[2:], mode=interpolation_mode).squeeze()
            return stacked_gradcam, gradcams, importance_tensor


        match stack_aggregation_method:
            case 'mean':
                # This might show the average place where the model is looking
                stacked_gradcam = torch.mean(stacked_gradcam, dim=0)
            case 'sum':
                # This might show every place where the model is looking
                stacked_gradcam = torch.sum(stacked_gradcam, dim=0)
            case "absmean":
                # Based on a normalization between -1 and 1
                # This could show us where the model seeing both very important and very unimportant areas as high intensity
                # while if it is somewhere in the middle it is less intense
                stacked_gradcam = torch.mean(torch.abs(stacked_gradcam), dim=0)
            case 'max':
                # This might show where the model look the most for information
                # It can be thought of as a measurement where any of the layers see anything
                # and can show where the model has not looked at at all
                stacked_gradcam, _ = torch.max(stacked_gradcam, dim=0)
            case 'min':
                # shows where the model is looking at typically
                # It can also be thought of as a measurement of consistency across layers
                # modify stacked_gradcam so that it only has values that are greater than 0
                stacked_gradcam = torch.relu(stacked_gradcam)
                # remove all maps that only contain 0s
                stacked_gradcam, _ = torch.min(stacked_gradcam, dim=0)
            case 'absmax':
                # This might show importace of the most important and least important areas as the data is normalized between -1 and 1
                # similar to absmean but rather than averaging we see the maximum (so more intensity in general)
                stacked_gradcam, _ = torch.max(torch.abs(stacked_gradcam), dim=0)
            case 'sum':
                # This shows every place where the model is looking
                # It also shows the cumulative effect of the layers on the final output
                stacked_gradcam = torch.sum(stacked_gradcam, dim=0)
            case 'l2norm':
                # This gives a sense of the overall magnitude considering the strength of activations across maps.
                stacked_gradcam = torch.norm(stacked_gradcam, p=2, dim=0)
            case "l1norm":
                # This gives a sense of differenc considering the strength of activations across maps.
                stacked_gradcam = torch.norm(stacked_gradcam, p=1, dim=0)
            case "variance":
                # This might show where the model is having the most amount difficulty agreeing based on the variance
                # Low values would typically show that the model is in agreement about a particular area
                # High values would show that the model is having a hard time agreeing
                stacked_gradcam = torch.var(stacked_gradcam, dim=0)
            case "std":
                # This might show where the model is having the most amount difficulty agreeing based on the standard deviation
                # Low values would typically show that the model is in agreement about a particular area
                # High values would show that the model is having a hard time agreeing
                stacked_gradcam = torch.std(stacked_gradcam, dim=0)
            case "kurtosis":
                # This might show where the model is having the most amount difficulty agreeing based on the kurtosis
                # Low values would typically show that the model is in agreement about a particular area
                # High values would show that the model is having a hard time agreeing
                mean_stacked_gradcam = torch.mean(stacked_gradcam, dim=0, keepdim=True)
                deviations = stacked_gradcam - mean_stacked_gradcam
                fourth_moment = torch.mean(deviations ** 4, dim=0)
                second_moment = torch.mean(deviations ** 2, dim=0)
                stacked_gradcam = fourth_moment / (second_moment ** 2)
                # as kurtosis divides it is possible to get a division by zero I need to handle this
                stacked_gradcam[torch.isnan(stacked_gradcam)] = 0
            case "huber":
                # Huber loss pooling
                # This is a robust loss function that is less sensitive to outliers
                # Step 1: Apply the Huber loss function element-wise
                huber_loss = torch.where(
                    torch.abs(stacked_gradcam) <= stacked_gradcam,
                    0.5 * stacked_gradcam**2,
                    stacked_gradcam * (torch.abs(stacked_gradcam) - 0.5 * stacked_gradcam)
                )
                
                # Step 2: Mask out zero values
                nonzero_mask = stacked_gradcam != 0
                huber_loss = huber_loss * nonzero_mask
                stacked_gradcam = torch.mean(huber_loss, dim=0)
            case "weighted":
                # This is a weighted sum of the gradcams
                # the weights are the average filter importance
                # Convert list to tensor and normalize between .1 and 1 avoiding getting rid of zeros
                normalized_importance = self.normalize_nonzero(importance_tensor, high=1, low=.1)
                # Apply weights to stacked gradcam
                stacked_gradcam = torch.sum(stacked_gradcam * normalized_importance[:, None, None], dim=0)
            case "weighted_inverted":
                # This need to take the inverse of the importance tensor
                # Convert list to tensor and normalize between 0 and 1
                normalized_importance = self.normalize_tensor(importance_tensor, high=1, low=0)
                # invert the importance tensor
                inverted_importance = 1 - normalized_importance
                # Apply weights to stacked gradcam
                stacked_gradcam = torch.sum(stacked_gradcam * inverted_importance[:, None, None], dim=0)
                # change the importance tensor to the inverted importance tensor
                importance_tensor = inverted_importance
            case "singlemax":
                # find the filter that has the most importance based on the average filter importance
                max_index = torch.argmax(importance_tensor)
                # get the gradcam for that layer
                stacked_gradcam = gradcams[max_index]
            case "singlemin":
                # find the filter that has the least importance based on the average filter importance
                min_index = torch.argmin(importance_tensor)
                # get the gradcam for that layer
                stacked_gradcam = gradcams[min_index]
            case "count":
                # this will be the count of occurences that are greater than 0
                nonzero_masks = [torch.where(gradcam > 0, torch.tensor(1), torch.tensor(0)) for gradcam in gradcams]
                stacked_gradcam = torch.sum(torch.stack(nonzero_masks), dim=0, dtype=torch.float32)

            case _:
                raise ValueError(f"Invalid stack_aggregation_method: {stack_aggregation_method}")
            
        # resize the gradcam to the size of the image
        stacked_gradcam = F.interpolate(stacked_gradcam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[2:], mode=interpolation_mode).squeeze()
        self.storage.clear()
        return stacked_gradcam, gradcams, importance_tensor



        # make a function that takes a list of layers and a list of aggregation methods and returns the stacked and averaged gradcam
    
    def get_gradcams_and_importance(self, input_tensor, target_class, layers,
                                 gradient_aggregation_method,
                                layer_aggregation_method, stack_relu,
                                interpolation_mode='nearest'):

        # Forward pass
        self.eval()
        output = self(input_tensor)

        # Zero gradients
        self.zero_grad()

        # Backward pass
        target = output[0, target_class]
        target.backward()

        # Get the activations and gradients for the target layer
        activations = [self.storage._storage[layer_name]['activations'] for layer_name in layers]
        gradients = [self.storage._storage[layer_name]['gradients'] for layer_name in layers]
        # print(f"activation shapes: {[activation.shape for activation in activations]}") #activation shapes: [torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 64, 56, 56])...
        # print(f"gradient shapes: {[gradient.shape for gradient in gradients]}") # gradient shapes: [torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 64, 56, 56])...

        # raise ValueError("stop here so I can see the gradients")


        if any(torch.isnan(grad).any() or torch.isinf(grad).any() for grad in gradients):
            print(f'Gradients for layer {layer_name} contain NaN or Inf values')

        largest_map = max([layer["activations"].shape[2:] for layer in self.storage._storage.values()])

        filter_importances = []
        for layer_name, gradients in zip(layers, gradients):
            match gradient_aggregation_method:
                case 'mean':
                    # Global average pooling
                    # This will show the average importance of each filter
                    filter_importances.append(torch.mean(gradients, dim=(2, 3))[0, :])
                case 'max':
                    # Global max pooling
                    # This will show the filter with the most importance
                    filter_importances.append(torch.amax(gradients, dim=[0, 2, 3]))
                case 'min': 
                    # Global min pooling
                    # This will show the filter with the least importance
                    filter_importances.append(torch.amin(gradients, dim=[0, 2, 3]))
                case 'absmax':
                    # This will show the filter with the most importance regardless of the sign
                    # can be less sensitive to the noise or outliers
                    filter_importances.append(torch.amax(torch.abs(gradients), dim=[0, 2, 3]))
                case 'sum':
                    # This will show the cumulative importance of each filter
                    filter_importances.append(torch.sum(gradients, dim=[0, 2, 3]))
                case 'l2norm':
                    # similar to the absmax but it show magnitude of the importance
                    # can be less sensitive to the noise or outliers
                    filter_importances.append(torch.norm(gradients, p=2, dim=[0, 2, 3]))
                case "absmean":

                    filter_importances.append(torch.mean(torch.abs(gradients), dim=[0, 2, 3]))
                case "variance":
                    # Global variance pooling
                    # Spread of influence within a feature map
                    filter_importances.append(torch.var(gradients, dim=[0, 2, 3]))
                case "std":
                    filter_importances.append(torch.std(gradients, dim=[0, 2, 3]))

                case "kurtosis":
                    # Global kurtosis pooling
                    # Sharpness of the importance distribution
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
                    deviations = gradients - mean_gradients
                    fourth_moment = torch.mean(deviations ** 4, dim=[0, 2, 3])
                    second_moment = torch.mean(deviations ** 2, dim=[0, 2, 3])
                    filter_importance = fourth_moment / (second_moment ** 2)
                    # as kurtosis divides it is possible to get a division by zero I need to handle this
                    filter_importance[torch.isnan(filter_importance)] = 0
                    filter_importances.append(filter_importance)
                case "huber":
                    # Huber loss pooling
                    # This is a robust loss function that is less sensitive to outliers
                    # Step 1: Apply the Huber loss function element-wise
                    huber_loss = torch.where(
                        torch.abs(gradients) <= gradients,
                        0.5 * gradients**2,
                        gradients * (torch.abs(gradients) - 0.5 * gradients)
                    )
                    
                    # Step 2: Mask out zero values
                    nonzero_mask = gradients != 0
                    huber_loss = huber_loss * nonzero_mask  # Exclude zeros from the loss calculation
                    
                    # Step 3: Pool the values along the specified dimensions (dim=[0, 2, 3])
                    # Using mean pooling but with the Huber loss instead of raw values
                    filter_importances.append(torch.mean(huber_loss, dim=[0, 2, 3]))
                # make one that only shows the single most important filter
                case "singlemax":
                    # this would need to put a 1 where the max is and 0 everywhere else
                    # find the sum of each layers gradients
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
                    # find the max of the sum
                    max_index = torch.argmax(mean_gradients)
                    # create a tensor with 1 where the max is and 0 everywhere else
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[max_index] = 1
                    filter_importances.append(filter_importance)
                case "singlemin":
                    
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
                    min_index = torch.argmin(mean_gradients)
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[min_index] = 1
                    filter_importances.append(filter_importance)
                case _:
                    raise ValueError(f"Invalid gradient_aggregation_method: {gradient_aggregation_method}")

        # for each layer produce a gradcam
        gradcams, importance_lists = self.generate_gradcam(filter_importances, activations)
        
        gradcams = [self.normalize_tensor(gradcam, high=1, low=0) for gradcam in gradcams]
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
                # calculate the entropy of the importance list
                # this is a measure of uncertainty
                # high entropy means that the model is uncertain about the importance
                # low entropy means that the model is certain about the importance
                # Convert gradients to valid probabilities using softmax
                importance_lists = [torch.distributions.Categorical(
                    probs=torch.nn.functional.softmax(importance_list, dim=0)
                ).entropy() for importance_list in importance_lists]
                
            case _:
                raise ValueError(f"Invalid layer_aggregation_method: {layer_aggregation_method}")
            

        importance_tensor = torch.stack(importance_lists)
        importance_tensor = torch.relu(importance_tensor) if stack_relu else importance_tensor

        gradcams = [torch.nn.functional.interpolate(gradcam.unsqueeze(0).unsqueeze(0), size=largest_map, mode=interpolation_mode).squeeze() for gradcam in gradcams]
        # stack the gradcams with a relu to prevent negative values from cancelling out positive values
        stacked_gradcam = torch.stack(gradcams)


        if len(gradcams) == 1:
            stacked_gradcam = F.interpolate(stacked_gradcam.unsqueeze(0), size=input_tensor.shape[2:], mode=interpolation_mode).squeeze()
            return stacked_gradcam, gradcams, importance_tensor
        
        return stacked_gradcam, gradcams, importance_tensor

    # now I need a function that takes in stacked_gradcam, gradcams, importance_tensor, and winsor_percentile
    # then outputs the stacked_gradcam (this is for efficiency when showing how winsorization works)
    def winsorize_stacked_gradcam(self,input_tensor, stacked_gradcam, gradcams, importance_tensor, interpolation_mode='nearest', winsor_percentile=99):
        importance_tensor = self.winsorize_preserve_zeros(importance_tensor, percentile=winsor_percentile)
        # This is a weighted sum of the gradcams
        normalized_importance = self.normalize_nonzero(importance_tensor, high=1, low=.1)
        # Apply weights to stacked gradcam
        stacked_gradcam = torch.sum(stacked_gradcam * normalized_importance[:, None, None], dim=0)
        # resize the gradcam to the size of the image
                # resize the gradcam to the size of the image
        stacked_gradcam = F.interpolate(stacked_gradcam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[2:], mode=interpolation_mode).squeeze()
        self.storage.clear()
        return stacked_gradcam, importance_tensor
        

    def generate_saliency_map(self, input_tensor, target_class):
        self.eval()
        input_tensor.requires_grad_()
        # Unregister ReLU hooks to get raw gradients
        self._unregister_hooks()
        
        # Forward pass
        output = self(input_tensor)
        self.zero_grad()
        self.storage.clear()
        # Backward pass on a single target class
        target = output[0, target_class]
        target.backward()
        # Get the gradients
        # these will be the gradients of the loss with respect to the input
        saliency_map = input_tensor.grad.data
        # take only the absolute value of the gradients
        saliency_map = torch.abs(saliency_map)
        # Normalize the saliency map between 0 and 1
        saliency_map = self.normalize_tensor(saliency_map, high=1, low=0)

        # squeeze the tensor to remove the channel dimension
        saliency_map = saliency_map.squeeze()
        # Re-register ReLU hooks for guided backpropagation
        self._register_hooks()
        # get the average across the channels
        saliency_map = torch.mean(saliency_map, dim=0)

        return saliency_map

    def forward(self, x):
        return self.base_model(x)
        # return self.resnet50(x)

    def normalize_tensor(self, tensor, high=1, low=-1):
        # as this will divide handle if the case is that all are 0
        if ((tensor.max() - tensor.min()) * (high - low)) == 0:
            return torch.zeros_like(tensor)
        return low + (tensor - tensor.min()) / (tensor.max() - tensor.min()) * (high - low)
    def __del__(self):
        self._unregister_hooks()
    def remove_hooks(self):
        self._unregister_hooks()


class Dense121Modified(nn.Module):
    def __init__(self, model_path=None, dataset=None):
        super().__init__()
        if model_path is None:
            model = densenet121(weights='IMAGENET1K_V1')
        else:
            model = densenet121(weights=None)
        # change the last layer to be the number of classes
        if dataset == "pascal_voc":
            num_classes = 20
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),  # Add dropout with 50% probability
                nn.Linear(1024, num_classes)  # Replace with your number of classes
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load the model
        if model_path is not None:
            if device.type == "cpu":
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
            else:
                model.load_state_dict(torch.load(model_path))
        self.base_model = model
        self.storage = ActivationGradientStorage()
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output, name):
            self.storage.store_activation(name, output)
            
        # Remove backward hooks - use output.register_hook instead
        # for name, module in self.resnet50.named_modules():
        #     if isinstance(module, nn.Conv2d):
        #         full_name = f"resnet50.{name}"
        #         self.hooks.append(
        #             module.register_forward_hook(
        #                 partial(forward_hook, name=full_name)
        #             )
        #         )
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Conv2d):
                full_name = f"base_model.{name}"
                self.hooks.append(
                    module.register_forward_hook(
                        partial(forward_hook, name=full_name)
                    )
                )
        
                
    def _unregister_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def __call__(self, x):
        self.storage.clear()  # Clear before forward pass
        return super().__call__(x)  # Use parent's __call__ to avoid recursion


    def generate_gradcam(self, filter_importances, activations):
        # print(f"activations at max location: {activations[0, torch.argmax(filter_importance)]}")
        gradcams = []
        for filter_importance, activations in zip(filter_importances, activations):
            gradcam = torch.sum(filter_importance[:, None, None] * activations.squeeze(), dim=0)
            # print(f"gradcam min:{gradcam.min()}, max:{gradcam.max()}")
            # Apply ReLU
            # the reason you apply ReLU is to remove the negative values
            gradcam = torch.relu(gradcam)
            gradcams.append(gradcam)
        return gradcams, filter_importances
    
    @staticmethod
    def winsorize_preserve_zeros(tensor, percentile=99):
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
        """
        Normalize only non-zero values in the importance tensor to ensure all maps contribute.
        
        Args:
            importance_tensor: Tensor of importance values
            min_weight: Minimum weight for non-zero values after normalization
            max_weight: Maximum weight for non-zero values after normalization
            
        Returns:
            Normalized tensor where zeros stay zero, and all other values are
            normalized between min_weight and max_weight
        """
        # Create a copy to avoid modifying the original
        normalized = tensor.clone()
        
        # Create a mask of non-zero values
        nonzero_mask = normalized > 0
        
        # If there are no non-zero values, return the original tensor
        if not nonzero_mask.any():
            return normalized
        
        # Extract just the non-zero values
        nonzero_values = normalized[nonzero_mask]
        
        # Normalize the non-zero values between min_weight and max_weight
        if nonzero_values.max() == nonzero_values.min():
            # If all non-zero values are the same, set them all to max_weight
            nonzero_values = torch.ones_like(nonzero_values) * high
        else:
            # Otherwise, normalize them
            nonzero_values = low + (nonzero_values - nonzero_values.min()) / \
                            (nonzero_values.max() - nonzero_values.min()) * (high - low)
        
        # Put the normalized values back in the right places
        normalized[nonzero_mask] = nonzero_values
        
        return normalized
    
    # make a function that takes a list of layers and a list of aggregation methods and returns the stacked and averaged gradcam
    def generate_stacked_gradcam(self, input_tensor, target_class, layers,
                                 gradient_aggregation_method, stack_aggregation_method,
                                layer_aggregation_method, stack_relu, winsor_percentile=99,
                                interpolation_mode='nearest'):

        # Forward pass
        self.eval()
        output = self(input_tensor)

        # Zero gradients
        self.zero_grad()

        # Backward pass
        target = output[0, target_class]
        target.backward()

        # Get the activations and gradients for the target layer
        activations = [self.storage._storage[layer_name]['activations'] for layer_name in layers]
        gradients = [self.storage._storage[layer_name]['gradients'] for layer_name in layers]
        # print(f"activation shapes: {[activation.shape for activation in activations]}") #activation shapes: [torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 64, 56, 56])...
        # print(f"gradient shapes: {[gradient.shape for gradient in gradients]}") # gradient shapes: [torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 64, 56, 56])...

        # raise ValueError("stop here so I can see the gradients")


        if any(torch.isnan(grad).any() or torch.isinf(grad).any() for grad in gradients):
            print(f'Gradients for layer {layer_name} contain NaN or Inf values')

        largest_map = max([layer["activations"].shape[2:] for layer in self.storage._storage.values()])

        filter_importances = []
        for layer_name, gradients in zip(layers, gradients):
            match gradient_aggregation_method:
                case 'mean':
                    # Global average pooling
                    # This will show the average importance of each filter
                    filter_importances.append(torch.mean(gradients, dim=(2, 3))[0, :])
                case 'max':
                    # Global max pooling
                    # This will show the filter with the most importance
                    filter_importances.append(torch.amax(gradients, dim=[0, 2, 3]))
                case 'min': 
                    # Global min pooling
                    # This will show the filter with the least importance
                    filter_importances.append(torch.amin(gradients, dim=[0, 2, 3]))
                case 'absmax':
                    # This will show the filter with the most importance regardless of the sign
                    # can be less sensitive to the noise or outliers
                    filter_importances.append(torch.amax(torch.abs(gradients), dim=[0, 2, 3]))
                case 'sum':
                    # This will show the cumulative importance of each filter
                    filter_importances.append(torch.sum(gradients, dim=[0, 2, 3]))
                case 'l2norm':
                    # similar to the absmax but it show magnitude of the importance
                    # can be less sensitive to the noise or outliers
                    filter_importances.append(torch.norm(gradients, p=2, dim=[0, 2, 3]))
                case "absmean":

                    filter_importances.append(torch.mean(torch.abs(gradients), dim=[0, 2, 3]))
                case "variance":
                    # Global variance pooling
                    # Spread of influence within a feature map
                    filter_importances.append(torch.var(gradients, dim=[0, 2, 3]))
                case "std":
                    filter_importances.append(torch.std(gradients, dim=[0, 2, 3]))

                case "kurtosis":
                    # Global kurtosis pooling
                    # Sharpness of the importance distribution
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
                    deviations = gradients - mean_gradients
                    fourth_moment = torch.mean(deviations ** 4, dim=[0, 2, 3])
                    second_moment = torch.mean(deviations ** 2, dim=[0, 2, 3])
                    filter_importance = fourth_moment / (second_moment ** 2)
                    # as kurtosis divides it is possible to get a division by zero I need to handle this
                    filter_importance[torch.isnan(filter_importance)] = 0
                    filter_importances.append(filter_importance)
                case "huber":
                    # Huber loss pooling
                    # This is a robust loss function that is less sensitive to outliers
                    # Step 1: Apply the Huber loss function element-wise
                    huber_loss = torch.where(
                        torch.abs(gradients) <= gradients,
                        0.5 * gradients**2,
                        gradients * (torch.abs(gradients) - 0.5 * gradients)
                    )
                    
                    # Step 2: Mask out zero values
                    nonzero_mask = gradients != 0
                    huber_loss = huber_loss * nonzero_mask  # Exclude zeros from the loss calculation
                    
                    # Step 3: Pool the values along the specified dimensions (dim=[0, 2, 3])
                    # Using mean pooling but with the Huber loss instead of raw values
                    filter_importances.append(torch.mean(huber_loss, dim=[0, 2, 3]))
                # make one that only shows the single most important filter
                case "singlemax":
                    # this would need to put a 1 where the max is and 0 everywhere else
                    # find the sum of each layers gradients
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
                    # find the max of the sum
                    max_index = torch.argmax(mean_gradients)
                    # create a tensor with 1 where the max is and 0 everywhere else
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[max_index] = 1
                    filter_importances.append(filter_importance)
                case "singlemin":
                    
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
                    min_index = torch.argmin(mean_gradients)
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[min_index] = 1
                    filter_importances.append(filter_importance)
                case _:
                    raise ValueError(f"Invalid gradient_aggregation_method: {gradient_aggregation_method}")

        # for each layer produce a gradcam
        gradcams, importance_lists = self.generate_gradcam(filter_importances, activations)
        
        gradcams = [self.normalize_tensor(gradcam, high=1, low=0) for gradcam in gradcams]
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
                # calculate the entropy of the importance list
                # this is a measure of uncertainty
                # high entropy means that the model is uncertain about the importance
                # low entropy means that the model is certain about the importance
                # Convert gradients to valid probabilities using softmax
                importance_lists = [torch.distributions.Categorical(
                    probs=torch.nn.functional.softmax(importance_list, dim=0)
                ).entropy() for importance_list in importance_lists]
                
            case _:
                raise ValueError(f"Invalid layer_aggregation_method: {layer_aggregation_method}")
            

        importance_tensor = torch.stack(importance_lists)
        importance_tensor = torch.relu(importance_tensor) if stack_relu else importance_tensor
        # print(f"importance_tensor min:{importance_tensor.min()}, max:{importance_tensor.max()}")
        # # Clamp the importance tensor based on the standard deviation threshold
        importance_tensor = self.winsorize_preserve_zeros(importance_tensor, percentile=winsor_percentile)


        gradcams = [torch.nn.functional.interpolate(gradcam.unsqueeze(0).unsqueeze(0), size=largest_map, mode=interpolation_mode).squeeze() for gradcam in gradcams]
        # stack the gradcams with a relu to prevent negative values from cancelling out positive values
        stacked_gradcam = torch.stack(gradcams)


        if len(gradcams) == 1:
            stacked_gradcam = F.interpolate(stacked_gradcam.unsqueeze(0), size=input_tensor.shape[2:], mode=interpolation_mode).squeeze()
            return stacked_gradcam, gradcams, importance_tensor


        match stack_aggregation_method:
            case 'mean':
                # This might show the average place where the model is looking
                stacked_gradcam = torch.mean(stacked_gradcam, dim=0)
            case 'sum':
                # This might show every place where the model is looking
                stacked_gradcam = torch.sum(stacked_gradcam, dim=0)
            case "absmean":
                # Based on a normalization between -1 and 1
                # This could show us where the model seeing both very important and very unimportant areas as high intensity
                # while if it is somewhere in the middle it is less intense
                stacked_gradcam = torch.mean(torch.abs(stacked_gradcam), dim=0)
            case 'max':
                # This might show where the model look the most for information
                # It can be thought of as a measurement where any of the layers see anything
                # and can show where the model has not looked at at all
                stacked_gradcam, _ = torch.max(stacked_gradcam, dim=0)
            case 'min':
                # shows where the model is looking at typically
                # It can also be thought of as a measurement of consistency across layers
                # modify stacked_gradcam so that it only has values that are greater than 0
                stacked_gradcam = torch.relu(stacked_gradcam)
                # remove all maps that only contain 0s
                stacked_gradcam, _ = torch.min(stacked_gradcam, dim=0)
            case 'absmax':
                # This might show importace of the most important and least important areas as the data is normalized between -1 and 1
                # similar to absmean but rather than averaging we see the maximum (so more intensity in general)
                stacked_gradcam, _ = torch.max(torch.abs(stacked_gradcam), dim=0)
            case 'sum':
                # This shows every place where the model is looking
                # It also shows the cumulative effect of the layers on the final output
                stacked_gradcam = torch.sum(stacked_gradcam, dim=0)
            case 'l2norm':
                # This gives a sense of the overall magnitude considering the strength of activations across maps.
                stacked_gradcam = torch.norm(stacked_gradcam, p=2, dim=0)
            case "l1norm":
                # This gives a sense of differenc considering the strength of activations across maps.
                stacked_gradcam = torch.norm(stacked_gradcam, p=1, dim=0)
            case "variance":
                # This might show where the model is having the most amount difficulty agreeing based on the variance
                # Low values would typically show that the model is in agreement about a particular area
                # High values would show that the model is having a hard time agreeing
                stacked_gradcam = torch.var(stacked_gradcam, dim=0)
            case "std":
                # This might show where the model is having the most amount difficulty agreeing based on the standard deviation
                # Low values would typically show that the model is in agreement about a particular area
                # High values would show that the model is having a hard time agreeing
                stacked_gradcam = torch.std(stacked_gradcam, dim=0)
            case "kurtosis":
                # This might show where the model is having the most amount difficulty agreeing based on the kurtosis
                # Low values would typically show that the model is in agreement about a particular area
                # High values would show that the model is having a hard time agreeing
                mean_stacked_gradcam = torch.mean(stacked_gradcam, dim=0, keepdim=True)
                deviations = stacked_gradcam - mean_stacked_gradcam
                fourth_moment = torch.mean(deviations ** 4, dim=0)
                second_moment = torch.mean(deviations ** 2, dim=0)
                stacked_gradcam = fourth_moment / (second_moment ** 2)
                # as kurtosis divides it is possible to get a division by zero I need to handle this
                stacked_gradcam[torch.isnan(stacked_gradcam)] = 0
            case "huber":
                # Huber loss pooling
                # This is a robust loss function that is less sensitive to outliers
                # Step 1: Apply the Huber loss function element-wise
                huber_loss = torch.where(
                    torch.abs(stacked_gradcam) <= stacked_gradcam,
                    0.5 * stacked_gradcam**2,
                    stacked_gradcam * (torch.abs(stacked_gradcam) - 0.5 * stacked_gradcam)
                )
                
                # Step 2: Mask out zero values
                nonzero_mask = stacked_gradcam != 0
                huber_loss = huber_loss * nonzero_mask
                stacked_gradcam = torch.mean(huber_loss, dim=0)
            case "weighted":
                # This is a weighted sum of the gradcams
                # the weights are the average filter importance
                # Convert list to tensor and normalize between .1 and 1 avoiding getting rid of zeros
                normalized_importance = self.normalize_nonzero(importance_tensor, high=1, low=.1)
                # Apply weights to stacked gradcam
                stacked_gradcam = torch.sum(stacked_gradcam * normalized_importance[:, None, None], dim=0)
            case "weighted_inverted":
                # This need to take the inverse of the importance tensor
                # Convert list to tensor and normalize between 0 and 1
                normalized_importance = self.normalize_tensor(importance_tensor, high=1, low=0)
                # invert the importance tensor
                inverted_importance = 1 - normalized_importance
                # Apply weights to stacked gradcam
                stacked_gradcam = torch.sum(stacked_gradcam * inverted_importance[:, None, None], dim=0)
                # change the importance tensor to the inverted importance tensor
                importance_tensor = inverted_importance
            case "singlemax":
                # find the filter that has the most importance based on the average filter importance
                max_index = torch.argmax(importance_tensor)
                # get the gradcam for that layer
                stacked_gradcam = gradcams[max_index]
            case "singlemin":
                # find the filter that has the least importance based on the average filter importance
                min_index = torch.argmin(importance_tensor)
                # get the gradcam for that layer
                stacked_gradcam = gradcams[min_index]
            case "count":
                # this will be the count of occurences that are greater than 0
                nonzero_masks = [torch.where(gradcam > 0, torch.tensor(1), torch.tensor(0)) for gradcam in gradcams]
                stacked_gradcam = torch.sum(torch.stack(nonzero_masks), dim=0, dtype=torch.float32)

            case _:
                raise ValueError(f"Invalid stack_aggregation_method: {stack_aggregation_method}")
            
        # resize the gradcam to the size of the image
        stacked_gradcam = F.interpolate(stacked_gradcam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[2:], mode=interpolation_mode).squeeze()
        self.storage.clear()
        return stacked_gradcam, gradcams, importance_tensor



        # make a function that takes a list of layers and a list of aggregation methods and returns the stacked and averaged gradcam
    
    def get_gradcams_and_importance(self, input_tensor, target_class, layers,
                                 gradient_aggregation_method,
                                layer_aggregation_method, stack_relu,
                                interpolation_mode='nearest'):

        # Forward pass
        self.eval()
        output = self(input_tensor)

        # Zero gradients
        self.zero_grad()

        # Backward pass
        target = output[0, target_class]
        target.backward()

        # Get the activations and gradients for the target layer
        activations = [self.storage._storage[layer_name]['activations'] for layer_name in layers]
        gradients = [self.storage._storage[layer_name]['gradients'] for layer_name in layers]
        # print(f"activation shapes: {[activation.shape for activation in activations]}") #activation shapes: [torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 64, 56, 56])...
        # print(f"gradient shapes: {[gradient.shape for gradient in gradients]}") # gradient shapes: [torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 64, 56, 56])...

        # raise ValueError("stop here so I can see the gradients")


        if any(torch.isnan(grad).any() or torch.isinf(grad).any() for grad in gradients):
            print(f'Gradients for layer {layer_name} contain NaN or Inf values')

        largest_map = max([layer["activations"].shape[2:] for layer in self.storage._storage.values()])

        filter_importances = []
        for layer_name, gradients in zip(layers, gradients):
            match gradient_aggregation_method:
                case 'mean':
                    # Global average pooling
                    # This will show the average importance of each filter
                    filter_importances.append(torch.mean(gradients, dim=(2, 3))[0, :])
                case 'max':
                    # Global max pooling
                    # This will show the filter with the most importance
                    filter_importances.append(torch.amax(gradients, dim=[0, 2, 3]))
                case 'min': 
                    # Global min pooling
                    # This will show the filter with the least importance
                    filter_importances.append(torch.amin(gradients, dim=[0, 2, 3]))
                case 'absmax':
                    # This will show the filter with the most importance regardless of the sign
                    # can be less sensitive to the noise or outliers
                    filter_importances.append(torch.amax(torch.abs(gradients), dim=[0, 2, 3]))
                case 'sum':
                    # This will show the cumulative importance of each filter
                    filter_importances.append(torch.sum(gradients, dim=[0, 2, 3]))
                case 'l2norm':
                    # similar to the absmax but it show magnitude of the importance
                    # can be less sensitive to the noise or outliers
                    filter_importances.append(torch.norm(gradients, p=2, dim=[0, 2, 3]))
                case "absmean":

                    filter_importances.append(torch.mean(torch.abs(gradients), dim=[0, 2, 3]))
                case "variance":
                    # Global variance pooling
                    # Spread of influence within a feature map
                    filter_importances.append(torch.var(gradients, dim=[0, 2, 3]))
                case "std":
                    filter_importances.append(torch.std(gradients, dim=[0, 2, 3]))

                case "kurtosis":
                    # Global kurtosis pooling
                    # Sharpness of the importance distribution
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
                    deviations = gradients - mean_gradients
                    fourth_moment = torch.mean(deviations ** 4, dim=[0, 2, 3])
                    second_moment = torch.mean(deviations ** 2, dim=[0, 2, 3])
                    filter_importance = fourth_moment / (second_moment ** 2)
                    # as kurtosis divides it is possible to get a division by zero I need to handle this
                    filter_importance[torch.isnan(filter_importance)] = 0
                    filter_importances.append(filter_importance)
                case "huber":
                    # Huber loss pooling
                    # This is a robust loss function that is less sensitive to outliers
                    # Step 1: Apply the Huber loss function element-wise
                    huber_loss = torch.where(
                        torch.abs(gradients) <= gradients,
                        0.5 * gradients**2,
                        gradients * (torch.abs(gradients) - 0.5 * gradients)
                    )
                    
                    # Step 2: Mask out zero values
                    nonzero_mask = gradients != 0
                    huber_loss = huber_loss * nonzero_mask  # Exclude zeros from the loss calculation
                    
                    # Step 3: Pool the values along the specified dimensions (dim=[0, 2, 3])
                    # Using mean pooling but with the Huber loss instead of raw values
                    filter_importances.append(torch.mean(huber_loss, dim=[0, 2, 3]))
                # make one that only shows the single most important filter
                case "singlemax":
                    # this would need to put a 1 where the max is and 0 everywhere else
                    # find the sum of each layers gradients
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
                    # find the max of the sum
                    max_index = torch.argmax(mean_gradients)
                    # create a tensor with 1 where the max is and 0 everywhere else
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[max_index] = 1
                    filter_importances.append(filter_importance)
                case "singlemin":
                    
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
                    min_index = torch.argmin(mean_gradients)
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[min_index] = 1
                    filter_importances.append(filter_importance)
                case _:
                    raise ValueError(f"Invalid gradient_aggregation_method: {gradient_aggregation_method}")

        # for each layer produce a gradcam
        gradcams, importance_lists = self.generate_gradcam(filter_importances, activations)
        
        gradcams = [self.normalize_tensor(gradcam, high=1, low=0) for gradcam in gradcams]
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
                # calculate the entropy of the importance list
                # this is a measure of uncertainty
                # high entropy means that the model is uncertain about the importance
                # low entropy means that the model is certain about the importance
                # Convert gradients to valid probabilities using softmax
                importance_lists = [torch.distributions.Categorical(
                    probs=torch.nn.functional.softmax(importance_list, dim=0)
                ).entropy() for importance_list in importance_lists]
                
            case _:
                raise ValueError(f"Invalid layer_aggregation_method: {layer_aggregation_method}")
            

        importance_tensor = torch.stack(importance_lists)
        importance_tensor = torch.relu(importance_tensor) if stack_relu else importance_tensor

        gradcams = [torch.nn.functional.interpolate(gradcam.unsqueeze(0).unsqueeze(0), size=largest_map, mode=interpolation_mode).squeeze() for gradcam in gradcams]
        # stack the gradcams with a relu to prevent negative values from cancelling out positive values
        stacked_gradcam = torch.stack(gradcams)


        if len(gradcams) == 1:
            stacked_gradcam = F.interpolate(stacked_gradcam.unsqueeze(0), size=input_tensor.shape[2:], mode=interpolation_mode).squeeze()
            return stacked_gradcam, gradcams, importance_tensor
        
        return stacked_gradcam, gradcams, importance_tensor

    # now I need a function that takes in stacked_gradcam, gradcams, importance_tensor, and winsor_percentile
    # then outputs the stacked_gradcam (this is for efficiency when showing how winsorization works)
    def winsorize_stacked_gradcam(self,input_tensor, stacked_gradcam, gradcams, importance_tensor, interpolation_mode='nearest', winsor_percentile=99):
        importance_tensor = self.winsorize_preserve_zeros(importance_tensor, percentile=winsor_percentile)
        # This is a weighted sum of the gradcams
        normalized_importance = self.normalize_nonzero(importance_tensor, high=1, low=.1)
        # Apply weights to stacked gradcam
        stacked_gradcam = torch.sum(stacked_gradcam * normalized_importance[:, None, None], dim=0)
        # resize the gradcam to the size of the image
                # resize the gradcam to the size of the image
        stacked_gradcam = F.interpolate(stacked_gradcam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[2:], mode=interpolation_mode).squeeze()
        self.storage.clear()
        return stacked_gradcam, importance_tensor
        

    def generate_saliency_map(self, input_tensor, target_class):
        self.eval()
        input_tensor.requires_grad_()
        # Unregister ReLU hooks to get raw gradients
        self._unregister_hooks()
        
        # Forward pass
        output = self(input_tensor)
        self.zero_grad()
        self.storage.clear()
        # Backward pass on a single target class
        target = output[0, target_class]
        target.backward()
        # Get the gradients
        # these will be the gradients of the loss with respect to the input
        saliency_map = input_tensor.grad.data
        # take only the absolute value of the gradients
        saliency_map = torch.abs(saliency_map)
        # Normalize the saliency map between 0 and 1
        saliency_map = self.normalize_tensor(saliency_map, high=1, low=0)

        # squeeze the tensor to remove the channel dimension
        saliency_map = saliency_map.squeeze()
        # Re-register ReLU hooks for guided backpropagation
        self._register_hooks()
        # get the average across the channels
        saliency_map = torch.mean(saliency_map, dim=0)

        return saliency_map

    def forward(self, x):
        return self.base_model(x)
        # return self.resnet50(x)

    def normalize_tensor(self, tensor, high=1, low=-1):
        # as this will divide handle if the case is that all are 0
        if ((tensor.max() - tensor.min()) * (high - low)) == 0:
            return torch.zeros_like(tensor)
        return low + (tensor - tensor.min()) / (tensor.max() - tensor.min()) * (high - low)
    def __del__(self):
        self._unregister_hooks()
    def remove_hooks(self):
        self._unregister_hooks()


class InceptionV3Modified(nn.Module):
    def __init__(self, model_path=None, dataset=None):
        super().__init__()
        if model_path is None:
            model = inception_v3(weights='IMAGENET1K_V1')
        else:
            model = inception_v3(weights=None)
        # change the last layer to be the number of classes
        if dataset == "pascal_voc":
            num_classes = 20
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            if model.aux_logits:
                model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load the model
        if model_path is not None:
            if device.type == "cpu":
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
            else:
                model.load_state_dict(torch.load(model_path))
        self.base_model = model
        self.storage = ActivationGradientStorage()
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output, name):
            self.storage.store_activation(name, output)
            
        # Remove backward hooks - use output.register_hook instead
        # for name, module in self.resnet50.named_modules():
        #     if isinstance(module, nn.Conv2d):
        #         full_name = f"resnet50.{name}"
        #         self.hooks.append(
        #             module.register_forward_hook(
        #                 partial(forward_hook, name=full_name)
        #             )
        #         )
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Conv2d):
                full_name = f"base_model.{name}"
                self.hooks.append(
                    module.register_forward_hook(
                        partial(forward_hook, name=full_name)
                    )
                )
        
                
    def _unregister_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def __call__(self, x):
        self.storage.clear()  # Clear before forward pass
        return super().__call__(x)  # Use parent's __call__ to avoid recursion


    def generate_gradcam(self, filter_importances, activations):
        # print(f"activations at max location: {activations[0, torch.argmax(filter_importance)]}")
        gradcams = []
        for filter_importance, activations in zip(filter_importances, activations):
            gradcam = torch.sum(filter_importance[:, None, None] * activations.squeeze(), dim=0)
            # print(f"gradcam min:{gradcam.min()}, max:{gradcam.max()}")
            # Apply ReLU
            # the reason you apply ReLU is to remove the negative values
            gradcam = torch.relu(gradcam)
            gradcams.append(gradcam)
        return gradcams, filter_importances
    
    @staticmethod
    def winsorize_preserve_zeros(tensor, percentile=99):
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
        """
        Normalize only non-zero values in the importance tensor to ensure all maps contribute.
        
        Args:
            importance_tensor: Tensor of importance values
            min_weight: Minimum weight for non-zero values after normalization
            max_weight: Maximum weight for non-zero values after normalization
            
        Returns:
            Normalized tensor where zeros stay zero, and all other values are
            normalized between min_weight and max_weight
        """
        # Create a copy to avoid modifying the original
        normalized = tensor.clone()
        
        # Create a mask of non-zero values
        nonzero_mask = normalized > 0
        
        # If there are no non-zero values, return the original tensor
        if not nonzero_mask.any():
            return normalized
        
        # Extract just the non-zero values
        nonzero_values = normalized[nonzero_mask]
        
        # Normalize the non-zero values between min_weight and max_weight
        if nonzero_values.max() == nonzero_values.min():
            # If all non-zero values are the same, set them all to max_weight
            nonzero_values = torch.ones_like(nonzero_values) * high
        else:
            # Otherwise, normalize them
            nonzero_values = low + (nonzero_values - nonzero_values.min()) / \
                            (nonzero_values.max() - nonzero_values.min()) * (high - low)
        
        # Put the normalized values back in the right places
        normalized[nonzero_mask] = nonzero_values
        
        return normalized
    
    # make a function that takes a list of layers and a list of aggregation methods and returns the stacked and averaged gradcam
    def generate_stacked_gradcam(self, input_tensor, target_class, layers,
                                 gradient_aggregation_method, stack_aggregation_method,
                                layer_aggregation_method, stack_relu, winsor_percentile=99,
                                interpolation_mode='nearest'):

        # Forward pass
        self.eval()
        output = self(input_tensor)

        # turn the logits to probabilities
        outputs = torch.sigmoid(logits)
        # turn the probabilities to one hot encoded labels
        outputs = self.model(input_tensor)
        logits = outputs.logits

        # turn the logits to probabilities
        probabilities = torch.sigmoid(logits)
        # turn the probabilities to one hot encoded labels
        outputs = (probabilities > 0.5).float()

        # Zero gradients
        self.zero_grad()

        # Backward pass on the target class
        # this is now inception v3 so I need to get the target class from the logits
        target = outputs[0, target_class]
        target.backward()

        # Get the activations and gradients for the target layer
        activations = [self.storage._storage[layer_name]['activations'] for layer_name in layers]
        gradients = [self.storage._storage[layer_name]['gradients'] for layer_name in layers]
        # print(f"activation shapes: {[activation.shape for activation in activations]}") #activation shapes: [torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 64, 56, 56])...
        # print(f"gradient shapes: {[gradient.shape for gradient in gradients]}") # gradient shapes: [torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 64, 56, 56])...

        # raise ValueError("stop here so I can see the gradients")


        if any(torch.isnan(grad).any() or torch.isinf(grad).any() for grad in gradients):
            print(f'Gradients for layer {layer_name} contain NaN or Inf values')

        largest_map = max([layer["activations"].shape[2:] for layer in self.storage._storage.values()])

        filter_importances = []
        for layer_name, gradients in zip(layers, gradients):
            match gradient_aggregation_method:
                case 'mean':
                    # Global average pooling
                    # This will show the average importance of each filter
                    filter_importances.append(torch.mean(gradients, dim=(2, 3))[0, :])
                case 'max':
                    # Global max pooling
                    # This will show the filter with the most importance
                    filter_importances.append(torch.amax(gradients, dim=[0, 2, 3]))
                case 'min': 
                    # Global min pooling
                    # This will show the filter with the least importance
                    filter_importances.append(torch.amin(gradients, dim=[0, 2, 3]))
                case 'absmax':
                    # This will show the filter with the most importance regardless of the sign
                    # can be less sensitive to the noise or outliers
                    filter_importances.append(torch.amax(torch.abs(gradients), dim=[0, 2, 3]))
                case 'sum':
                    # This will show the cumulative importance of each filter
                    filter_importances.append(torch.sum(gradients, dim=[0, 2, 3]))
                case 'l2norm':
                    # similar to the absmax but it show magnitude of the importance
                    # can be less sensitive to the noise or outliers
                    filter_importances.append(torch.norm(gradients, p=2, dim=[0, 2, 3]))
                case "absmean":

                    filter_importances.append(torch.mean(torch.abs(gradients), dim=[0, 2, 3]))
                case "variance":
                    # Global variance pooling
                    # Spread of influence within a feature map
                    filter_importances.append(torch.var(gradients, dim=[0, 2, 3]))
                case "std":
                    filter_importances.append(torch.std(gradients, dim=[0, 2, 3]))

                case "kurtosis":
                    # Global kurtosis pooling
                    # Sharpness of the importance distribution
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
                    deviations = gradients - mean_gradients
                    fourth_moment = torch.mean(deviations ** 4, dim=[0, 2, 3])
                    second_moment = torch.mean(deviations ** 2, dim=[0, 2, 3])
                    filter_importance = fourth_moment / (second_moment ** 2)
                    # as kurtosis divides it is possible to get a division by zero I need to handle this
                    filter_importance[torch.isnan(filter_importance)] = 0
                    filter_importances.append(filter_importance)
                case "huber":
                    # Huber loss pooling
                    # This is a robust loss function that is less sensitive to outliers
                    # Step 1: Apply the Huber loss function element-wise
                    huber_loss = torch.where(
                        torch.abs(gradients) <= gradients,
                        0.5 * gradients**2,
                        gradients * (torch.abs(gradients) - 0.5 * gradients)
                    )
                    
                    # Step 2: Mask out zero values
                    nonzero_mask = gradients != 0
                    huber_loss = huber_loss * nonzero_mask  # Exclude zeros from the loss calculation
                    
                    # Step 3: Pool the values along the specified dimensions (dim=[0, 2, 3])
                    # Using mean pooling but with the Huber loss instead of raw values
                    filter_importances.append(torch.mean(huber_loss, dim=[0, 2, 3]))
                # make one that only shows the single most important filter
                case "singlemax":
                    # this would need to put a 1 where the max is and 0 everywhere else
                    # find the sum of each layers gradients
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
                    # find the max of the sum
                    max_index = torch.argmax(mean_gradients)
                    # create a tensor with 1 where the max is and 0 everywhere else
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[max_index] = 1
                    filter_importances.append(filter_importance)
                case "singlemin":
                    
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
                    min_index = torch.argmin(mean_gradients)
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[min_index] = 1
                    filter_importances.append(filter_importance)
                case _:
                    raise ValueError(f"Invalid gradient_aggregation_method: {gradient_aggregation_method}")

        # for each layer produce a gradcam
        gradcams, importance_lists = self.generate_gradcam(filter_importances, activations)
        
        gradcams = [self.normalize_tensor(gradcam, high=1, low=0) for gradcam in gradcams]
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
                # calculate the entropy of the importance list
                # this is a measure of uncertainty
                # high entropy means that the model is uncertain about the importance
                # low entropy means that the model is certain about the importance
                # Convert gradients to valid probabilities using softmax
                importance_lists = [torch.distributions.Categorical(
                    probs=torch.nn.functional.softmax(importance_list, dim=0)
                ).entropy() for importance_list in importance_lists]
                
            case _:
                raise ValueError(f"Invalid layer_aggregation_method: {layer_aggregation_method}")
            

        importance_tensor = torch.stack(importance_lists)
        importance_tensor = torch.relu(importance_tensor) if stack_relu else importance_tensor
        # print(f"importance_tensor min:{importance_tensor.min()}, max:{importance_tensor.max()}")
        # # Clamp the importance tensor based on the standard deviation threshold
        importance_tensor = self.winsorize_preserve_zeros(importance_tensor, percentile=winsor_percentile)


        gradcams = [torch.nn.functional.interpolate(gradcam.unsqueeze(0).unsqueeze(0), size=largest_map, mode=interpolation_mode).squeeze() for gradcam in gradcams]
        # stack the gradcams with a relu to prevent negative values from cancelling out positive values
        stacked_gradcam = torch.stack(gradcams)


        if len(gradcams) == 1:
            stacked_gradcam = F.interpolate(stacked_gradcam.unsqueeze(0), size=input_tensor.shape[2:], mode=interpolation_mode).squeeze()
            return stacked_gradcam, gradcams, importance_tensor


        match stack_aggregation_method:
            case 'mean':
                # This might show the average place where the model is looking
                stacked_gradcam = torch.mean(stacked_gradcam, dim=0)
            case 'sum':
                # This might show every place where the model is looking
                stacked_gradcam = torch.sum(stacked_gradcam, dim=0)
            case "absmean":
                # Based on a normalization between -1 and 1
                # This could show us where the model seeing both very important and very unimportant areas as high intensity
                # while if it is somewhere in the middle it is less intense
                stacked_gradcam = torch.mean(torch.abs(stacked_gradcam), dim=0)
            case 'max':
                # This might show where the model look the most for information
                # It can be thought of as a measurement where any of the layers see anything
                # and can show where the model has not looked at at all
                stacked_gradcam, _ = torch.max(stacked_gradcam, dim=0)
            case 'min':
                # shows where the model is looking at typically
                # It can also be thought of as a measurement of consistency across layers
                # modify stacked_gradcam so that it only has values that are greater than 0
                stacked_gradcam = torch.relu(stacked_gradcam)
                # remove all maps that only contain 0s
                stacked_gradcam, _ = torch.min(stacked_gradcam, dim=0)
            case 'absmax':
                # This might show importace of the most important and least important areas as the data is normalized between -1 and 1
                # similar to absmean but rather than averaging we see the maximum (so more intensity in general)
                stacked_gradcam, _ = torch.max(torch.abs(stacked_gradcam), dim=0)
            case 'sum':
                # This shows every place where the model is looking
                # It also shows the cumulative effect of the layers on the final output
                stacked_gradcam = torch.sum(stacked_gradcam, dim=0)
            case 'l2norm':
                # This gives a sense of the overall magnitude considering the strength of activations across maps.
                stacked_gradcam = torch.norm(stacked_gradcam, p=2, dim=0)
            case "l1norm":
                # This gives a sense of differenc considering the strength of activations across maps.
                stacked_gradcam = torch.norm(stacked_gradcam, p=1, dim=0)
            case "variance":
                # This might show where the model is having the most amount difficulty agreeing based on the variance
                # Low values would typically show that the model is in agreement about a particular area
                # High values would show that the model is having a hard time agreeing
                stacked_gradcam = torch.var(stacked_gradcam, dim=0)
            case "std":
                # This might show where the model is having the most amount difficulty agreeing based on the standard deviation
                # Low values would typically show that the model is in agreement about a particular area
                # High values would show that the model is having a hard time agreeing
                stacked_gradcam = torch.std(stacked_gradcam, dim=0)
            case "kurtosis":
                # This might show where the model is having the most amount difficulty agreeing based on the kurtosis
                # Low values would typically show that the model is in agreement about a particular area
                # High values would show that the model is having a hard time agreeing
                mean_stacked_gradcam = torch.mean(stacked_gradcam, dim=0, keepdim=True)
                deviations = stacked_gradcam - mean_stacked_gradcam
                fourth_moment = torch.mean(deviations ** 4, dim=0)
                second_moment = torch.mean(deviations ** 2, dim=0)
                stacked_gradcam = fourth_moment / (second_moment ** 2)
                # as kurtosis divides it is possible to get a division by zero I need to handle this
                stacked_gradcam[torch.isnan(stacked_gradcam)] = 0
            case "huber":
                # Huber loss pooling
                # This is a robust loss function that is less sensitive to outliers
                # Step 1: Apply the Huber loss function element-wise
                huber_loss = torch.where(
                    torch.abs(stacked_gradcam) <= stacked_gradcam,
                    0.5 * stacked_gradcam**2,
                    stacked_gradcam * (torch.abs(stacked_gradcam) - 0.5 * stacked_gradcam)
                )
                
                # Step 2: Mask out zero values
                nonzero_mask = stacked_gradcam != 0
                huber_loss = huber_loss * nonzero_mask
                stacked_gradcam = torch.mean(huber_loss, dim=0)
            case "weighted":
                # This is a weighted sum of the gradcams
                # the weights are the average filter importance
                # Convert list to tensor and normalize between .1 and 1 avoiding getting rid of zeros
                normalized_importance = self.normalize_nonzero(importance_tensor, high=1, low=.1)
                # Apply weights to stacked gradcam
                stacked_gradcam = torch.sum(stacked_gradcam * normalized_importance[:, None, None], dim=0)
            case "weighted_inverted":
                # This need to take the inverse of the importance tensor
                # Convert list to tensor and normalize between 0 and 1
                normalized_importance = self.normalize_tensor(importance_tensor, high=1, low=0)
                # invert the importance tensor
                inverted_importance = 1 - normalized_importance
                # Apply weights to stacked gradcam
                stacked_gradcam = torch.sum(stacked_gradcam * inverted_importance[:, None, None], dim=0)
                # change the importance tensor to the inverted importance tensor
                importance_tensor = inverted_importance
            case "singlemax":
                # find the filter that has the most importance based on the average filter importance
                max_index = torch.argmax(importance_tensor)
                # get the gradcam for that layer
                stacked_gradcam = gradcams[max_index]
            case "singlemin":
                # find the filter that has the least importance based on the average filter importance
                min_index = torch.argmin(importance_tensor)
                # get the gradcam for that layer
                stacked_gradcam = gradcams[min_index]
            case "count":
                # this will be the count of occurences that are greater than 0
                nonzero_masks = [torch.where(gradcam > 0, torch.tensor(1), torch.tensor(0)) for gradcam in gradcams]
                stacked_gradcam = torch.sum(torch.stack(nonzero_masks), dim=0, dtype=torch.float32)

            case _:
                raise ValueError(f"Invalid stack_aggregation_method: {stack_aggregation_method}")
            
        # resize the gradcam to the size of the image
        stacked_gradcam = F.interpolate(stacked_gradcam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[2:], mode=interpolation_mode).squeeze()
        self.storage.clear()
        return stacked_gradcam, gradcams, importance_tensor



        # make a function that takes a list of layers and a list of aggregation methods and returns the stacked and averaged gradcam
    
    def get_gradcams_and_importance(self, input_tensor, target_class, layers,
                                 gradient_aggregation_method,
                                layer_aggregation_method, stack_relu,
                                interpolation_mode='nearest'):

        # Forward pass
        self.eval()
        output = self(input_tensor)

        # Zero gradients
        self.zero_grad()

        # Backward pass
        target = output[0, target_class]
        target.backward()

        # Get the activations and gradients for the target layer without the layers that say AuxLogits
        activations = [self.storage._storage[layer_name]['activations'] for layer_name in layers if "AuxLogits" not in layer_name]
        gradients = [self.storage._storage[layer_name]['gradients'] for layer_name in layers if "AuxLogits" not in layer_name]
        # print(f"activation shapes: {[activation.shape for activation in activations]}") #activation shapes: [torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 64, 56, 56])...
        # print(f"gradient shapes: {[gradient.shape for gradient in gradients]}") # gradient shapes: [torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 64, 56, 56])...

        # raise ValueError("stop here so I can see the gradients")


        if any(torch.isnan(grad).any() or torch.isinf(grad).any() for grad in gradients):
            print(f'Gradients for layer {layer_name} contain NaN or Inf values')

        largest_map = max([layer["activations"].shape[2:] for layer in self.storage._storage.values()])

        filter_importances = []
        for layer_name, gradients in zip(layers, gradients):
            match gradient_aggregation_method:
                case 'mean':
                    # Global average pooling
                    # This will show the average importance of each filter
                    filter_importances.append(torch.mean(gradients, dim=(2, 3))[0, :])
                case 'max':
                    # Global max pooling
                    # This will show the filter with the most importance
                    filter_importances.append(torch.amax(gradients, dim=[0, 2, 3]))
                case 'min': 
                    # Global min pooling
                    # This will show the filter with the least importance
                    filter_importances.append(torch.amin(gradients, dim=[0, 2, 3]))
                case 'absmax':
                    # This will show the filter with the most importance regardless of the sign
                    # can be less sensitive to the noise or outliers
                    filter_importances.append(torch.amax(torch.abs(gradients), dim=[0, 2, 3]))
                case 'sum':
                    # This will show the cumulative importance of each filter
                    filter_importances.append(torch.sum(gradients, dim=[0, 2, 3]))
                case 'l2norm':
                    # similar to the absmax but it show magnitude of the importance
                    # can be less sensitive to the noise or outliers
                    filter_importances.append(torch.norm(gradients, p=2, dim=[0, 2, 3]))
                case "absmean":

                    filter_importances.append(torch.mean(torch.abs(gradients), dim=[0, 2, 3]))
                case "variance":
                    # Global variance pooling
                    # Spread of influence within a feature map
                    filter_importances.append(torch.var(gradients, dim=[0, 2, 3]))
                case "std":
                    filter_importances.append(torch.std(gradients, dim=[0, 2, 3]))

                case "kurtosis":
                    # Global kurtosis pooling
                    # Sharpness of the importance distribution
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
                    deviations = gradients - mean_gradients
                    fourth_moment = torch.mean(deviations ** 4, dim=[0, 2, 3])
                    second_moment = torch.mean(deviations ** 2, dim=[0, 2, 3])
                    filter_importance = fourth_moment / (second_moment ** 2)
                    # as kurtosis divides it is possible to get a division by zero I need to handle this
                    filter_importance[torch.isnan(filter_importance)] = 0
                    filter_importances.append(filter_importance)
                case "huber":
                    # Huber loss pooling
                    # This is a robust loss function that is less sensitive to outliers
                    # Step 1: Apply the Huber loss function element-wise
                    huber_loss = torch.where(
                        torch.abs(gradients) <= gradients,
                        0.5 * gradients**2,
                        gradients * (torch.abs(gradients) - 0.5 * gradients)
                    )
                    
                    # Step 2: Mask out zero values
                    nonzero_mask = gradients != 0
                    huber_loss = huber_loss * nonzero_mask  # Exclude zeros from the loss calculation
                    
                    # Step 3: Pool the values along the specified dimensions (dim=[0, 2, 3])
                    # Using mean pooling but with the Huber loss instead of raw values
                    filter_importances.append(torch.mean(huber_loss, dim=[0, 2, 3]))
                # make one that only shows the single most important filter
                case "singlemax":
                    # this would need to put a 1 where the max is and 0 everywhere else
                    # find the sum of each layers gradients
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
                    # find the max of the sum
                    max_index = torch.argmax(mean_gradients)
                    # create a tensor with 1 where the max is and 0 everywhere else
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[max_index] = 1
                    filter_importances.append(filter_importance)
                case "singlemin":
                    
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
                    min_index = torch.argmin(mean_gradients)
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[min_index] = 1
                    filter_importances.append(filter_importance)
                case _:
                    raise ValueError(f"Invalid gradient_aggregation_method: {gradient_aggregation_method}")

        # for each layer produce a gradcam
        gradcams, importance_lists = self.generate_gradcam(filter_importances, activations)
        
        gradcams = [self.normalize_tensor(gradcam, high=1, low=0) for gradcam in gradcams]
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
                # calculate the entropy of the importance list
                # this is a measure of uncertainty
                # high entropy means that the model is uncertain about the importance
                # low entropy means that the model is certain about the importance
                # Convert gradients to valid probabilities using softmax
                importance_lists = [torch.distributions.Categorical(
                    probs=torch.nn.functional.softmax(importance_list, dim=0)
                ).entropy() for importance_list in importance_lists]
                
            case _:
                raise ValueError(f"Invalid layer_aggregation_method: {layer_aggregation_method}")
            

        importance_tensor = torch.stack(importance_lists)
        importance_tensor = torch.relu(importance_tensor) if stack_relu else importance_tensor

        gradcams = [torch.nn.functional.interpolate(gradcam.unsqueeze(0).unsqueeze(0), size=largest_map, mode=interpolation_mode).squeeze() for gradcam in gradcams]
        # stack the gradcams with a relu to prevent negative values from cancelling out positive values
        stacked_gradcam = torch.stack(gradcams)


        if len(gradcams) == 1:
            stacked_gradcam = F.interpolate(stacked_gradcam.unsqueeze(0), size=input_tensor.shape[2:], mode=interpolation_mode).squeeze()
            return stacked_gradcam, gradcams, importance_tensor
        
        return stacked_gradcam, gradcams, importance_tensor

    # now I need a function that takes in stacked_gradcam, gradcams, importance_tensor, and winsor_percentile
    # then outputs the stacked_gradcam (this is for efficiency when showing how winsorization works)
    def winsorize_stacked_gradcam(self,input_tensor, stacked_gradcam, gradcams, importance_tensor, interpolation_mode='nearest', winsor_percentile=99):
        importance_tensor = self.winsorize_preserve_zeros(importance_tensor, percentile=winsor_percentile)
        # This is a weighted sum of the gradcams
        normalized_importance = self.normalize_nonzero(importance_tensor, high=1, low=.1)
        # Apply weights to stacked gradcam
        stacked_gradcam = torch.sum(stacked_gradcam * normalized_importance[:, None, None], dim=0)
        # resize the gradcam to the size of the image
                # resize the gradcam to the size of the image
        stacked_gradcam = F.interpolate(stacked_gradcam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[2:], mode=interpolation_mode).squeeze()
        self.storage.clear()
        return stacked_gradcam, importance_tensor
        

    def generate_saliency_map(self, input_tensor, target_class):
        self.eval()
        input_tensor.requires_grad_()
        # Unregister ReLU hooks to get raw gradients
        self._unregister_hooks()
        
        # Forward pass
        output = self(input_tensor)
        self.zero_grad()
        self.storage.clear()
        # Backward pass on a single target class
        target = output[0, target_class]
        target.backward()
        # Get the gradients
        # these will be the gradients of the loss with respect to the input
        saliency_map = input_tensor.grad.data
        # take only the absolute value of the gradients
        saliency_map = torch.abs(saliency_map)
        # Normalize the saliency map between 0 and 1
        saliency_map = self.normalize_tensor(saliency_map, high=1, low=0)

        # squeeze the tensor to remove the channel dimension
        saliency_map = saliency_map.squeeze()
        # Re-register ReLU hooks for guided backpropagation
        self._register_hooks()
        # get the average across the channels
        saliency_map = torch.mean(saliency_map, dim=0)

        return saliency_map

    def forward(self, x):
        return self.base_model(x)
        # return self.resnet50(x)

    def normalize_tensor(self, tensor, high=1, low=-1):
        # as this will divide handle if the case is that all are 0
        if ((tensor.max() - tensor.min()) * (high - low)) == 0:
            return torch.zeros_like(tensor)
        return low + (tensor - tensor.min()) / (tensor.max() - tensor.min()) * (high - low)
    def __del__(self):
        self._unregister_hooks()
    def remove_hooks(self):
        self._unregister_hooks()


class VGG16Modified(nn.Module):
    def __init__(self, model_path=None, dataset=None):
        super().__init__()
        if model_path is None:
            model = vgg16(weights='IMAGENET1K_V1')
        else:
            model = vgg16(weights=None)
        # change the last layer to be the number of classes
        if dataset == "pascal_voc":
            num_classes = 20
            model.classifier[6] =  nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(4096, num_classes)
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load the model
        if model_path is not None:
            if device.type == "cpu":
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
            else:
                model.load_state_dict(torch.load(model_path))
        self.base_model = model
        self.storage = ActivationGradientStorage()
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output, name):
            self.storage.store_activation(name, output)
            
        # Remove backward hooks - use output.register_hook instead
        # for name, module in self.resnet50.named_modules():
        #     if isinstance(module, nn.Conv2d):
        #         full_name = f"resnet50.{name}"
        #         self.hooks.append(
        #             module.register_forward_hook(
        #                 partial(forward_hook, name=full_name)
        #             )
        #         )
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Conv2d):
                full_name = f"base_model.{name}"
                self.hooks.append(
                    module.register_forward_hook(
                        partial(forward_hook, name=full_name)
                    )
                )
        
                
    def _unregister_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def __call__(self, x):
        self.storage.clear()  # Clear before forward pass
        return super().__call__(x)  # Use parent's __call__ to avoid recursion


    def generate_gradcam(self, filter_importances, activations):
        # print(f"activations at max location: {activations[0, torch.argmax(filter_importance)]}")
        gradcams = []
        for filter_importance, activations in zip(filter_importances, activations):
            gradcam = torch.sum(filter_importance[:, None, None] * activations.squeeze(), dim=0)
            # print(f"gradcam min:{gradcam.min()}, max:{gradcam.max()}")
            # Apply ReLU
            # the reason you apply ReLU is to remove the negative values
            gradcam = torch.relu(gradcam)
            gradcams.append(gradcam)
        return gradcams, filter_importances
    
    @staticmethod
    def winsorize_preserve_zeros(tensor, percentile=99):
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
        """
        Normalize only non-zero values in the importance tensor to ensure all maps contribute.
        
        Args:
            importance_tensor: Tensor of importance values
            min_weight: Minimum weight for non-zero values after normalization
            max_weight: Maximum weight for non-zero values after normalization
            
        Returns:
            Normalized tensor where zeros stay zero, and all other values are
            normalized between min_weight and max_weight
        """
        # Create a copy to avoid modifying the original
        normalized = tensor.clone()
        
        # Create a mask of non-zero values
        nonzero_mask = normalized > 0
        
        # If there are no non-zero values, return the original tensor
        if not nonzero_mask.any():
            return normalized
        
        # Extract just the non-zero values
        nonzero_values = normalized[nonzero_mask]
        
        # Normalize the non-zero values between min_weight and max_weight
        if nonzero_values.max() == nonzero_values.min():
            # If all non-zero values are the same, set them all to max_weight
            nonzero_values = torch.ones_like(nonzero_values) * high
        else:
            # Otherwise, normalize them
            nonzero_values = low + (nonzero_values - nonzero_values.min()) / \
                            (nonzero_values.max() - nonzero_values.min()) * (high - low)
        
        # Put the normalized values back in the right places
        normalized[nonzero_mask] = nonzero_values
        
        return normalized
    
    # make a function that takes a list of layers and a list of aggregation methods and returns the stacked and averaged gradcam
    def generate_stacked_gradcam(self, input_tensor, target_class, layers,
                                 gradient_aggregation_method, stack_aggregation_method,
                                layer_aggregation_method, stack_relu, winsor_percentile=99,
                                interpolation_mode='nearest'):

        # Forward pass
        self.eval()
        output = self(input_tensor)

        # Zero gradients
        self.zero_grad()

        # Backward pass
        target = output[0, target_class]
        target.backward()

        # Get the activations and gradients for the target layer
        activations = [self.storage._storage[layer_name]['activations'] for layer_name in layers]
        gradients = [self.storage._storage[layer_name]['gradients'] for layer_name in layers]
        # print(f"activation shapes: {[activation.shape for activation in activations]}") #activation shapes: [torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 64, 56, 56])...
        # print(f"gradient shapes: {[gradient.shape for gradient in gradients]}") # gradient shapes: [torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 64, 56, 56])...

        # raise ValueError("stop here so I can see the gradients")


        if any(torch.isnan(grad).any() or torch.isinf(grad).any() for grad in gradients):
            print(f'Gradients for layer {layer_name} contain NaN or Inf values')

        largest_map = max([layer["activations"].shape[2:] for layer in self.storage._storage.values()])

        filter_importances = []
        for layer_name, gradients in zip(layers, gradients):
            match gradient_aggregation_method:
                case 'mean':
                    # Global average pooling
                    # This will show the average importance of each filter
                    filter_importances.append(torch.mean(gradients, dim=(2, 3))[0, :])
                case 'max':
                    # Global max pooling
                    # This will show the filter with the most importance
                    filter_importances.append(torch.amax(gradients, dim=[0, 2, 3]))
                case 'min': 
                    # Global min pooling
                    # This will show the filter with the least importance
                    filter_importances.append(torch.amin(gradients, dim=[0, 2, 3]))
                case 'absmax':
                    # This will show the filter with the most importance regardless of the sign
                    # can be less sensitive to the noise or outliers
                    filter_importances.append(torch.amax(torch.abs(gradients), dim=[0, 2, 3]))
                case 'sum':
                    # This will show the cumulative importance of each filter
                    filter_importances.append(torch.sum(gradients, dim=[0, 2, 3]))
                case 'l2norm':
                    # similar to the absmax but it show magnitude of the importance
                    # can be less sensitive to the noise or outliers
                    filter_importances.append(torch.norm(gradients, p=2, dim=[0, 2, 3]))
                case "absmean":

                    filter_importances.append(torch.mean(torch.abs(gradients), dim=[0, 2, 3]))
                case "variance":
                    # Global variance pooling
                    # Spread of influence within a feature map
                    filter_importances.append(torch.var(gradients, dim=[0, 2, 3]))
                case "std":
                    filter_importances.append(torch.std(gradients, dim=[0, 2, 3]))

                case "kurtosis":
                    # Global kurtosis pooling
                    # Sharpness of the importance distribution
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
                    deviations = gradients - mean_gradients
                    fourth_moment = torch.mean(deviations ** 4, dim=[0, 2, 3])
                    second_moment = torch.mean(deviations ** 2, dim=[0, 2, 3])
                    filter_importance = fourth_moment / (second_moment ** 2)
                    # as kurtosis divides it is possible to get a division by zero I need to handle this
                    filter_importance[torch.isnan(filter_importance)] = 0
                    filter_importances.append(filter_importance)
                case "huber":
                    # Huber loss pooling
                    # This is a robust loss function that is less sensitive to outliers
                    # Step 1: Apply the Huber loss function element-wise
                    huber_loss = torch.where(
                        torch.abs(gradients) <= gradients,
                        0.5 * gradients**2,
                        gradients * (torch.abs(gradients) - 0.5 * gradients)
                    )
                    
                    # Step 2: Mask out zero values
                    nonzero_mask = gradients != 0
                    huber_loss = huber_loss * nonzero_mask  # Exclude zeros from the loss calculation
                    
                    # Step 3: Pool the values along the specified dimensions (dim=[0, 2, 3])
                    # Using mean pooling but with the Huber loss instead of raw values
                    filter_importances.append(torch.mean(huber_loss, dim=[0, 2, 3]))
                # make one that only shows the single most important filter
                case "singlemax":
                    # this would need to put a 1 where the max is and 0 everywhere else
                    # find the sum of each layers gradients
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
                    # find the max of the sum
                    max_index = torch.argmax(mean_gradients)
                    # create a tensor with 1 where the max is and 0 everywhere else
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[max_index] = 1
                    filter_importances.append(filter_importance)
                case "singlemin":
                    
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
                    min_index = torch.argmin(mean_gradients)
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[min_index] = 1
                    filter_importances.append(filter_importance)
                case _:
                    raise ValueError(f"Invalid gradient_aggregation_method: {gradient_aggregation_method}")

        # for each layer produce a gradcam
        gradcams, importance_lists = self.generate_gradcam(filter_importances, activations)
        
        gradcams = [self.normalize_tensor(gradcam, high=1, low=0) for gradcam in gradcams]
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
                # calculate the entropy of the importance list
                # this is a measure of uncertainty
                # high entropy means that the model is uncertain about the importance
                # low entropy means that the model is certain about the importance
                # Convert gradients to valid probabilities using softmax
                importance_lists = [torch.distributions.Categorical(
                    probs=torch.nn.functional.softmax(importance_list, dim=0)
                ).entropy() for importance_list in importance_lists]
                
            case _:
                raise ValueError(f"Invalid layer_aggregation_method: {layer_aggregation_method}")
            

        importance_tensor = torch.stack(importance_lists)
        importance_tensor = torch.relu(importance_tensor) if stack_relu else importance_tensor
        # print(f"importance_tensor min:{importance_tensor.min()}, max:{importance_tensor.max()}")
        # # Clamp the importance tensor based on the standard deviation threshold
        importance_tensor = self.winsorize_preserve_zeros(importance_tensor, percentile=winsor_percentile)


        gradcams = [torch.nn.functional.interpolate(gradcam.unsqueeze(0).unsqueeze(0), size=largest_map, mode=interpolation_mode).squeeze() for gradcam in gradcams]
        # stack the gradcams with a relu to prevent negative values from cancelling out positive values
        stacked_gradcam = torch.stack(gradcams)


        if len(gradcams) == 1:
            stacked_gradcam = F.interpolate(stacked_gradcam.unsqueeze(0), size=input_tensor.shape[2:], mode=interpolation_mode).squeeze()
            return stacked_gradcam, gradcams, importance_tensor


        match stack_aggregation_method:
            case 'mean':
                # This might show the average place where the model is looking
                stacked_gradcam = torch.mean(stacked_gradcam, dim=0)
            case 'sum':
                # This might show every place where the model is looking
                stacked_gradcam = torch.sum(stacked_gradcam, dim=0)
            case "absmean":
                # Based on a normalization between -1 and 1
                # This could show us where the model seeing both very important and very unimportant areas as high intensity
                # while if it is somewhere in the middle it is less intense
                stacked_gradcam = torch.mean(torch.abs(stacked_gradcam), dim=0)
            case 'max':
                # This might show where the model look the most for information
                # It can be thought of as a measurement where any of the layers see anything
                # and can show where the model has not looked at at all
                stacked_gradcam, _ = torch.max(stacked_gradcam, dim=0)
            case 'min':
                # shows where the model is looking at typically
                # It can also be thought of as a measurement of consistency across layers
                # modify stacked_gradcam so that it only has values that are greater than 0
                stacked_gradcam = torch.relu(stacked_gradcam)
                # remove all maps that only contain 0s
                stacked_gradcam, _ = torch.min(stacked_gradcam, dim=0)
            case 'absmax':
                # This might show importace of the most important and least important areas as the data is normalized between -1 and 1
                # similar to absmean but rather than averaging we see the maximum (so more intensity in general)
                stacked_gradcam, _ = torch.max(torch.abs(stacked_gradcam), dim=0)
            case 'sum':
                # This shows every place where the model is looking
                # It also shows the cumulative effect of the layers on the final output
                stacked_gradcam = torch.sum(stacked_gradcam, dim=0)
            case 'l2norm':
                # This gives a sense of the overall magnitude considering the strength of activations across maps.
                stacked_gradcam = torch.norm(stacked_gradcam, p=2, dim=0)
            case "l1norm":
                # This gives a sense of differenc considering the strength of activations across maps.
                stacked_gradcam = torch.norm(stacked_gradcam, p=1, dim=0)
            case "variance":
                # This might show where the model is having the most amount difficulty agreeing based on the variance
                # Low values would typically show that the model is in agreement about a particular area
                # High values would show that the model is having a hard time agreeing
                stacked_gradcam = torch.var(stacked_gradcam, dim=0)
            case "std":
                # This might show where the model is having the most amount difficulty agreeing based on the standard deviation
                # Low values would typically show that the model is in agreement about a particular area
                # High values would show that the model is having a hard time agreeing
                stacked_gradcam = torch.std(stacked_gradcam, dim=0)
            case "kurtosis":
                # This might show where the model is having the most amount difficulty agreeing based on the kurtosis
                # Low values would typically show that the model is in agreement about a particular area
                # High values would show that the model is having a hard time agreeing
                mean_stacked_gradcam = torch.mean(stacked_gradcam, dim=0, keepdim=True)
                deviations = stacked_gradcam - mean_stacked_gradcam
                fourth_moment = torch.mean(deviations ** 4, dim=0)
                second_moment = torch.mean(deviations ** 2, dim=0)
                stacked_gradcam = fourth_moment / (second_moment ** 2)
                # as kurtosis divides it is possible to get a division by zero I need to handle this
                stacked_gradcam[torch.isnan(stacked_gradcam)] = 0
            case "huber":
                # Huber loss pooling
                # This is a robust loss function that is less sensitive to outliers
                # Step 1: Apply the Huber loss function element-wise
                huber_loss = torch.where(
                    torch.abs(stacked_gradcam) <= stacked_gradcam,
                    0.5 * stacked_gradcam**2,
                    stacked_gradcam * (torch.abs(stacked_gradcam) - 0.5 * stacked_gradcam)
                )
                
                # Step 2: Mask out zero values
                nonzero_mask = stacked_gradcam != 0
                huber_loss = huber_loss * nonzero_mask
                stacked_gradcam = torch.mean(huber_loss, dim=0)
            case "weighted":
                # This is a weighted sum of the gradcams
                # the weights are the average filter importance
                # Convert list to tensor and normalize between .1 and 1 avoiding getting rid of zeros
                normalized_importance = self.normalize_nonzero(importance_tensor, high=1, low=.1)
                # Apply weights to stacked gradcam
                stacked_gradcam = torch.sum(stacked_gradcam * normalized_importance[:, None, None], dim=0)
            case "weighted_inverted":
                # This need to take the inverse of the importance tensor
                # Convert list to tensor and normalize between 0 and 1
                normalized_importance = self.normalize_tensor(importance_tensor, high=1, low=0)
                # invert the importance tensor
                inverted_importance = 1 - normalized_importance
                # Apply weights to stacked gradcam
                stacked_gradcam = torch.sum(stacked_gradcam * inverted_importance[:, None, None], dim=0)
                # change the importance tensor to the inverted importance tensor
                importance_tensor = inverted_importance
            case "singlemax":
                # find the filter that has the most importance based on the average filter importance
                max_index = torch.argmax(importance_tensor)
                # get the gradcam for that layer
                stacked_gradcam = gradcams[max_index]
            case "singlemin":
                # find the filter that has the least importance based on the average filter importance
                min_index = torch.argmin(importance_tensor)
                # get the gradcam for that layer
                stacked_gradcam = gradcams[min_index]
            case "count":
                # this will be the count of occurences that are greater than 0
                nonzero_masks = [torch.where(gradcam > 0, torch.tensor(1), torch.tensor(0)) for gradcam in gradcams]
                stacked_gradcam = torch.sum(torch.stack(nonzero_masks), dim=0, dtype=torch.float32)

            case _:
                raise ValueError(f"Invalid stack_aggregation_method: {stack_aggregation_method}")
            
        # resize the gradcam to the size of the image
        stacked_gradcam = F.interpolate(stacked_gradcam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[2:], mode=interpolation_mode).squeeze()
        self.storage.clear()
        return stacked_gradcam, gradcams, importance_tensor



        # make a function that takes a list of layers and a list of aggregation methods and returns the stacked and averaged gradcam
    
    def get_gradcams_and_importance(self, input_tensor, target_class, layers,
                                 gradient_aggregation_method,
                                layer_aggregation_method, stack_relu,
                                interpolation_mode='nearest'):

        # Forward pass
        self.eval()
        output = self(input_tensor)

        # Zero gradients
        self.zero_grad()

        # Backward pass
        target = output[0, target_class]
        target.backward()

        # Get the activations and gradients for the target layer
        activations = [self.storage._storage[layer_name]['activations'] for layer_name in layers]
        gradients = [self.storage._storage[layer_name]['gradients'] for layer_name in layers]
        # print(f"activation shapes: {[activation.shape for activation in activations]}") #activation shapes: [torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 64, 56, 56])...
        # print(f"gradient shapes: {[gradient.shape for gradient in gradients]}") # gradient shapes: [torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 64, 56, 56])...

        # raise ValueError("stop here so I can see the gradients")


        if any(torch.isnan(grad).any() or torch.isinf(grad).any() for grad in gradients):
            print(f'Gradients for layer {layer_name} contain NaN or Inf values')

        largest_map = max([layer["activations"].shape[2:] for layer in self.storage._storage.values()])

        filter_importances = []
        for layer_name, gradients in zip(layers, gradients):
            match gradient_aggregation_method:
                case 'mean':
                    # Global average pooling
                    # This will show the average importance of each filter
                    filter_importances.append(torch.mean(gradients, dim=(2, 3))[0, :])
                case 'max':
                    # Global max pooling
                    # This will show the filter with the most importance
                    filter_importances.append(torch.amax(gradients, dim=[0, 2, 3]))
                case 'min': 
                    # Global min pooling
                    # This will show the filter with the least importance
                    filter_importances.append(torch.amin(gradients, dim=[0, 2, 3]))
                case 'absmax':
                    # This will show the filter with the most importance regardless of the sign
                    # can be less sensitive to the noise or outliers
                    filter_importances.append(torch.amax(torch.abs(gradients), dim=[0, 2, 3]))
                case 'sum':
                    # This will show the cumulative importance of each filter
                    filter_importances.append(torch.sum(gradients, dim=[0, 2, 3]))
                case 'l2norm':
                    # similar to the absmax but it show magnitude of the importance
                    # can be less sensitive to the noise or outliers
                    filter_importances.append(torch.norm(gradients, p=2, dim=[0, 2, 3]))
                case "absmean":

                    filter_importances.append(torch.mean(torch.abs(gradients), dim=[0, 2, 3]))
                case "variance":
                    # Global variance pooling
                    # Spread of influence within a feature map
                    filter_importances.append(torch.var(gradients, dim=[0, 2, 3]))
                case "std":
                    filter_importances.append(torch.std(gradients, dim=[0, 2, 3]))

                case "kurtosis":
                    # Global kurtosis pooling
                    # Sharpness of the importance distribution
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
                    deviations = gradients - mean_gradients
                    fourth_moment = torch.mean(deviations ** 4, dim=[0, 2, 3])
                    second_moment = torch.mean(deviations ** 2, dim=[0, 2, 3])
                    filter_importance = fourth_moment / (second_moment ** 2)
                    # as kurtosis divides it is possible to get a division by zero I need to handle this
                    filter_importance[torch.isnan(filter_importance)] = 0
                    filter_importances.append(filter_importance)
                case "huber":
                    # Huber loss pooling
                    # This is a robust loss function that is less sensitive to outliers
                    # Step 1: Apply the Huber loss function element-wise
                    huber_loss = torch.where(
                        torch.abs(gradients) <= gradients,
                        0.5 * gradients**2,
                        gradients * (torch.abs(gradients) - 0.5 * gradients)
                    )
                    
                    # Step 2: Mask out zero values
                    nonzero_mask = gradients != 0
                    huber_loss = huber_loss * nonzero_mask  # Exclude zeros from the loss calculation
                    
                    # Step 3: Pool the values along the specified dimensions (dim=[0, 2, 3])
                    # Using mean pooling but with the Huber loss instead of raw values
                    filter_importances.append(torch.mean(huber_loss, dim=[0, 2, 3]))
                # make one that only shows the single most important filter
                case "singlemax":
                    # this would need to put a 1 where the max is and 0 everywhere else
                    # find the sum of each layers gradients
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
                    # find the max of the sum
                    max_index = torch.argmax(mean_gradients)
                    # create a tensor with 1 where the max is and 0 everywhere else
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[max_index] = 1
                    filter_importances.append(filter_importance)
                case "singlemin":
                    
                    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
                    min_index = torch.argmin(mean_gradients)
                    filter_importance = torch.zeros_like(mean_gradients)
                    filter_importance[min_index] = 1
                    filter_importances.append(filter_importance)
                case _:
                    raise ValueError(f"Invalid gradient_aggregation_method: {gradient_aggregation_method}")

        # for each layer produce a gradcam
        gradcams, importance_lists = self.generate_gradcam(filter_importances, activations)
        
        gradcams = [self.normalize_tensor(gradcam, high=1, low=0) for gradcam in gradcams]
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
                # calculate the entropy of the importance list
                # this is a measure of uncertainty
                # high entropy means that the model is uncertain about the importance
                # low entropy means that the model is certain about the importance
                # Convert gradients to valid probabilities using softmax
                importance_lists = [torch.distributions.Categorical(
                    probs=torch.nn.functional.softmax(importance_list, dim=0)
                ).entropy() for importance_list in importance_lists]
                
            case _:
                raise ValueError(f"Invalid layer_aggregation_method: {layer_aggregation_method}")
            

        importance_tensor = torch.stack(importance_lists)
        importance_tensor = torch.relu(importance_tensor) if stack_relu else importance_tensor

        gradcams = [torch.nn.functional.interpolate(gradcam.unsqueeze(0).unsqueeze(0), size=largest_map, mode=interpolation_mode).squeeze() for gradcam in gradcams]
        # stack the gradcams with a relu to prevent negative values from cancelling out positive values
        stacked_gradcam = torch.stack(gradcams)


        if len(gradcams) == 1:
            stacked_gradcam = F.interpolate(stacked_gradcam.unsqueeze(0), size=input_tensor.shape[2:], mode=interpolation_mode).squeeze()
            return stacked_gradcam, gradcams, importance_tensor
        
        return stacked_gradcam, gradcams, importance_tensor

    # now I need a function that takes in stacked_gradcam, gradcams, importance_tensor, and winsor_percentile
    # then outputs the stacked_gradcam (this is for efficiency when showing how winsorization works)
    def winsorize_stacked_gradcam(self,input_tensor, stacked_gradcam, gradcams, importance_tensor, interpolation_mode='nearest', winsor_percentile=99):
        importance_tensor = self.winsorize_preserve_zeros(importance_tensor, percentile=winsor_percentile)
        # This is a weighted sum of the gradcams
        normalized_importance = self.normalize_nonzero(importance_tensor, high=1, low=.1)
        # Apply weights to stacked gradcam
        stacked_gradcam = torch.sum(stacked_gradcam * normalized_importance[:, None, None], dim=0)
        # resize the gradcam to the size of the image
                # resize the gradcam to the size of the image
        stacked_gradcam = F.interpolate(stacked_gradcam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[2:], mode=interpolation_mode).squeeze()
        self.storage.clear()
        return stacked_gradcam, importance_tensor
        

    def generate_saliency_map(self, input_tensor, target_class):
        self.eval()
        input_tensor.requires_grad_()
        # Unregister ReLU hooks to get raw gradients
        self._unregister_hooks()
        
        # Forward pass
        output = self(input_tensor)
        self.zero_grad()
        self.storage.clear()
        # Backward pass on a single target class
        target = output[0, target_class]
        target.backward()
        # Get the gradients
        # these will be the gradients of the loss with respect to the input
        saliency_map = input_tensor.grad.data
        # take only the absolute value of the gradients
        saliency_map = torch.abs(saliency_map)
        # Normalize the saliency map between 0 and 1
        saliency_map = self.normalize_tensor(saliency_map, high=1, low=0)

        # squeeze the tensor to remove the channel dimension
        saliency_map = saliency_map.squeeze()
        # Re-register ReLU hooks for guided backpropagation
        self._register_hooks()
        # get the average across the channels
        saliency_map = torch.mean(saliency_map, dim=0)

        return saliency_map

    def forward(self, x):
        return self.base_model(x)
        # return self.resnet50(x)

    def normalize_tensor(self, tensor, high=1, low=-1):
        # as this will divide handle if the case is that all are 0
        if ((tensor.max() - tensor.min()) * (high - low)) == 0:
            return torch.zeros_like(tensor)
        return low + (tensor - tensor.min()) / (tensor.max() - tensor.min()) * (high - low)
    def __del__(self):
        self._unregister_hooks()
    def remove_hooks(self):
        self._unregister_hooks()

