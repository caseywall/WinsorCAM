import io
import numpy as np
import ipywidgets as widgets
from ipywidgets import VBox, Layout, HTML, Image as WImage, IntSlider, IntRangeSlider, ToggleButtons, FloatSlider, Dropdown, HBox
import torch
from matplotlib import pyplot as plt
# get display module from IPython
from IPython.display import display
from utils import *
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def imagenet_controller(dataset, labels, image_width=224, initial_model='densenet121'):
    """Simplified controller for ImageNet dataset with mode selection and model switching

    Note there is an incompatibility between Inception v3 and FullGrad mode within pytorch_grad_cam

    Args:
        dataset: The dataset to use for images
        labels: List of class label strings (e.g., ImageNet class names)
        image_width: Width of the display widget
        initial_model: Initial model to load (default: 'densenet121')
    """

    # Main control widgets
    model_selection_buttons = ToggleButtons(
        options=['resnet50', 'densenet121', 'vgg16', 'inception_v3', 'convnext_tiny', 'efficientnet_b0'],
        value=initial_model,
        description='Model:',
        tooltips=['ResNet-50', 'DenseNet-121', 'VGG-16', 'Inception V3', 'ConvNeXt Tiny', 'EfficientNet-B0'],
        layout=Layout(width='98%', margin='0px')
    )

    mode_buttons = ToggleButtons(
        options=['Winsor-CAM', 'Naive Grad-CAM', 'Final Layer Grad-CAM',
                 'Grad-CAM++', 'Score-CAM', 'FullGrad',
                 'Ablation-CAM', 'Layer-CAM', 'Shapley-CAM'],
        value='Winsor-CAM',
        description='Mode:',
        tooltips=['Winsor-CAM with selected layers', 'Naive Grad-CAM aggregation across all layers',
                  'Final layer Grad-CAM', 'Grad-CAM++ (uses final layer)', 'Score-CAM (uses final layer)',
                  'FullGrad (uses all layers)', 'Ablation-CAM (uses final layer)',
                  'Layer-CAM (uses final layer)', 'Shapley-CAM (uses final layer)'],
        layout=Layout(width='98%', margin='0px')
    )

    # Image selection
    if hasattr(dataset, 'name_map'):
        image_options = list(reversed(list(dataset.name_map.keys())))
        initial_image_key = image_options[0] if image_options else 0
    else:
        image_options = list(range(len(dataset)))
        initial_image_key = 0

    image_selection_dropdown = widgets.Dropdown(
        options=image_options,
        value=initial_image_key,
        description='Image:',
        layout=Layout(width='98%'),
    )

    # Visualization parameters in compact layout
    transparency_slider = FloatSlider(min=0, max=1, step=.01, value=.9, description='Transparency:', readout=True, readout_format='.2f', layout=Layout(width='98%'), continuous_update=False)
    threshold_slider = FloatSlider(min=0.0, max=1.0, step=0.01, value=0.5, description='Threshold:', readout=True, readout_format='.2f', layout=Layout(width='98%'), continuous_update=False)

    winsor_slider = IntSlider(min=0, max=100, step=1, value=80, description='Winsor %:', readout=True, readout_format='d', layout=Layout(width='98%'), continuous_update=False)
    colormap_buttons = ToggleButtons(options=['nipy_spectral', 'Reds', 'viridis', 'plasma', 'inferno', 'magma'], value='nipy_spectral', description='Colormap:', layout=Layout(width='98%', margin='0px'))

    # Range slider will be initialized after we load the model
    range_slider = IntRangeSlider(value=(0, 10), min=0, max=10, step=1, description='Layer Range:', readout=True, readout_format='d', layout=Layout(width='98%'), continuous_update=False)

    operation_buttons = ToggleButtons(options=['max', 'mean'], value='mean', description='Operation:', tooltips=['Maximum value', 'Mean value'], layout=Layout(width='48%', margin='0px'))
    interpolation_buttons = ToggleButtons(options=['bilinear', 'nearest-exact'], value='bilinear', description='Interp:', tooltips=['Bilinear', 'Nearest'], layout=Layout(width='48%', margin='0px'))

    title_text = HTML(value="<h3>Winsor-CAM Interactive Controller</h3>")
    info_text = HTML(value="")
    

    # Initialize with first image
    dataset_key = image_selection_dropdown.value
    input_tensor, sample_label, label_tensor = dataset[dataset_key]
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    unnormalized_image = denormalize(input_tensor.squeeze(0).cpu()).squeeze(0).permute(1, 2, 0).numpy()
    # Helper function to load a model

    

    # State dictionary
    state = {
        'base_image': torch.from_numpy(unnormalized_image).float(),
        'input_tensor': input_tensor.squeeze(0),
        'dataset_key': dataset_key,
        'sample_label': sample_label,
        'label_tensor': label_tensor,
        'predicted_class': None,
        'output': None,
        'transparency': transparency_slider.value,
        'filter_image': None,
        'stacked_gradcam': None,
        'grads': None,
        'importance_tensor': None,
        'operation': operation_buttons.value,
        'gradient_aggregation_method': 'mean',
        'interpolation_mode': interpolation_buttons.value,
        'winsor_percentile': winsor_slider.value,
        'colormap': colormap_buttons.value,
        'threshold': threshold_slider.value,
        'mode': mode_buttons.value,
        'range_min': None,
        'range_max': None,
        'model': None,
        'layer_names': None,
        'model_name': None,
    }



    def load_model_for_ui(model_choice):
        """Load a model and return it with its layer names"""
        model_obj, layer_names_obj, inception_flag = create_winsorgradcam_model(
            model_choice, 
            device=device.type, 
            num_classes=None,
            cpu_only_flag=device
        )
        model_obj.eval()
        model_obj.to(device)
        # if the inception model, adjust input size by modifying the dataset.transforms
        if inception_flag:
            dataset.transform.transforms[0] = transforms.Resize((299, 299))
            # we need to update the input tensor and unnormalized image accordingly
            # I think we can do this using def update_model_selection(change) but it needs to be called without arguments
            input_tensor, sample_label, _ = dataset[dataset_key]
            input_tensor = input_tensor.unsqueeze(0).to(device)
            
            unnormalized_image = denormalize(input_tensor.squeeze(0).cpu()).squeeze(0).permute(1, 2, 0).numpy()
            
            state['base_image'] = torch.from_numpy(unnormalized_image).float()
            state['input_tensor'] = input_tensor.squeeze(0)
        else:
            dataset.transform.transforms[0] = transforms.Resize((224, 224))
            input_tensor, sample_label, _ = dataset[dataset_key]
            input_tensor = input_tensor.unsqueeze(0).to(device)
            
            unnormalized_image = denormalize(input_tensor.squeeze(0).cpu()).squeeze(0).permute(1, 2, 0).numpy()
            
            state['base_image'] = torch.from_numpy(unnormalized_image).float()
            state['input_tensor'] = input_tensor.squeeze(0)
            
                 
            
        return model_obj, layer_names_obj, inception_flag
    # Load initial model
    current_model, current_layer_names, inception = load_model_for_ui(initial_model)

    initial_min = 0
    initial_max = len(current_layer_names)

    # Now update range slider with actual layer count
    initial_min = 0
    initial_max = len(current_layer_names)
    range_slider.min = initial_min
    range_slider.max = initial_max
    range_slider.value = (initial_min, initial_max)  # Must be tuple
    state['range_min'] = range_slider.value[0]
    state['range_max'] = range_slider.value[1]

    
    state['model'] = current_model
    state['layer_names'] = current_layer_names
    state['model_name'] = initial_model

    # Get prediction
    state['model'].eval()
    with torch.no_grad():
        output = state['model'](state['input_tensor'].unsqueeze(0).to(device))
    predicted_class = torch.argmax(output).item()
    state['output'] = output
    state['predicted_class'] = predicted_class

    # Get GradCAMs
    state['model'].storage.clear()
    
    stacked_gradcam, gradcams, importance_tensor = state['model'].get_gradcams_and_importance(
        state['input_tensor'].to(device).unsqueeze(0),
        predicted_class, 
        state['layer_names'],
        state['gradient_aggregation_method'],
        state['operation'], 
        stack_relu=True,
        interpolation_mode=state['interpolation_mode']
    )
    
    state['importance_tensor'] = importance_tensor.clone()
    state['stacked_gradcam'] = stacked_gradcam.clone()
    
    # Update range slider max based on model layers
    range_slider.max = len(state['layer_names'])
    range_slider.value = (0, len(state['layer_names']))  # Must be tuple
    state['range_min'] = 0
    state['range_max'] = len(state['layer_names'])
    
    # Initialize filter image based on mode
    filter_image, grads = state['model'].winsorize_stacked_gradcam(
        state['input_tensor'].unsqueeze(0),
        state['stacked_gradcam'][state['range_min']:state['range_max']],
        state['importance_tensor'][state['range_min']:state['range_max']],
        interpolation_mode=state['interpolation_mode'],
        winsor_percentile=state['winsor_percentile']
    )

    state['filter_image'] = filter_image.clone()
    state['grads'] = grads

    def create_combined_visualization():
        """Create figure with 5 panels"""
        fig, axes = plt.subplots(1, 5, figsize=(25, 5), dpi=80)
        fig.subplots_adjust(wspace=0.1, hspace=0.05)

        # Original Image
        axes[0].imshow(state['base_image'].cpu().numpy())
        axes[0].set_title('Original Image', fontsize=12, pad=10)
        axes[0].axis('off')

        # Overlay with heatmap
        axes[1].imshow(state['base_image'].cpu().numpy())
        filter_image_cpu = state['filter_image'].cpu() if state['filter_image'].is_cuda else state['filter_image']
        colored_filter_image = create_colored_heatmap(
            filter_image_cpu.squeeze(),
            op_multiplier=state['transparency'],
            size=state['base_image'].shape[0:2],
            colormap=state['colormap'],
            interpolation_mode=state['interpolation_mode']
        )
        axes[1].imshow(colored_filter_image.squeeze())
        axes[1].set_title(f'{state["mode"]} Overlay', fontsize=12, pad=10)
        axes[1].axis('off')

        # Otsu binarized masked image
        stacked_gradcam_np = filter_image_cpu.squeeze().cpu().numpy()
        masked_image_otsu = generate_masked_image(state['base_image'].cpu().numpy(), stacked_gradcam_np)
        axes[2].imshow(masked_image_otsu.squeeze())
        axes[2].set_title('Masked Image (Otsu)', fontsize=12, pad=10)
        axes[2].axis('off')

        # Threshold binarized masked image
        stacked_gradcam_np_norm = (stacked_gradcam_np - stacked_gradcam_np.min()) / (stacked_gradcam_np.max() - stacked_gradcam_np.min() + 1e-6)
        masked_image_threshold = generate_masked_image_set(state['base_image'].cpu().numpy(), stacked_gradcam_np_norm, threshold=state['threshold'])
        axes[3].imshow(masked_image_threshold.squeeze())
        axes[3].set_title(f'Masked Image (T={state["threshold"]:.2f})', fontsize=12, pad=10)
        axes[3].axis('off')

        # Gradient plot
        grads_cpu = state['grads'].cpu() if isinstance(state['grads'], torch.Tensor) and state['grads'].is_cuda else state['grads']
        grads_np = grads_cpu.cpu().detach().numpy() if isinstance(grads_cpu, torch.Tensor) else np.array(grads_cpu)

        # Normalize to [0, 1] range
        g_min, g_max = grads_np.min(), grads_np.max()
        normalized_grads = np.ones_like(grads_np) if g_max <= g_min else (grads_np - g_min) / (g_max - g_min)
        normalized_grads = np.clip(normalized_grads, 0.0, 1.0)

        axes[4].plot(normalized_grads, linewidth=2, color='#2E86AB')
        axes[4].set_ylim(-0.05, 1.1)
        axes[4].set_xlabel('Layer Index', fontsize=10)
        axes[4].set_ylabel('Normalized Importance', fontsize=10)

        title = f'Layer Importance (Winsor {state["winsor_percentile"]}%)' if state['mode'] == 'Winsor-CAM' else f'Layer Importance ({state["mode"]})'
        axes[4].set_title(title, fontsize=12, pad=10)
        axes[4].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        axes[4].spines['top'].set_visible(False)
        axes[4].spines['right'].set_visible(False)

        # Set x-axis ticks
        num_layers = len(normalized_grads)
        if num_layers > 0:
            axes[4].set_xlim(0, num_layers - 1)
            num_ticks = min(num_layers, 8)
            x_ticks = np.linspace(0, num_layers - 1, num_ticks)
            x_labels = [str(int(state['range_min'] + tick)) for tick in x_ticks]
            axes[4].set_xticks(x_ticks)
            axes[4].set_xticklabels(x_labels, fontsize=9)
            axes[4].tick_params(axis='y', labelsize=9)

        buf = io.BytesIO()
        plt.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0.2, dpi=60)
        plt.close(fig)
        return buf.getvalue()

    def update_visualization():
        """Update the visualization and info text"""
        image_widget.value = create_combined_visualization()
        update_info_text()

    def update_simple_param(change, param_name):
        """Generic update function for simple parameters"""
        state[param_name] = change['new']
        update_visualization()

    def update_winsor_percentile(change):
        state['winsor_percentile'] = change['new']
        new_filter_image, new_grads = state['model'].winsorize_stacked_gradcam(
            state['input_tensor'].unsqueeze(0),
            state['stacked_gradcam'][state['range_min']:state['range_max']],
            state['importance_tensor'][state['range_min']:state['range_max']],
            interpolation_mode=state['interpolation_mode'],
            winsor_percentile=state['winsor_percentile']
        )
        state['filter_image'] = new_filter_image.clone()
        state['grads'] = new_grads
        update_visualization()

    def update_range(change):
        state['range_min'], state['range_max'] = change['new']

        new_filter_image, new_grads = state['model'].winsorize_stacked_gradcam(
            state['input_tensor'].unsqueeze(0),
            state['stacked_gradcam'][state['range_min']:state['range_max']],
            state['importance_tensor'][state['range_min']:state['range_max']],
            interpolation_mode=state['interpolation_mode'],
            winsor_percentile=state['winsor_percentile']
        )
        state['filter_image'] = new_filter_image.clone()
        state['grads'] = new_grads
        update_visualization()

    def update_operation(change=None):
        if change is not None:
            state['operation'] = change['new']

        new_stacked_gradcam, _, new_importance_tensor = state['model'].get_gradcams_and_importance(
            state['input_tensor'].to(device).unsqueeze(0),
            state['predicted_class'],
            state['layer_names'],
            state['gradient_aggregation_method'],
            state['operation'],
            stack_relu=True,
            interpolation_mode=state['interpolation_mode']
        )

        state['stacked_gradcam'] = new_stacked_gradcam.clone()
        state['importance_tensor'] = new_importance_tensor.clone()

        new_filter_image, new_grads = state['model'].winsorize_stacked_gradcam(
            state['input_tensor'].unsqueeze(0),
            state['stacked_gradcam'][state['range_min']:state['range_max']],
            state['importance_tensor'][state['range_min']:state['range_max']],
            interpolation_mode=state['interpolation_mode'],
            winsor_percentile=state['winsor_percentile']
        )
        state['filter_image'] = new_filter_image.clone()
        state['grads'] = new_grads

        if change is not None:
            update_visualization()

    def update_interpolation(change):
        state['interpolation_mode'] = change['new']
        state['model'].storage.clear()
        update_operation(None)  # Recalculate based on current mode
        update_visualization()

    def update_mode(change=None):
        if change is None:
            mode = state['mode']
        else:
            mode = change['new']
        # Show loading message for computationally intensive modes
        if mode in ['Grad-CAM++', 'Score-CAM', 'FullGrad', 'Ablation-CAM', 'Layer-CAM', 'Shapley-CAM']:
            info_text.value = f"<p style='font-size: 16px; color: #FF6B35;'><b>⏳ Computing {mode}...</b></p>"
            
            # Create loading placeholder
            loading_fig = plt.figure(figsize=(25, 5), dpi=80)
            loading_ax = loading_fig.add_subplot(111)
            loading_ax.text(0.5, 0.5, f'Computing {mode.upper()}...', 
                           ha='center', va='center', fontsize=24, color='#FF6B35', weight='bold')
            loading_ax.set_xlim(0, 1)
            loading_ax.set_ylim(0, 1)
            loading_ax.axis('off')
            loading_buf = io.BytesIO()
            plt.savefig(loading_buf, format='PNG', bbox_inches='tight', dpi=60)
            plt.close(loading_fig)
            image_widget.value = loading_buf.getvalue()
        
        state['mode'] = mode
        
        match state['mode']:
            case 'Winsor-CAM':
                # Enable all controls for Winsor-CAM
                model_selection_buttons.disabled = False
                image_selection_dropdown.disabled = False
                interpolation_buttons.disabled = False
                range_slider.disabled = False
                winsor_slider.disabled = False
                operation_buttons.disabled = False
                
                # Restore range from slider
                state['range_min'] = range_slider.value[0]
                state['range_max'] = range_slider.value[1]

                state['filter_image'], state['grads'] = state['model'].winsorize_stacked_gradcam(
                    state['input_tensor'].unsqueeze(0),
                    state['stacked_gradcam'][state['range_min']:state['range_max']],
                    state['importance_tensor'][state['range_min']:state['range_max']],
                    interpolation_mode=state['interpolation_mode'],
                    winsor_percentile=state['winsor_percentile']
                )

            case 'Naive Grad-CAM':
                # Enable all controls for Naive Grad-CAM
                model_selection_buttons.disabled = False
                image_selection_dropdown.disabled = False
                interpolation_buttons.disabled = False
                range_slider.disabled = True
                winsor_slider.disabled = True
                operation_buttons.disabled = True
                
                naive_gradcam = torch.mean(state['stacked_gradcam'], dim=0, keepdim=True)
                state['filter_image'] = naive_gradcam.clone()
                state['grads'] = torch.ones_like(state['stacked_gradcam'][:,0,0])
                state['range_min'] = 0
                state['range_max'] = state['stacked_gradcam'].shape[0]
                
            case 'Final Layer Grad-CAM':
                # Enable all controls for Final Layer Grad-CAM
                model_selection_buttons.disabled = False
                image_selection_dropdown.disabled = False
                interpolation_buttons.disabled = False
                range_slider.disabled = True
                winsor_slider.disabled = True
                operation_buttons.disabled = True
                
                final_layer_gradcam = state['stacked_gradcam'][-1].unsqueeze(0)
                state['filter_image'] = final_layer_gradcam.clone()
                state['grads'] = torch.zeros_like(state['stacked_gradcam'][:,0,0])
                state['grads'][-1] = 1.0
                
            case 'Grad-CAM++' | 'Score-CAM' | 'XGrad-CAM' | 'FullGrad' | 'Ablation-CAM' | 'Layer-CAM' | 'Shapley-CAM':
                # Disable all controls except Transparency, Mode, Binary Threshold, Colormap, and Model Selection
                range_slider.disabled = True
                winsor_slider.disabled = True
                operation_buttons.disabled = True
                interpolation_buttons.disabled = True
                
                # Get the CAM using the comparative method
                match state['mode']:
                    case 'Grad-CAM++':
                        cam = state['model'].get_cam_comparative(state['input_tensor'], 'gradcampp')
                    case 'Score-CAM':
                        cam = state['model'].get_cam_comparative(state['input_tensor'], 'scorecam')
                    case 'XGrad-CAM':
                        cam = state['model'].get_cam_comparative(state['input_tensor'], 'xgradcam')
                    case 'FullGrad':
                        cam = state['model'].get_cam_comparative(state['input_tensor'], 'fullgrad')
                    case 'Ablation-CAM':
                        cam = state['model'].get_cam_comparative(state['input_tensor'], 'ablation')
                    case 'Layer-CAM':
                        cam = state['model'].get_cam_comparative(state['input_tensor'], 'layercam')
                    case 'Shapley-CAM':
                        cam = state['model'].get_cam_comparative(state['input_tensor'], 'shapleycam')
                
                # Convert to tensor efficiently and ensure proper dtype
                if isinstance(cam, np.ndarray):
                    cam = torch.from_numpy(cam).float().to(state['input_tensor'].device)
                else:
                    cam = torch.tensor(cam, dtype=torch.float32, device=state['input_tensor'].device)
                
                # Ensure cam is 2D (squeeze out extra dimensions if needed)
                while cam.ndim > 2:
                    cam = cam.squeeze(0)
                
                # Add batch dimension for consistency
                state['filter_image'] = cam.unsqueeze(0) if cam.ndim == 2 else cam
                
                # Set grads based on method type
                if state['mode'] == 'FullGrad':
                    # FullGrad uses all layers, so show uniform importance
                    state['grads'] = torch.ones_like(state['stacked_gradcam'][:,0,0])
                else:
                    # Other methods use only the final layer
                    state['grads'] = torch.zeros_like(state['stacked_gradcam'][:,0,0])
                    state['grads'][-1] = 1.0
                
                state['range_min'] = 0
                state['range_max'] = state['stacked_gradcam'].shape[0]
        
        if change is not None:
            image_widget.value = create_combined_visualization()
            update_info_text()

    def update_image_selection(change=None):
        """Update everything when a new image is selected"""
        if change is None:
            # If no change provided, use the current dropdown value
            dataset_key = image_selection_dropdown.value
        else:
            dataset_key = change['new']
        state['dataset_key'] = dataset_key
        input_tensor, sample_label, _ = dataset[dataset_key]
        input_tensor = input_tensor.unsqueeze(0).to(device)
        unnormalized_image = denormalize(input_tensor.squeeze(0).cpu()).squeeze(0).permute(1, 2, 0).numpy()
        
        state['base_image'] = torch.from_numpy(unnormalized_image).float()
        state['input_tensor'] = input_tensor.squeeze(0)
        state['sample_label'] = sample_label
        
        state['model'].eval()
        with torch.no_grad():
            output = state['model'](state['input_tensor'].unsqueeze(0).to(device))
        predicted_class = torch.argmax(output).item()
        state['output'] = output
        state['predicted_class'] = predicted_class
        
        state['model'].storage.clear()

        stacked_gradcam, gradcams, importance_tensor = state['model'].get_gradcams_and_importance(
            state['input_tensor'].to(device).unsqueeze(0),
            predicted_class, 
            state['layer_names'],
            state['gradient_aggregation_method'],
            state['operation'], 
            stack_relu=True,
            interpolation_mode=state['interpolation_mode']
        )
        
        state['importance_tensor'] = importance_tensor.clone()
        state['stacked_gradcam'] = stacked_gradcam.clone()
        # Recalculate based on current mode
        update_mode(None)

        
        image_widget.value = create_combined_visualization()
        update_info_text()
            
    def update_model_selection(change):
        """Update model when selection changes"""
        new_model_name = change['new']
        if new_model_name == state['model_name']:
            return  # No change needed
        
        # Handle Inception v3 + FullGrad incompatibility
        if new_model_name == 'inception_v3':
            # Remove FullGrad from options if it's currently available
            current_options = list(mode_buttons.options)
            if 'FullGrad' in current_options:
                current_options.remove('FullGrad')
                mode_buttons.options = current_options
                # If FullGrad was selected, switch to Winsor-CAM
                if state['mode'] == 'FullGrad':
                    mode_buttons.value = 'Winsor-CAM'
                    update_mode({'new': 'Winsor-CAM'})
                    return  # Exit early since update_mode will handle the rest
        else:
            # For non-Inception models, ensure FullGrad is available
            current_options = list(mode_buttons.options)
            if 'FullGrad' not in current_options:
                # Add FullGrad back to options
                all_options = ['Winsor-CAM', 'Naive Grad-CAM', 'Final Layer Grad-CAM', 
                               'Grad-CAM++', 'Score-CAM', 'FullGrad', 
                               'Ablation-CAM', 'Layer-CAM', 'Shapley-CAM']
                mode_buttons.options = all_options
        
        # Show loading message immediately
        info_text.value = f"<p style='font-size: 16px; color: #FF6B35;'><b>⏳ Loading {new_model_name}...</b></p>"
        
        # Create a simple loading placeholder image
        loading_fig = plt.figure(figsize=(25, 5), dpi=80)
        loading_ax = loading_fig.add_subplot(111)
        loading_ax.text(0.5, 0.5, f'Loading {new_model_name.upper()}...', 
                       ha='center', va='center', fontsize=24, color='#FF6B35', weight='bold')
        loading_ax.set_xlim(0, 1)
        loading_ax.set_ylim(0, 1)
        loading_ax.axis('off')
        loading_buf = io.BytesIO()
        plt.savefig(loading_buf, format='PNG', bbox_inches='tight', dpi=60)
        plt.close(loading_fig)
        image_widget.value = loading_buf.getvalue()
        
        
        new_model, new_layer_names, inception = load_model_for_ui(new_model_name)
        
        # Update state
        state['model'] = new_model
        state['layer_names'] = new_layer_names
        state['model_name'] = new_model_name
        
        # Update range slider
        range_slider.max = len(new_layer_names)
        range_slider.value = (0, len(new_layer_names))  # Must be tuple
        state['range_min'] = 0
        state['range_max'] = len(new_layer_names)
        
        # Get prediction with new model
        state['model'].eval()
        with torch.no_grad():
            output = state['model'](state['input_tensor'].unsqueeze(0).to(device))
        predicted_class = torch.argmax(output).item()
        state['output'] = output
        state['predicted_class'] = predicted_class
        
        # Generate new GradCAMs
        state['model'].storage.clear()
        
        stacked_gradcam, gradcams, importance_tensor = state['model'].get_gradcams_and_importance(
            state['input_tensor'].to(device).unsqueeze(0),
            predicted_class, 
            state['layer_names'],
            state['gradient_aggregation_method'],
            state['operation'], 
            stack_relu=True,
            interpolation_mode=state['interpolation_mode']
        )
        
        state['importance_tensor'] = importance_tensor.clone()
        state['stacked_gradcam'] = stacked_gradcam.clone()
        
        # Recalculate based on current mode
        update_mode(None)

        # state['filter_image'] = filter_image.clone()
        # state['grads'] = grads
        
        image_widget.value = create_combined_visualization()
        update_info_text()

    def update_info_text():
        confidence_score = torch.softmax(state['output'], dim=1)[0][state['predicted_class']].item()
        deterministic_status = "Disabled (VGG16)" if state['model_name'] == 'vgg16' else "Enabled"
        info_text.value = f"""
        <p><b>Current Settings:</b></p>
        <ul style="column-count: 2; column-gap: 20px;">
            <li>Model: {state['model_name']}</li>
            <li>Deterministic Mode: {deterministic_status}</li>
            <li>Selected Image: {state['dataset_key']}</li>
            <li>True Label: {labels[state['sample_label']]}</li>
            <li>Predicted Class: {labels[state['predicted_class']]} (confidence: {confidence_score:.4f})</li>
            <li>Mode: {state['mode']}</li>
            <li>Winsor Percentile: {state['winsor_percentile']}%</li>
            <li>Layer Range: {state['range_min']} - {state['range_max']}</li>
            <li>Binary Threshold: {state['threshold']:.2f}</li>
        </ul>
        """

    # Create initial visualization
    img_bytes = create_combined_visualization()
    image_widget = WImage(
        value=img_bytes,
        format='png',
        width=image_width*5,
    )

    # Update info text initially
    update_info_text()

    # Attach observers
    model_selection_buttons.observe(update_model_selection, names='value')
    transparency_slider.observe(lambda c: update_simple_param(c, 'transparency'), names='value')
    winsor_slider.observe(update_winsor_percentile, names='value')
    threshold_slider.observe(lambda c: update_simple_param(c, 'threshold'), names='value')
    range_slider.observe(update_range, names='value')
    operation_buttons.observe(update_operation, names='value')
    interpolation_buttons.observe(update_interpolation, names='value')
    colormap_buttons.observe(lambda c: update_simple_param(c, 'colormap'), names='value')
    mode_buttons.observe(update_mode, names='value')
    image_selection_dropdown.observe(update_image_selection, names='value')

    # Create UI
    centered_ui = VBox([
        title_text,
        info_text,
        image_widget,
        winsor_slider,
        image_selection_dropdown,
        threshold_slider,
        transparency_slider,
        range_slider,
        model_selection_buttons,
        mode_buttons,
        operation_buttons,
        interpolation_buttons,
        colormap_buttons,
    ])

    centered_ui.layout = Layout(
        display='flex',
        flex_flow='column',
        align_items='stretch',
        width='100%'
    )

    display(centered_ui)

def pascal_voc_controller(dataset, labels, image_width=224, initial_model='densenet121'):
    """Simplified controller for Pascal VOC dataset with mode selection and model switching

    Note there is an incompatibility between Inception v3 and FullGrad mode within pytorch_grad_cam

    Args:
        dataset: The dataset to use for images
        labels: List of class label strings (e.g., Pascal VOC class names)
        image_width: Width of the display widget
        initial_model: Initial model to load (default: 'densenet121')
    """

    # Main control widgets
    model_selection_buttons = ToggleButtons(
        options=['resnet50', 'densenet121', 'vgg16', 'inception_v3', 'convnext_tiny', 'efficientnet_b0'],
        value=initial_model,
        description='Model:',
        tooltips=['ResNet-50', 'DenseNet-121', 'VGG-16', 'Inception V3', 'ConvNeXt Tiny', 'EfficientNet-B0'],
        layout=Layout(width='98%', margin='0px')
    )

    mode_buttons = ToggleButtons(
        options=['Winsor-CAM', 'Naive Grad-CAM', 'Final Layer Grad-CAM',
                 'Grad-CAM++', 'Score-CAM', 'FullGrad',
                 'Ablation-CAM', 'Layer-CAM', 'Shapley-CAM'],
        value='Winsor-CAM',
        description='Mode:',
        tooltips=['Winsor-CAM with selected layers', 'Naive Grad-CAM aggregation across all layers',
                  'Final layer Grad-CAM', 'Grad-CAM++ (uses final layer)', 'Score-CAM (uses final layer)',
                  'FullGrad (uses all layers)', 'Ablation-CAM (uses final layer)',
                  'Layer-CAM (uses final layer)', 'Shapley-CAM (uses final layer)'],
        layout=Layout(width='98%', margin='0px')
    )

    # Image selection
    image_options = dataset.get_available_images()
    initial_image_key = image_options[0] if image_options else 0
    image_selection_dropdown = widgets.Dropdown(
        options=image_options,
        value=initial_image_key,
        description='Image:',
        layout=Layout(width='98%'),
    )

    # Visualization parameters in compact layout
    transparency_slider = FloatSlider(min=0, max=1, step=.01, value=.9, description='Transparency:', readout=True, readout_format='.2f', layout=Layout(width='98%'), continuous_update=False)
    threshold_slider = FloatSlider(min=0.0, max=1.0, step=0.01, value=0.5, description='Threshold:', readout=True, readout_format='.2f', layout=Layout(width='98%'), continuous_update=False)

    winsor_slider = IntSlider(min=0, max=100, step=1, value=80, description='Winsor %:', readout=True, readout_format='d', layout=Layout(width='98%'), continuous_update=False)
    colormap_buttons = ToggleButtons(options=['nipy_spectral', 'Reds', 'viridis', 'plasma', 'inferno', 'magma'], value='nipy_spectral', description='Colormap:', layout=Layout(width='98%', margin='0px'))

    # Range slider will be initialized after we load the model
    range_slider = IntRangeSlider(value=(0, 10), min=0, max=10, step=1, description='Layer Range:', readout=True, readout_format='d', layout=Layout(width='98%'), continuous_update=False)

    operation_buttons = ToggleButtons(options=['max', 'mean'], value='mean', description='Operation:', tooltips=['Maximum value', 'Mean value'], layout=Layout(width='48%', margin='0px'))
    interpolation_buttons = ToggleButtons(options=['bilinear', 'nearest-exact'], value='bilinear', description='Interp:', tooltips=['Bilinear', 'Nearest'], layout=Layout(width='48%', margin='0px'))

    title_text = HTML(value="<h3>Winsor-CAM Interactive Controller</h3>")
    info_text = HTML(value="")
    

    # Initialize with first image
    dataset_key = image_selection_dropdown.value
    input_tensor, label_tensor, one_hot, class_name = dataset[dataset_key]
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    unnormalized_image = denormalize(input_tensor.squeeze(0).cpu()).squeeze(0).permute(1, 2, 0).numpy()
    # Helper function to load a model

    

    # State dictionary
    state = {
        'base_image': torch.from_numpy(unnormalized_image).float(),
        'input_tensor': input_tensor.squeeze(0),
        'dataset_key': dataset_key,
        'sample_label': class_name,
        'label_tensor': label_tensor,
        'predicted_class': None,
        'output': None,
        'transparency': transparency_slider.value,
        'filter_image': None,
        'stacked_gradcam': None,
        'grads': None,
        'importance_tensor': None,
        'operation': operation_buttons.value,
        'gradient_aggregation_method': 'mean',
        'interpolation_mode': interpolation_buttons.value,
        'winsor_percentile': winsor_slider.value,
        'colormap': colormap_buttons.value,
        'threshold': threshold_slider.value,
        'mode': mode_buttons.value,
        'range_min': None,
        'range_max': None,
        'model': None,
        'layer_names': None,
        'model_name': None,
    }


    def load_model_for_ui(model_choice):
        """Load a model and return it with its layer names"""
        model_obj, layer_names_obj, inception_flag = create_winsorgradcam_model(
            model_choice, 
            device=device.type, 
            num_classes=20,
            cpu_only_flag=device
        )
        model_obj.eval()
        model_obj.to(device)
        return model_obj, layer_names_obj, inception_flag
    # Load initial model
    current_model, current_layer_names, inception = load_model_for_ui(initial_model)

    # Update dataset transforms for future image selections and re-fetch current image
    if inception:
        dataset.transforms_image.transforms[0] = transforms.Resize((299, 299))
        dataset.transforms_label.transforms[0] = transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.NEAREST)
        # Re-fetch the currently selected image from the dataset so transforms are applied exactly once
        input_tensor_new, label_tensor_new, one_hot_new, class_name_new = dataset[state['dataset_key']]
        input_tensor_new = input_tensor_new.unsqueeze(0).to(device)
        unnormalized_image = denormalize(input_tensor_new.squeeze(0).cpu()).squeeze(0).permute(1, 2, 0).numpy()
        state['base_image'] = torch.from_numpy(unnormalized_image).float()
        state['input_tensor'] = input_tensor_new.squeeze(0)
        state['label_tensor'] = label_tensor_new
        state['sample_label'] = class_name_new
    else:
        dataset.transforms_image.transforms[0] = transforms.Resize((224, 224))
        dataset.transforms_label.transforms[0] = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)
        # Re-fetch the currently selected image from the dataset so transforms are applied exactly once
        input_tensor_new, label_tensor_new, one_hot_new, class_name_new = dataset[state['dataset_key']]
        input_tensor_new = input_tensor_new.unsqueeze(0).to(device)
        unnormalized_image = denormalize(input_tensor_new.squeeze(0).cpu()).squeeze(0).permute(1, 2, 0).numpy()
        state['base_image'] = torch.from_numpy(unnormalized_image).float()
        state['input_tensor'] = input_tensor_new.squeeze(0)
        state['label_tensor'] = label_tensor_new
        state['sample_label'] = class_name_new
            

    initial_min = 0
    initial_max = len(current_layer_names)

    # Now update range slider with actual layer count
    initial_min = 0
    initial_max = len(current_layer_names)
    range_slider.min = initial_min
    range_slider.max = initial_max
    range_slider.value = (initial_min, initial_max)  # Must be tuple
    state['range_min'] = range_slider.value[0]
    state['range_max'] = range_slider.value[1]

    
    state['model'] = current_model
    state['layer_names'] = current_layer_names
    state['model_name'] = initial_model

    # Get prediction
    state['model'].eval()
    with torch.no_grad():
        output = state['model'](state['input_tensor'].unsqueeze(0).to(device))
    predicted_class = torch.argmax(output).item()
    state['output'] = output
    state['predicted_class'] = predicted_class

    # Get GradCAMs
    state['model'].storage.clear()

    
    stacked_gradcam, gradcams, importance_tensor = state['model'].get_gradcams_and_importance(
        state['input_tensor'].to(device).unsqueeze(0),
        predicted_class, 
        state['layer_names'],
        state['gradient_aggregation_method'],
        state['operation'], 
        stack_relu=True,
        interpolation_mode=state['interpolation_mode']
    )
    
    state['importance_tensor'] = importance_tensor.clone()
    state['stacked_gradcam'] = stacked_gradcam.clone()
    
    # Update range slider max based on model layers
    range_slider.max = len(state['layer_names'])
    range_slider.value = (0, len(state['layer_names']))  # Must be tuple
    state['range_min'] = 0
    state['range_max'] = len(state['layer_names'])
    # Initialize filter image based on mode
    filter_image, grads = state['model'].winsorize_stacked_gradcam(
        state['input_tensor'].unsqueeze(0),
        state['stacked_gradcam'][state['range_min']:state['range_max']],
        state['importance_tensor'][state['range_min']:state['range_max']],
        interpolation_mode=state['interpolation_mode'],
        winsor_percentile=state['winsor_percentile']
    )
    state['filter_image'] = filter_image.clone()
    state['grads'] = grads

    def create_combined_visualization():
        """Create figure with 5 panels"""
        fig = plt.figure(figsize=(25, 5), dpi=80, constrained_layout=True)
        gs = fig.add_gridspec(1, 5, wspace=0.05, hspace=0.05)
        
        # Original Image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(state['base_image'].cpu().numpy())
        ax1.set_title('Original Image', fontsize=12, pad=10)
        ax1.axis('off')
        
        # Overlay with heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(state['base_image'].cpu().numpy())
        
        filter_image_cpu = state['filter_image'].cpu() if state['filter_image'].is_cuda else state['filter_image']
        
        colored_filter_image = create_colored_heatmap(
            filter_image_cpu.squeeze(),
            op_multiplier=state['transparency'],
            size=state['base_image'].shape[:2],
            colormap=state['colormap'],
            interpolation_mode=state['interpolation_mode']
        )
        ax2.imshow(colored_filter_image.squeeze())
        ax2.set_title(f'{state["mode"]} Overlay', fontsize=12, pad=10)
        ax2.axis('off')
        
        # Convert label tensor to binary mask for centroid calculation
        # label_tensor is (H, W, 3), convert to binary (H, W) where any non-black pixel is 1
        label_np = state['label_tensor'].cpu().numpy()
        gt_binary_mask = ((label_np[:,:,0] > 0) | (label_np[:,:,1] > 0) | (label_np[:,:,2] > 0)).astype(np.float32)
        
        # Create binary masks for IoU visualization
        stacked_gradcam_np = filter_image_cpu.squeeze().cpu().numpy()
        
        # Otsu binarized mask
        binary_mask_otsu = make_binary_mask(stacked_gradcam_np)
        
        # IoU visualization for Otsu
        ax3 = fig.add_subplot(gs[0, 2])

        iou_vis_otsu = visualize_mask_iou(binary_mask_otsu, gt_binary_mask)

        ax3.imshow(iou_vis_otsu)
        
        # Calculate and visualize centroids for Otsu
        centroid_pred = find_heatmap_centroid(filter_image_cpu.squeeze())
        centroid_gt = find_heatmap_centroid(gt_binary_mask)
        ax3.scatter(*centroid_pred, color='yellow', marker='o', edgecolors='black', s=100, zorder=3)
        ax3.scatter(*centroid_gt, color='magenta', marker='o', edgecolors='black', s=100, zorder=3)
        ax3.plot([centroid_pred[0], centroid_gt[0]], [centroid_pred[1], centroid_gt[1]], color='orange', linewidth=2, zorder=2)
        dist_otsu = np.linalg.norm(np.array(centroid_pred) - np.array(centroid_gt))
        mid_x = (centroid_pred[0] + centroid_gt[0]) / 2
        mid_y = (centroid_pred[1] + centroid_gt[1]) / 2
        ax3.text(mid_x, mid_y - 25, f"{dist_otsu:.2f}", color='orange', fontsize=12, ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        
        # Calculate IoU for Otsu
        iou_otsu = calculate_iou(filter_image_cpu.squeeze(), state['label_tensor'], state['input_tensor'].unsqueeze(0), state['interpolation_mode'])
        ax3.set_title(f'Otsu IoU Vis (IoU={iou_otsu:.3f}, Dist={dist_otsu:.1f}px)', fontsize=12, pad=10)
        ax3.axis('off')
        
        # Threshold binarized mask
        stacked_gradcam_np_norm = (stacked_gradcam_np - stacked_gradcam_np.min()) / (stacked_gradcam_np.max() - stacked_gradcam_np.min() + 1e-6)
        binary_mask_threshold = (stacked_gradcam_np_norm > state['threshold']).astype(np.uint8)
        
        # IoU visualization for threshold
        ax4 = fig.add_subplot(gs[0, 3])
        iou_vis_threshold = visualize_mask_iou(binary_mask_threshold, gt_binary_mask)
        ax4.imshow(iou_vis_threshold)
        
        # Calculate and visualize centroids for threshold
        ax4.scatter(*centroid_pred, color='yellow', marker='o', edgecolors='black', s=100, zorder=3)
        ax4.scatter(*centroid_gt, color='magenta', marker='o', edgecolors='black', s=100, zorder=3)
        ax4.plot([centroid_pred[0], centroid_gt[0]], [centroid_pred[1], centroid_gt[1]], color='orange', linewidth=2, zorder=2)
        dist_thresh = dist_otsu  # Same centroid as Otsu
        ax4.text(mid_x, mid_y - 25, f"{dist_thresh:.2f}", color='orange', fontsize=12, ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        
        # Calculate IoU for threshold - use pre-binarized mask directly
        binary_mask_threshold_tensor = torch.from_numpy(binary_mask_threshold).to(torch.uint8)
        gt_binary_mask_tensor = torch.from_numpy(gt_binary_mask).to(torch.uint8)
        intersection_thresh = torch.logical_and(binary_mask_threshold_tensor, gt_binary_mask_tensor)
        union_thresh = torch.logical_or(binary_mask_threshold_tensor, gt_binary_mask_tensor)
        iou_thresh = (intersection_thresh.sum() / union_thresh.sum()).item() if union_thresh.sum() > 0 else 0.0
        ax4.set_title(f'Threshold T={state["threshold"]:.2f} IoU Vis (IoU={iou_thresh:.3f}, Dist={dist_thresh:.1f}px)', fontsize=12, pad=10)
        ax4.axis('off')
        
        # Gradient plot
        ax5 = fig.add_subplot(gs[0, 4])
        grads_cpu = state['grads'].cpu() if isinstance(state['grads'], torch.Tensor) and state['grads'].is_cuda else state['grads']
        
        # Robust normalization
        if isinstance(grads_cpu, torch.Tensor):
            grads_np = grads_cpu.cpu().detach().numpy()
        else:
            grads_np = np.array(grads_cpu)
        
        # Normalize to [0, 1] range with robust handling
        g_min = grads_np.min()
        g_max = grads_np.max()
        
        if g_max > g_min:
            normalized_grads = (grads_np - g_min) / (g_max - g_min)
        else:
            # All values are the same - treat as maximum importance
            normalized_grads = np.ones_like(grads_np)
        
        # Clip to ensure values are in [0, 1]
        normalized_grads = np.clip(normalized_grads, 0.0, 1.0)
        
        ax5.plot(normalized_grads, linewidth=2, color='#2E86AB')
        ax5.set_ylim(-0.05, 1.1)
        ax5.set_xlabel('Layer Index', fontsize=10)
        ax5.set_ylabel('Normalized Importance', fontsize=10)
        
        # Update title based on mode
        if state['mode'] == 'Winsor-CAM':
            title = f'Layer Importance (Winsor {state["winsor_percentile"]}%)'
        else:
            title = f'Layer Importance ({state["mode"]})'
        ax5.set_title(title, fontsize=12, pad=10)
        
        ax5.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        
        # Set x-axis for gradient plot
        num_layers = len(normalized_grads)
        if num_layers > 0:
            ax5.set_xlim(0, num_layers - 1)
            num_ticks = min(num_layers, 8)
            x_ticks = np.linspace(0, num_layers - 1, num_ticks)
            x_labels = [str(int(state['range_min'] + tick)) for tick in x_ticks]
            ax5.set_xticks(x_ticks)
            ax5.set_xticklabels(x_labels, fontsize=9)
            ax5.tick_params(axis='y', labelsize=9)
        
        buf = io.BytesIO()

        plt.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0.2, dpi=60)
        plt.close(fig)
        
        return buf.getvalue()

    def update_visualization():
        """Update the visualization and info text"""
        image_widget.value = create_combined_visualization()
        update_info_text()

    def update_simple_param(change, param_name):
        """Generic update function for simple parameters"""
        state[param_name] = change['new']
        update_visualization()

    def update_winsor_percentile(change):
        state['winsor_percentile'] = change['new']
        new_filter_image, new_grads = state['model'].winsorize_stacked_gradcam(
            state['input_tensor'].unsqueeze(0),
            state['stacked_gradcam'][state['range_min']:state['range_max']],
            state['importance_tensor'][state['range_min']:state['range_max']],
            interpolation_mode=state['interpolation_mode'],
            winsor_percentile=state['winsor_percentile']
        )
        state['filter_image'] = new_filter_image.clone()
        state['grads'] = new_grads
        update_visualization()

    def update_range(change):
        state['range_min'], state['range_max'] = change['new']

        new_filter_image, new_grads = state['model'].winsorize_stacked_gradcam(
            state['input_tensor'].unsqueeze(0),
            state['stacked_gradcam'][state['range_min']:state['range_max']],
            state['importance_tensor'][state['range_min']:state['range_max']],
            interpolation_mode=state['interpolation_mode'],
            winsor_percentile=state['winsor_percentile']
        )
        state['filter_image'] = new_filter_image.clone()
        state['grads'] = new_grads
        update_visualization()

    def update_operation(change=None):
        if change is not None:
            state['operation'] = change['new']

        new_stacked_gradcam, _, new_importance_tensor = state['model'].get_gradcams_and_importance(
            state['input_tensor'].to(device).unsqueeze(0),
            state['predicted_class'],
            state['layer_names'],
            state['gradient_aggregation_method'],
            state['operation'],
            stack_relu=True,
            interpolation_mode=state['interpolation_mode']
        )

        state['stacked_gradcam'] = new_stacked_gradcam.clone()
        state['importance_tensor'] = new_importance_tensor.clone()

        new_filter_image, new_grads = state['model'].winsorize_stacked_gradcam(
            state['input_tensor'].unsqueeze(0),
            state['stacked_gradcam'][state['range_min']:state['range_max']],
            state['importance_tensor'][state['range_min']:state['range_max']],
            interpolation_mode=state['interpolation_mode'],
            winsor_percentile=state['winsor_percentile']
        )
        state['filter_image'] = new_filter_image.clone()
        state['grads'] = new_grads

        if change is not None:
            update_visualization()

    def update_interpolation(change):
        state['interpolation_mode'] = change['new']
        state['model'].storage.clear()
        update_operation(None)  # Recalculate based on current mode
        update_visualization()

    def update_mode(change=None):
        if change is None:
            mode = state['mode']
        else:
            mode = change['new']

        if mode in ['Grad-CAM++', 'Score-CAM', 'FullGrad', 'Ablation-CAM', 'Layer-CAM', 'Shapley-CAM']:
            info_text.value = f"<p style='font-size: 16px; color: #FF6B35;'><b>⏳ Computing {mode}...</b></p>"
            
            # Create loading placeholder
            loading_fig = plt.figure(figsize=(25, 5), dpi=80)
            loading_ax = loading_fig.add_subplot(111)
            loading_ax.text(0.5, 0.5, f'Computing {mode.upper()}...', 
                           ha='center', va='center', fontsize=24, color='#FF6B35', weight='bold')
            loading_ax.set_xlim(0, 1)
            loading_ax.set_ylim(0, 1)
            loading_ax.axis('off')
            loading_buf = io.BytesIO()
            plt.savefig(loading_buf, format='PNG', bbox_inches='tight', dpi=60)
            plt.close(loading_fig)
            image_widget.value = loading_buf.getvalue()
        
        state['mode'] = mode
        
        match state['mode']:
            case 'Winsor-CAM':
                # Enable all controls for Winsor-CAM
                model_selection_buttons.disabled = False
                image_selection_dropdown.disabled = False
                interpolation_buttons.disabled = False
                range_slider.disabled = False
                winsor_slider.disabled = False
                operation_buttons.disabled = False
                
                # Restore range from slider
                state['range_min'] = range_slider.value[0]
                state['range_max'] = range_slider.value[1]

                state['filter_image'], state['grads'] = state['model'].winsorize_stacked_gradcam(
                    state['input_tensor'].unsqueeze(0),
                    state['stacked_gradcam'][state['range_min']:state['range_max']],
                    state['importance_tensor'][state['range_min']:state['range_max']],
                    interpolation_mode=state['interpolation_mode'],
                    winsor_percentile=state['winsor_percentile']
                )

            case 'Naive Grad-CAM':
                # Enable all controls for Naive Grad-CAM
                model_selection_buttons.disabled = False
                image_selection_dropdown.disabled = False
                interpolation_buttons.disabled = False
                range_slider.disabled = True
                winsor_slider.disabled = True
                operation_buttons.disabled = True
                
                naive_gradcam = torch.mean(state['stacked_gradcam'], dim=0, keepdim=True)
                state['filter_image'] = naive_gradcam.clone()
                state['grads'] = torch.ones_like(state['stacked_gradcam'][:,0,0])
                state['range_min'] = 0
                state['range_max'] = state['stacked_gradcam'].shape[0]
                
            case 'Final Layer Grad-CAM':
                # Enable all controls for Final Layer Grad-CAM
                model_selection_buttons.disabled = False
                image_selection_dropdown.disabled = False
                interpolation_buttons.disabled = False
                range_slider.disabled = True
                winsor_slider.disabled = True
                operation_buttons.disabled = True
                
                final_layer_gradcam = state['stacked_gradcam'][-1].unsqueeze(0)
                state['filter_image'] = final_layer_gradcam.clone()
                state['grads'] = torch.zeros_like(state['stacked_gradcam'][:,0,0])
                state['grads'][-1] = 1.0
                
            case 'Grad-CAM++' | 'Score-CAM' | 'XGrad-CAM' | 'FullGrad' | 'Ablation-CAM' | 'Layer-CAM' | 'Shapley-CAM':
                # Disable all controls except Transparency, Mode, Binary Threshold, Colormap, and Model Selection
                range_slider.disabled = True
                winsor_slider.disabled = True
                operation_buttons.disabled = True
                interpolation_buttons.disabled = True
                
                # Get the CAM using the comparative method
                match state['mode']:
                    case 'Grad-CAM++':
                        cam = state['model'].get_cam_comparative(state['input_tensor'], 'gradcampp')
                    case 'Score-CAM':
                        cam = state['model'].get_cam_comparative(state['input_tensor'], 'scorecam')
                    case 'XGrad-CAM':
                        cam = state['model'].get_cam_comparative(state['input_tensor'], 'xgradcam')
                    case 'FullGrad':
                        cam = state['model'].get_cam_comparative(state['input_tensor'], 'fullgrad')
                    case 'Ablation-CAM':
                        cam = state['model'].get_cam_comparative(state['input_tensor'], 'ablation')
                    case 'Layer-CAM':
                        cam = state['model'].get_cam_comparative(state['input_tensor'], 'layercam')
                    case 'Shapley-CAM':
                        cam = state['model'].get_cam_comparative(state['input_tensor'], 'shapleycam')
                
                # Convert to tensor efficiently and ensure proper dtype
                if isinstance(cam, np.ndarray):
                    cam = torch.from_numpy(cam).float().to(state['input_tensor'].device)
                else:
                    cam = torch.tensor(cam, dtype=torch.float32, device=state['input_tensor'].device)
                
                # Ensure cam is 2D (squeeze out extra dimensions if needed)
                while cam.ndim > 2:
                    cam = cam.squeeze(0)
                
                # Add batch dimension for consistency
                state['filter_image'] = cam.unsqueeze(0) if cam.ndim == 2 else cam
                
                # Set grads based on method type
                if state['mode'] == 'FullGrad':
                    # FullGrad uses all layers, so show uniform importance
                    state['grads'] = torch.ones_like(state['stacked_gradcam'][:,0,0])
                else:
                    # Other methods use only the final layer
                    state['grads'] = torch.zeros_like(state['stacked_gradcam'][:,0,0])
                    state['grads'][-1] = 1.0
                
                state['range_min'] = 0
                state['range_max'] = state['stacked_gradcam'].shape[0]
        
        if change is not None:
            image_widget.value = create_combined_visualization()
            update_info_text()

    def update_image_selection(change=None):
        """Update everything when a new image is selected"""
        if change is None:
            # If no change provided, use the current dropdown value
            dataset_key = image_selection_dropdown.value
        else:
            dataset_key = change['new']
        state['dataset_key'] = dataset_key
        input_tensor, label_tensor, one_hot, class_name = dataset[dataset_key]
        input_tensor = input_tensor.unsqueeze(0).to(device)
        unnormalized_image = denormalize(input_tensor.squeeze(0).cpu()).squeeze(0).permute(1, 2, 0).numpy()
        
        state['base_image'] = torch.from_numpy(unnormalized_image).float()
        state['input_tensor'] = input_tensor.squeeze(0)
        state['sample_label'] = class_name
        state['label_tensor'] = label_tensor
        
        state['model'].eval()
        with torch.no_grad():
            output = state['model'](state['input_tensor'].unsqueeze(0).to(device))
        predicted_class = torch.argmax(output).item()
        state['output'] = output
        state['predicted_class'] = predicted_class
        
        state['model'].storage.clear()

        stacked_gradcam, gradcams, importance_tensor = state['model'].get_gradcams_and_importance(
            state['input_tensor'].to(device).unsqueeze(0),
            predicted_class, 
            state['layer_names'],
            state['gradient_aggregation_method'],
            state['operation'], 
            stack_relu=True,
            interpolation_mode=state['interpolation_mode']
        )
        
        state['importance_tensor'] = importance_tensor.clone()
        state['stacked_gradcam'] = stacked_gradcam.clone()
        # Recalculate based on current mode
        update_mode(None)

        
        image_widget.value = create_combined_visualization()
        update_info_text()
            

    
    def update_model_selection(change):
        """Update model when selection changes"""
        new_model_name = change['new']
        if new_model_name == state['model_name']:
            return  # No change needed
        
        # Handle Inception v3 + FullGrad incompatibility
        if new_model_name == 'inception_v3':
            # Remove FullGrad from options if it's currently available
            current_options = list(mode_buttons.options)
            if 'FullGrad' in current_options:
                current_options.remove('FullGrad')
                mode_buttons.options = current_options
                # If FullGrad was selected, switch to Winsor-CAM
                if state['mode'] == 'FullGrad':
                    mode_buttons.value = 'Winsor-CAM'
                    update_mode({'new': 'Winsor-CAM'})
                    return  # Exit early since update_mode will handle the rest
        else:
            # For non-Inception models, ensure FullGrad is available
            current_options = list(mode_buttons.options)
            if 'FullGrad' not in current_options:
                # Add FullGrad back to options
                all_options = ['Winsor-CAM', 'Naive Grad-CAM', 'Final Layer Grad-CAM', 
                               'Grad-CAM++', 'Score-CAM', 'FullGrad', 
                               'Ablation-CAM', 'Layer-CAM', 'Shapley-CAM']
                mode_buttons.options = all_options
        
        # Show loading message immediately
        info_text.value = f"<p style='font-size: 16px; color: #FF6B35;'><b>⏳ Loading {new_model_name}...</b></p>"
        
        # Create a simple loading placeholder image
        loading_fig = plt.figure(figsize=(25, 5), dpi=80)
        loading_ax = loading_fig.add_subplot(111)
        loading_ax.text(0.5, 0.5, f'Loading {new_model_name.upper()}...', 
                       ha='center', va='center', fontsize=24, color='#FF6B35', weight='bold')
        loading_ax.set_xlim(0, 1)
        loading_ax.set_ylim(0, 1)
        loading_ax.axis('off')
        loading_buf = io.BytesIO()
        plt.savefig(loading_buf, format='PNG', bbox_inches='tight', dpi=60)
        plt.close(loading_fig)
        image_widget.value = loading_buf.getvalue()
        
        
        new_model, new_layer_names, inception = load_model_for_ui(new_model_name)
        
        # Update dataset transforms for future image selections and re-fetch current image
        if inception:
            dataset.transforms_image.transforms[0] = transforms.Resize((299, 299))
            dataset.transforms_label.transforms[0] = transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.NEAREST)
            # Re-fetch the currently selected image from the dataset so transforms are applied exactly once
            input_tensor_new, label_tensor_new, one_hot_new, class_name_new = dataset[state['dataset_key']]
            input_tensor_new = input_tensor_new.unsqueeze(0).to(device)
            unnormalized_image = denormalize(input_tensor_new.squeeze(0).cpu()).squeeze(0).permute(1, 2, 0).numpy()
            state['base_image'] = torch.from_numpy(unnormalized_image).float()
            state['input_tensor'] = input_tensor_new.squeeze(0)
            state['label_tensor'] = label_tensor_new
            state['sample_label'] = class_name_new
        else:
            dataset.transforms_image.transforms[0] = transforms.Resize((224, 224))
            dataset.transforms_label.transforms[0] = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)
            # Re-fetch the currently selected image from the dataset so transforms are applied exactly once
            input_tensor_new, label_tensor_new, one_hot_new, class_name_new = dataset[state['dataset_key']]
            input_tensor_new = input_tensor_new.unsqueeze(0).to(device)
            unnormalized_image = denormalize(input_tensor_new.squeeze(0).cpu()).squeeze(0).permute(1, 2, 0).numpy()
            state['base_image'] = torch.from_numpy(unnormalized_image).float()
            state['input_tensor'] = input_tensor_new.squeeze(0)
            state['label_tensor'] = label_tensor_new
            state['sample_label'] = class_name_new
                
        # Update state
        state['model'] = new_model
        state['layer_names'] = new_layer_names
        state['model_name'] = new_model_name
        
        # Update range slider
        range_slider.max = len(new_layer_names)
        range_slider.value = (0, len(new_layer_names))  # Must be tuple
        state['range_min'] = 0
        state['range_max'] = len(new_layer_names)
        
        # Get prediction with new model
        state['model'].eval()
        with torch.no_grad():
            output = state['model'](state['input_tensor'].unsqueeze(0).to(device))
        predicted_class = torch.argmax(output).item()
        state['output'] = output
        state['predicted_class'] = predicted_class
        
        # Generate new GradCAMs
        state['model'].storage.clear()

        stacked_gradcam, gradcams, importance_tensor = state['model'].get_gradcams_and_importance(
            state['input_tensor'].to(device).unsqueeze(0),
            predicted_class, 
            state['layer_names'],
            state['gradient_aggregation_method'],
            state['operation'], 
            stack_relu=True,
            interpolation_mode=state['interpolation_mode']
        )
        
        state['importance_tensor'] = importance_tensor.clone()
        state['stacked_gradcam'] = stacked_gradcam.clone()

        # Recalculate based on current mode
        update_mode(None)
        
        image_widget.value = create_combined_visualization()
        update_info_text()

    def update_info_text():
        confidence_score = torch.softmax(state['output'], dim=1)[0][state['predicted_class']].item()
        deterministic_status = "Disabled (VGG16)" if state['model_name'] == 'vgg16' else "Enabled"
        info_text.value = f"""
        <p><b>Current Settings:</b></p>
        <ul style="column-count: 2; column-gap: 20px;">
            <li>Model: {state['model_name']}</li>
            <li>Deterministic Mode: {deterministic_status}</li>
            <li>Selected Image: {state['dataset_key']}</li>
            <li>True Label: {state['sample_label']}</li>
            <li>Predicted Class: {labels[state['predicted_class']]} (confidence: {confidence_score:.4f})</li>
            <li>Mode: {state['mode']}</li>
            <li>Winsor Percentile: {state['winsor_percentile']}%</li>
            <li>Layer Range: {state['range_min']} - {state['range_max']}</li>
            <li>Binary Threshold: {state['threshold']:.2f}</li>
        </ul>
        """

    # Create initial visualization
    img_bytes = create_combined_visualization()
    image_widget = WImage(
        value=img_bytes,
        format='png',
        width=image_width*5,
    )

    # Update info text initially
    update_info_text()

    # Attach observers
    model_selection_buttons.observe(update_model_selection, names='value')
    transparency_slider.observe(lambda c: update_simple_param(c, 'transparency'), names='value')
    winsor_slider.observe(update_winsor_percentile, names='value')
    threshold_slider.observe(lambda c: update_simple_param(c, 'threshold'), names='value')
    range_slider.observe(update_range, names='value')
    operation_buttons.observe(update_operation, names='value')
    interpolation_buttons.observe(update_interpolation, names='value')
    colormap_buttons.observe(lambda c: update_simple_param(c, 'colormap'), names='value')
    mode_buttons.observe(update_mode, names='value')
    image_selection_dropdown.observe(update_image_selection, names='value')

    # Create UI
    centered_ui = VBox([
        title_text,
        info_text,
        image_widget,
        winsor_slider,
        image_selection_dropdown,
        threshold_slider,
        transparency_slider,
        range_slider,
        model_selection_buttons,
        mode_buttons,
        operation_buttons,
        interpolation_buttons,
        colormap_buttons,
    ])

    centered_ui.layout = Layout(
        display='flex',
        flex_flow='column',
        align_items='stretch',
        width='100%'
    )

    display(centered_ui)

