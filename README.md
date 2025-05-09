# WinsorCAM

## Overview
This is the repository for the WinsorCAM project, which is a tool for interprepting the results of a CNN model. The tool is designed to help users understand the model's predictions by providing visualizations of where a model is localizing feature that are indicative of a certain class. This repository contains Python code that implements the WinsorCAM algorithm for some models like ResNet50, DenseNet121, VGG16, and InceptionV3. The code is made in Pytorch. This is a very preliminary version and may be modified in the future and is not majorly optimized.

## Usage

### Setting Up the Environment

To recreate the Conda environment used for this project, run:

```sh
conda env create -f environment.yml
```

### Running the example Jupyter Notebooks
The notebooks are currently location at the root of this repository. To run the notebooks, you can use your preferred Jupyter Notebook environment. After setting up you environment (this found in the usage section above), you can select the created environment in Jupyter Notebook then begin running the notebooks. Be aware the notebooks will walk you throught the process but you will need to change paths and other parameters to fit your needs. Also be aware data will be downloaded into a directory of your choosing and for ImageNet data, this can be quite large. 

There are currently two notebooks:
- `pascalvoc_example.ipynb`: This notebook demonstrates how to use the WinsorCAM algorithm on the Pascal VOC dataset. It includes code for loading the dataset and visualization. This models for this dataset can be shared on request if desired. The notebook cannot be run in full without either the models or making your own model and modifying the code within the models.py file.
- `imagenet_example.ipynb`: This notebook demonstrates how to use the WinsorCAM algorithm on the ImageNet dataset. It includes code for loading the dataset, and generating visualizations of the model's predictions. This notebook can be run in full without any needed models, but models will be downloaded from PyTorch. Also the data for this might be large so be aware of that.

## Example Output
Example gif outputs can be found the `examples` directory. These gifs show how a chosen Winsorization percentile can impact the output heatmap and also show how each layer is being weighted. New gifs can be generated when using the notebooks.