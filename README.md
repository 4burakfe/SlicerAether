# SlicerAether Extension for 3D Slicer
This extension allows users to utilize SwinUNETR and UNET models to segment multimodality images

**Author**: Burak Demir, MD, FEBNM  
**Version**: 1.0  
**Contact**: 4burakfe@gmail.com

## Overview

This extension is not currently on Extension Manager of the 3D Slicer but can be installed manually.
For installation download this repository and extract the zip folder. 
Then in the 3D Slicer go to Edit->Application Settings->Modules->Additional Module Paths
Here click >> button and manually add Aether folder.

SlicerAether requires Pytorch to be installed to operate. 
You can install it from extension manager with PyTorchUtils extension.
If Pytorch Utils is not installed the module will not be shown.
It is highly recommended to have CUDA capable GPU and if so be sure you have installed CUDA enabled version of Pytorch.

You can train your own models with scripts provided here: https://github.com/4burakfe/Claritas 

This repository also contains pretrained models ready for use: https://github.com/4burakfe/SlicerPETDenoise/releases/tag/Models


You can test this module with the cases in here: https://github.com/4burakfe/SlicerPETDenoise_SampleCases/releases/tag/images


## Dependencies

This extension uses several Python libraries inside Slicer's environment:
- `torch`
- `monai`
- `einops`

Torch must be installed with PyTorchUtils extension...
If PyTorchUtils is not installed the modules will not appear.
