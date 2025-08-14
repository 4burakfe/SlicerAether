# SlicerAether Extension for 3D Slicer
This extension allows users to utilize SwinUNETR and UNET models to segment multimodality images

**Author**: Burak Demir, MD, FEBNM  
**Version**: 1.0  
**Contact**: 4burakfe@gmail.com

[![You can watch the module in action from here](https://img.youtube.com/vi/PFSEcLBYJf4/maxresdefault.jpg)](https://youtu.be/PFSEcLBYJf4)


## How To Install

This extension is not currently on Extension Manager of the 3D Slicer but can be installed manually.

For installation download this repository and extract the zip folder. 

Then in the 3D Slicer go to Edit->Application Settings->Modules->Additional Module Paths

Here click >> button and manually add Aether folder.

SlicerAether requires Pytorch to be installed to operate. 

You can install it from extension manager with PyTorchUtils extension.

If Pytorch Utils is not installed the module will not be shown.

It is highly recommended to have CUDA capable GPU and if so be sure you have installed CUDA enabled version of Pytorch.

You can train your own models with scripts provided here: [https://github.com/4burakfe/Claritas ](https://github.com/4burakfe/Claritas/tree/main/Segmentation%20Edition)

This link contains pretrained models to segment liver and its tumors with FDG PET/CT ready for use: https://github.com/4burakfe/SlicerAether/releases/tag/Trained_Models

You can test this module with the cases in here: https://github.com/4burakfe/SlicerAether/releases/tag/Sample_Cases

## Usage

First select the folder containing your models. Afterwards, select model to perform segmentation. The module will use sliding window inference to segment the input volumes. Normally, the model's .txt file should contain settings for the module and load up automatically. However, it is highly recommended to limit the input image with ROI. In addition please be aware that input images' shortest sides should be greater than the block size (which is 96 by default, that corresponds to 192mm with 2mm isotropic voxel size). If your volume is smaller you can try smaller block sizes such as 64x64x64 or 32x32x32 but larger is better.

## Dependencies

This extension uses several Python libraries inside Slicer's environment:
- `torch`
- `monai`
- `einops`

Torch must be installed with PyTorchUtils extension...
If PyTorchUtils is not installed the modules will not appear.
