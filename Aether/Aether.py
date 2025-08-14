import os
import slicer
from slicer.ScriptedLoadableModule import *
import qt
import ctk
import vtk
import time  
import numpy as np
import configparser

  

class Aether(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Aether - PET/CT Segmentator"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = ["PyTorchUtils"]
        parent.contributors = ["Burak Demir, MD, FEBNM"]
        parent.helpText = """
        This module provides automated segmentation of medical images with SwinUNETR and UNET neural networks.
        """
        parent.acknowledgementText = """
        This file was developed by Burak Demir.
        """
        # **✅ Set the module icon**
        iconPath = os.path.join(os.path.dirname(__file__), "logo.png")
        self.parent.icon = qt.QIcon(iconPath)  # Assign icon to the module
        self.parent = parent

class AetherWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        # Create collapsible section
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)
        formLayout = qt.QFormLayout(parametersCollapsibleButton)
        
        
        # **✅ Load the banner image**
        moduleDir = os.path.dirname(__file__)  # Get module directory
        bannerPath = os.path.join(os.path.dirname(__file__), "banner.png")  # Change to your banner file



        
        self.architecture = qt.QComboBox()
        self.architecture.addItem("UNET")
        self.architecture.addItem("SwinUNETR")
        self.architecture.addItem("SwinUNETR+GCFN")
        formLayout.addRow("Select Model Architecture: ", self.architecture)



        # 1️⃣ Input Volume Selector (PET Image)
        self.inputVolumeSelector = slicer.qMRMLNodeComboBox()
        self.inputVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputVolumeSelector.selectNodeUponCreation = True
        self.inputVolumeSelector.addEnabled = False
        self.inputVolumeSelector.removeEnabled = False
        self.inputVolumeSelector.noneEnabled = False
        self.inputVolumeSelector.showHidden = False
        self.inputVolumeSelector.showChildNodeTypes = False
        self.inputVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.inputVolumeSelector.setToolTip("Select the Volume 1 for segmentation.")
        formLayout.addRow("Input Volume 1: ", self.inputVolumeSelector)


        # 1️⃣ Input Volume Selector (PET Image)
        self.inputVolumeSelector2 = slicer.qMRMLNodeComboBox()
        self.inputVolumeSelector2.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputVolumeSelector2.selectNodeUponCreation = True
        self.inputVolumeSelector2.addEnabled = False
        self.inputVolumeSelector2.removeEnabled = False
        self.inputVolumeSelector2.noneEnabled = False
        self.inputVolumeSelector2.showHidden = False
        self.inputVolumeSelector2.showChildNodeTypes = False
        self.inputVolumeSelector2.setMRMLScene(slicer.mrmlScene)
        self.inputVolumeSelector2.setToolTip("Select the Volume 2 for segmentation.")
        formLayout.addRow("Input Volume 2: ", self.inputVolumeSelector2)

        self.dualch_cbox = qt.QCheckBox()
        formLayout.addRow("Dual Volume Input (PET+CT): ", self.dualch_cbox)


        # --- Masking (pre-segmentation) ---
        self.useMaskCheck = qt.QCheckBox()
        formLayout.addRow("Mask with existing segmentation (pre):", self.useMaskCheck)

        self.maskSegSelector = slicer.qMRMLNodeComboBox()
        self.maskSegSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.maskSegSelector.selectNodeUponCreation = False
        self.maskSegSelector.addEnabled = False
        self.maskSegSelector.removeEnabled = False
        self.maskSegSelector.noneEnabled = True
        self.maskSegSelector.showHidden = False
        self.maskSegSelector.showChildNodeTypes = False
        self.maskSegSelector.setMRMLScene(slicer.mrmlScene)
        self.maskSegSelector.setToolTip("Segmentation to use as mask. Visible segments are combined.")
        formLayout.addRow("Mask Segmentation:", self.maskSegSelector)

        # --- Cropping (bounding box) ---
        self.useCropCheck = qt.QCheckBox()
        formLayout.addRow("Crop to ROI:", self.useCropCheck)

        # 2️⃣ ROI Selector (Select or Create New Markups ROI)
        self.roiSelector = slicer.qMRMLNodeComboBox()
        self.roiSelector.nodeTypes = ["vtkMRMLMarkupsROINode"]  # Use Markups ROI
        self.roiSelector.selectNodeUponCreation = True
        self.roiSelector.addEnabled = True  # Allow creation of new ROI
        self.roiSelector.removeEnabled = True  # Allow deletion of ROI
        self.roiSelector.noneEnabled = False
        self.roiSelector.showHidden = False
        self.roiSelector.showChildNodeTypes = False
        self.roiSelector.setMRMLScene(slicer.mrmlScene)
        self.roiSelector.setToolTip("Select or create a new Markups ROI.")
        formLayout.addRow("ROI: ", self.roiSelector)
        
        # 3️⃣ ROI Size Display (Text Box)
        self.roiSizeTextBox = qt.QLineEdit()
        self.roiSizeTextBox.setReadOnly(True)  # Make read-only
        self.roiSizeTextBox.setToolTip("Displays the ROI dimensions in mm.")
        formLayout.addRow("ROI Size (mm):", self.roiSizeTextBox)

        self.fill_outside_value = qt.QLineEdit()
        self.fill_outside_value.setText("0")
        formLayout.addRow("Value to fill outside of the ROI:", self.fill_outside_value)



        # Connect ROI selector change to update ROI size display
        self.roiSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateROISizeDisplay)

        # Connect ROI creation event to set default size
        self.roiSelector.connect("nodeAdded(vtkMRMLNode*)", self.setDefaultROISize) 















        # 📁 Select Model Folder Button and Display
        self.modelFolderPathEdit = qt.QLineEdit()
        self.modelFolderPathEdit.readOnly = True
        formLayout.addRow("Model Folder:", self.modelFolderPathEdit)

        self.selectFolderButton = qt.QPushButton("Select Model Folder")
        formLayout.addRow(self.selectFolderButton)

        self.selectFolderButton.connect("clicked(bool)", self.selectModelFolder)


        self.modelInfoBox = qt.QTextEdit()
        self.modelInfoBox.setReadOnly(True)
        self.modelInfoBox.setToolTip("Displays model info from .txt sidecar file.")

        self.modelselector = qt.QComboBox()

        models = []
        for dirpath, dirnames, filenames in os.walk(os.path.dirname(self.modelselector.currentText)):
           for filename in filenames:
              if filename.endswith(".pth"):
                 models.append(filename)
        self.modelselector.addItems(models)
        self.loadModelFolderPath()

        formLayout.addRow("Select Model: ", self.modelselector)


        formLayout.addRow("Model Info:", self.modelInfoBox)
        self.modelInfoBox.setPlainText("Select a model to display info.")

        preprocessCollapsibleButton = ctk.ctkCollapsibleButton()
        preprocessCollapsibleButton. collapsed = True
        preprocessCollapsibleButton.text = "PreProcessing Settings"
        self.layout.addWidget(preprocessCollapsibleButton)
        formLayoutpreprocess = qt.QFormLayout(preprocessCollapsibleButton)



        self.resample_voxel_size = qt.QLineEdit()
        self.resample_voxel_size.setText("[2,2,2]")
        formLayoutpreprocess.addRow("Voxel Spacing for Resample:", self.resample_voxel_size)

        self.rescale_intensity_input = qt.QLineEdit()
        self.rescale_intensity_input.setText("-135,215")
        formLayoutpreprocess.addRow("Input Intensity Range to Rescale Vol1:", self.rescale_intensity_input)

        self.rescale_intensity_output = qt.QLineEdit()
        self.rescale_intensity_output.setText("0,10")
        formLayoutpreprocess.addRow("Output Intensity Range to Rescale Vol1:", self.rescale_intensity_output)

        self.no_rescalecbox = qt.QCheckBox()
        formLayoutpreprocess.addRow("No Rescale Vol1: ", self.no_rescalecbox)



        self.rescale_intensity_input2 = qt.QLineEdit()
        self.rescale_intensity_input2.setText("-135,215")
        formLayoutpreprocess.addRow("Input Intensity Range to Rescale Vol2:", self.rescale_intensity_input2)

        self.rescale_intensity_output2 = qt.QLineEdit()
        self.rescale_intensity_output2.setText("0,10")
        formLayoutpreprocess.addRow("Output Intensity Range to Rescale Vol2:", self.rescale_intensity_output2)


        self.no_rescalecbox2 = qt.QCheckBox()
        formLayoutpreprocess.addRow("No Rescale Vol2: ", self.no_rescalecbox2)



        self.denoise_block_size = qt.QLineEdit()
        self.denoise_block_size.setText("(96,96,96)")
        formLayout.addRow("Block Size for denoising:", self.denoise_block_size)




        UNETCollapsibleButton = ctk.ctkCollapsibleButton()
        UNETCollapsibleButton.collapsed = True
        UNETCollapsibleButton.text = "Settings for UNET"
        self.layout.addWidget(UNETCollapsibleButton)
        formLayoutUNET = qt.QFormLayout(UNETCollapsibleButton)


        # Strides ComboBox
        self.strideComboBox = qt.QLineEdit()
        self.strideComboBox.setText("(2,2,2,2)")
        formLayoutUNET.addRow("Strides:", self.strideComboBox)

        # Channels Entry
        self.channels = qt.QLineEdit()
        self.channels.setText("(128,256,512,1024,2048)")
        formLayoutUNET.addRow("Channels:", self.channels)



        # Res Units
        self.resUnitSpinBox = qt.QSpinBox()
        self.resUnitSpinBox.setMinimum(1)
        self.resUnitSpinBox.setMaximum(10)
        self.resUnitSpinBox.setValue(2)
        formLayoutUNET.addRow("Residual Units:", self.resUnitSpinBox)

        # Res Units
        self.downkernelSpinBox = qt.QSpinBox()
        self.downkernelSpinBox.setMinimum(1)
        self.downkernelSpinBox.setMaximum(11)
        self.downkernelSpinBox.setValue(3)
        formLayoutUNET.addRow("Down Kernel:", self.downkernelSpinBox)

        self.upkernelSpinBox = qt.QSpinBox()
        self.upkernelSpinBox.setMinimum(1)
        self.upkernelSpinBox.setMaximum(11)
        self.upkernelSpinBox.setValue(3)
        formLayoutUNET.addRow("Up Kernel:", self.upkernelSpinBox)



        SwinCollapsibleButton = ctk.ctkCollapsibleButton()
        SwinCollapsibleButton.collapsed = True
        SwinCollapsibleButton.text = "Settings for SwinUNETR"
        self.layout.addWidget(SwinCollapsibleButton)
        formLayoutSwin = qt.QFormLayout(SwinCollapsibleButton)

        self.heads = qt.QLineEdit()
        self.heads.setText("(3,6,12,24)")
        formLayoutSwin.addRow("Number of Heads:", self.heads)

        self.depths = qt.QLineEdit()
        self.depths.setText("(2,2,2,2)")
        formLayoutSwin.addRow("Depths:", self.depths)

        # Channels Entry
        self.swinfeaturesize = qt.QSpinBox()
        self.swinfeaturesize.setMinimum(4)
        self.swinfeaturesize.setMaximum(256)
        self.swinfeaturesize.setValue(24)
        formLayoutSwin.addRow("Feature Size:", self.swinfeaturesize)

        # Channels Entry
        self.dropoutrate = qt.QLineEdit()
        self.dropoutrate.setText("0.0")
        formLayoutSwin.addRow("Dropout Path Rate:", self.dropoutrate)


        # 5️⃣ Output Log Text Box
        self.outputTextBox = qt.QTextEdit()
        self.outputTextBox.setReadOnly(True)
        self.outputTextBox.setToolTip("Displays processing logs and results.")
        formLayout.addRow("Processing Log:", self.outputTextBox)
        
        self.forceCPUcbox = qt.QCheckBox()
        formLayout.addRow("FORCE CPU: ", self.forceCPUcbox)
        
        # 6️⃣ Calculate Button
        self.calculateButton = qt.QPushButton("Start Segmentation")
        self.calculateButton.toolTip = "Start the segmentation process"
        self.calculateButton.enabled = True
        formLayout.addRow(self.calculateButton)

        # Connect Calculate button to function
        self.calculateButton.connect("clicked(bool)", self.onCalculateButtonClicked)
        
        self.segmentname = "Segment"
        # Info Text Box
        infoTextBox = qt.QTextEdit()
        infoTextBox.setReadOnly(True)  # Make the text box read-only
        infoTextBox.setPlainText(
            "This module provides automatic segmentation with ML models on medical images.\n"
            "This module is NOT a medical device. Research use only.\n"
            "Developed by: Burak Demir, MD, FEBNM \n"
            "For support, questions and feedback: 4burakfe@gmail.com\n"
            "Version: v1.0"
        )
        infoTextBox.setToolTip("Module information and instructions.")  # Add a tooltip for additional help
        self.layout.addWidget(infoTextBox)
        self.modelselector.connect("currentIndexChanged(int)", self.updateModelInfoBox)
        self.updateModelInfoBox()

        if os.path.exists(bannerPath):
            bannerLabel = qt.QLabel()
            bannerPixmap = qt.QPixmap(bannerPath)  # Load image
            bannerLabel.setPixmap(bannerPixmap.scaledToWidth(400, qt.Qt.SmoothTransformation))  # Adjust width

            # **Center the image**
            bannerLabel.setAlignment(qt.Qt.AlignCenter)

            # **Add to layout**
            self.layout.addWidget(bannerLabel)
        else:
            print(f"❌ WARNING: Banner file not found at {bannerPath}")





    def updateModelInfoBox(self):
        modelName = self.modelselector.currentText
        fullPath = self.modelFolderPathEdit.text+"/" + os.path.splitext(modelName)[0] + ".txt"
        if os.path.exists(fullPath):
            with open(fullPath, "r", encoding="utf-8") as f:
                info_lines = f.readlines()

            # Populate text box
            self.modelInfoBox.setPlainText("".join(info_lines))

            # Try to extract parameters
            for line in info_lines:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    key = str(key.strip().lower())
                    value = str(value.strip())
                    try:
                        if key == "strides":
                            self.strideComboBox.setText(value)
                        elif key == "dual_channel":
                            if value == "true":
                                self.dualch_cbox.setChecked(True)
                            else:
                                self.dualch_cbox.setChecked(False)
                        elif key == "channels":
                            self.channels.setText(value)
                        elif key == "res_units":
                            self.resUnitSpinBox.setValue(int(value))
                        elif key == "down_kernel":
                            self.downkernelSpinBox.setValue(int(value))
                        elif key == "up_kernel":
                            self.upkernelSpinBox.setValue(int(value))
                        elif key == "depths":
                            self.depths.setText(value)                        
                        elif key == "num_heads":
                            self.heads.setText(value)
                        elif key == "feature_size":
                            self.swinfeaturesize.setValue(int(value))
                        elif key == "do_rate":
                            self.dropoutrate.setText(value)
                        elif key == "block_size":
                            self.denoise_block_size.setText(value)
                        elif key == "voxel_spacing":
                            self.resample_voxel_size.setText(value)
                        elif key == "architecture":
                            if value == "SwinUNETR":
                                self.architecture.setCurrentIndex(1)
                            elif value == "UNET":
                                self.architecture.setCurrentIndex(0)
                            else:
                                self.architecture.setCurrentIndex(2)
                        elif key == "mask_before":
                            if value == "true":
                                self.useMaskCheck.setChecked(True)
                                print(value)
                            else:
                                self.useMaskCheck.setChecked(False)

                        elif key == "crop_before":
                            if value == "true":
                                self.useCropCheck.setChecked(True)
                            else:
                                self.useCropCheck.setChecked(False)

                        elif key == "input_intensity_vol1":
                            self.rescale_intensity_input.setText(value)
                        elif key == "input_intensity_vol2":
                            self.rescale_intensity_input2.setText(value)
                        elif key == "output_intensity_vol1":
                            self.rescale_intensity_output.setText(value)
                        elif key == "output_intensity_vol2":
                            self.rescale_intensity_output2.setText(value)
                        elif key == "no_rescale_vol1":
                            if value == "true":
                                self.no_rescalecbox.setChecked(True)
                            else:
                                self.no_rescalecbox.setChecked(False)
                        elif key == "no_rescale_vol2":
                            if value == "true":
                                self.no_rescalecbox2.setChecked(True)
                            else:
                                self.no_rescalecbox2.setChecked(False)
                        elif key == "segment_name":
                            self.segmentname = value
                        elif key == "fill_outside_roi_value":
                            self.fill_outside_value.setText(value)


                    except Exception as e:
                        print(f"⚠️ Error parsing parameter '{key}': {e}")
        else:
            self.modelInfoBox.setPlainText("ℹ️ No description file found.")



    def onCalculateButtonClicked(self):
        self.outputTextBox.clear()
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            self.outputTextBox.append("❌ PyTorch is not installed. Please install it before running this module.")
            return None

        self.outputTextBox.append("🚀 Starting AI denoising process...")
        try:
            from monai.networks.nets import UNet
        except ImportError:
            msgBox = qt.QMessageBox()
            msgBox.setIcon(qt.QMessageBox.Warning)
            msgBox.setWindowTitle("MONAI Not Installed")
            msgBox.setText("The MONAI library is required but not installed.\nI can install it. Would you like me to install it now? You may want to restart Slicer after installation.")
            msgBox.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
            response = msgBox.exec_()

            if response == qt.QMessageBox.Yes:
                import subprocess, sys
                try:
                    self.outputTextBox.append("📦 Installing MONAI, please wait...")
                    subprocess.check_call([sys.executable,  "-m","pip", "install", "monai"])
                    self.outputTextBox.append("✅ MONAI installed successfully.")
                    
                except Exception as e:
                    self.outputTextBox.append(f"❌ Failed to install MONAI: {e}")
                    return None
            else:
                self.outputTextBox.append("❌ MONAI installation canceled by user.")
                return None
        inputVolumeNode = self.inputVolumeSelector.currentNode()
        if inputVolumeNode is None:
            self.outputTextBox.append("❌ Process stopped. Please select a valid volume...")
            return None
        try:
            import einops
        except ImportError:
            msgBox = qt.QMessageBox()
            msgBox.setIcon(qt.QMessageBox.Warning)
            msgBox.setWindowTitle("einops is Not Installed")
            msgBox.setText("The einops library is required but not installed.\nI can install it. Would you like me to install it now? You may want to restart Slicer after installation.")
            msgBox.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
            response = msgBox.exec_()

            if response == qt.QMessageBox.Yes:
                import subprocess, sys
                try:
                    self.outputTextBox.append("📦 Installing MONAI, please wait...")
                    subprocess.check_call([sys.executable,  "-m","pip", "install", "einops==0.6.1"])

                    self.outputTextBox.append("✅ MONAI installed successfully.")
                    
                except Exception as e:
                    self.outputTextBox.append(f"❌ Failed to install MONAI: {e}")
                    return None
            else:
                self.outputTextBox.append("❌ MONAI installation canceled by user.")
                return None

        #Load AI Model
        model = self.load_model()
        if model is None:
            self.outputTextBox.append("❌ AI model loading failed. Stopping process.")
            return

        volumesLogic = slicer.modules.volumes.logic()
        outputVolumeNode = volumesLogic.CloneVolume(slicer.mrmlScene, inputVolumeNode, f"{inputVolumeNode.GetName()}_DN_{self.modelselector.currentText}")



        resample_parameters = {
            "InputVolume": inputVolumeNode.GetID(),  # ✅ Corrected to use GetID()
            "OutputVolume": outputVolumeNode.GetID(),  # ✅ Corrected to use GetID()
            "outputPixelSpacing": eval(self.resample_voxel_size.text),  # ✅ Ensured correct format
            "interpolationType": "linear"
        }
        resampleSuccess = slicer.cli.runSync(slicer.modules.resamplescalarvolume, None, resample_parameters)
        # --- Optional pre-processing: mask and/or crop ---
        # 1) Mask first (if enabled)
        if self.useMaskCheck.checkState():
            segNode = self.maskSegSelector.currentNode()
            if segNode is None:
                self.outputTextBox.append("⚠️ Masking requested but no segmentation selected. Skipping mask.")
            else:
                try:
                    self.outputTextBox.append("🩺 Applying mask using visible segments...")




                    labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode","maskLabel")
                    slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(segNode, labelNode, outputVolumeNode)
                    slicer.cli.runSync(slicer.modules.maskscalarvolume, None, {
                        "InputVolume": outputVolumeNode.GetID(),
                        "MaskVolume": labelNode.GetID(),
                        "OutputVolume": outputVolumeNode.GetID(),
                        "OutsideValue": float(self.fill_outside_value.text),
                        "Not": False
                    })
                    slicer.mrmlScene.RemoveNode(labelNode)

                    print("Masked volume:", outputVolumeNode.GetName())
                except Exception as e:
                    self.outputTextBox.append(f"❌ Masking failed: {e}")




        # 2) Crop to bounding box (after masking if both are enabled)
        if self.useCropCheck.checkState():
            self.outputTextBox.append("✂️ Computing bounding box...")
            roiNode = self.roiSelector.currentNode()

            cropVolumeLogic = slicer.modules.cropvolume.logic()
            cropVolumeLogic.CropVoxelBased(roiNode, outputVolumeNode, outputVolumeNode, False)


















        inputImage = slicer.util.arrayFromVolume(outputVolumeNode)  # shape: [slices, height, width]
        if self.no_rescalecbox.isChecked() == False:
            min_range_input, max_range_input = self.rescale_intensity_input.text.split(",", 1)
            min_range_output, max_range_output = self.rescale_intensity_output.text.split(",", 1)

            # Convert to integers
            min_range_input = float(min_range_input)
            max_range_input = float(max_range_input)
            min_range_output = float(min_range_output)
            max_range_output = float(max_range_output)

            from monai.transforms import ScaleIntensityRange

            transform = ScaleIntensityRange(
                a_min=min_range_input,      # e.g. min_range_input
                a_max=max_range_input,      # e.g. max_range_input
                b_min=min_range_output,     # e.g. min_range_output
                b_max=max_range_output,     # e.g. max_range_output
                clip=True                  # clip values outside input range
            )

            inputImage = transform(inputImage)
            self.outputTextBox.append(f"Input1 was rescaled!")





        inputImage2 = None
        if self.dualch_cbox.checkState():
            inputImage2Node = self.inputVolumeSelector2.currentNode()
            if inputImage2Node is None:
                self.outputTextBox.append("❌ Dual channel selected, but second volume not provided.")
                return None

            self.outputTextBox.append("📐 Resampling CT volume to match PET...")
            slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ResampledVol2")
            resampledimageNode = volumesLogic.CloneVolume(slicer.mrmlScene, inputVolumeNode, "ResampledVol2")


            # Set parameters explicitly
            resample_parameters2 = {
                "InputVolume": inputImage2Node.GetID(),  # ✅ Corrected to use GetID()
                "OutputVolume": resampledimageNode.GetID(),  # ✅ Corrected to use GetID()
                "outputPixelSpacing": eval(self.resample_voxel_size.text),  # ✅ Ensured correct format
                "interpolationType": "linear"
            }
            resampleSuccess = slicer.cli.runSync(slicer.modules.resamplescalarvolume, None, resample_parameters2)

            if(resampleSuccess):
                self.outputTextBox.append("✅ Resampling completed successfully.")
            else:
                self.outputTextBox.append("❌ Resampling failed.")
            inputImage2 = slicer.util.arrayFromVolume(resampledimageNode)

            if self.no_rescalecbox2.isChecked() == False:
                min_range_input, max_range_input = self.rescale_intensity_input2.text.split(",", 1)
                min_range_output, max_range_output = self.rescale_intensity_output2.text.split(",", 1)

                # Convert to integers
                min_range_input = float(min_range_input)
                max_range_input = float(max_range_input)
                min_range_output = float(min_range_output)
                max_range_output = float(max_range_output)

                from monai.transforms import ScaleIntensityRange

                transform = ScaleIntensityRange(
                    a_min=min_range_input,      # e.g. min_range_input
                    a_max=max_range_input,      # e.g. max_range_input
                    b_min=min_range_output,     # e.g. min_range_output
                    b_max=max_range_output,     # e.g. max_range_output
                    clip=True                  # clip values outside input range
                )

                inputImage2 = transform(inputImage2)
                self.outputTextBox.append(f"Input2 was rescaled!")





        inputTensor = torch.tensor(inputImage).float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)
        self.outputTextBox.append(f"Input image size is: {inputTensor.shape[-3:]}")



        inputTensor2 = None
        if inputImage2 != None:
            inputTensor2 = torch.tensor(inputImage2).float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)
            self.outputTextBox.append(f"Input image2 size is: {inputTensor2.shape[-3:]}")

        start_time = time.time()  # ⏱️ Start timer
        from monai.inferers import sliding_window_inference
        output_Tensor = None
        if self.dualch_cbox.checkState():
            combinedTensor=torch.cat([inputTensor, inputTensor2], dim=1)
            with torch.no_grad():
                device = next(model.parameters()).device
                combinedTensor = combinedTensor.to(device)
                output_Tensor = sliding_window_inference(
                    inputs=combinedTensor,
                    roi_size=eval(self.denoise_block_size.text),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.25,
                    mode="gaussian"
                )
            self.outputTextBox.append(f"Segmentation is done! Output image size: {output_Tensor.shape[-3:]}")
            elapsed = time.time() - start_time  # ⏱️ End timer
            self.outputTextBox.append(f"⏱️ Inference time: {elapsed:.2f} seconds")  # ✅ Log time
            outputArray = output_Tensor.squeeze().cpu().numpy()
            outputArray = np.clip(outputArray, 0, None)  # remove negative values
            binaryMask = (outputArray > 0.5).astype(np.uint8)

            slicer.util.updateVolumeFromArray(outputVolumeNode, outputArray)
            self.outputTextBox.append(f"Output image name is named as: {outputVolumeNode.GetName()}")
            del model
            del outputArray
            del inputTensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            import gc
            gc.collect()

        else:
            with torch.no_grad():
                device = next(model.parameters()).device
                inputTensor = inputTensor.to(device)
                outputTensor = sliding_window_inference(
                    inputs=inputTensor,
                    roi_size=(96, 96, 96),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.25,
                    mode="gaussian"
                )
            self.outputTextBox.append(f"Segmentation is done! Output image size: {outputTensor.shape[-3:]}")
            elapsed = time.time() - start_time  # ⏱️ End timer
            self.outputTextBox.append(f"⏱️ Inference time: {elapsed:.2f} seconds")  # ✅ Log time
            outputArray = outputTensor.squeeze().cpu().numpy()
            outputArray = np.clip(outputArray, 0, None)  # remove negative values
            binaryMask = (outputArray > 0.5).astype(np.uint8)

            self.outputTextBox.append(f"Output image size is resized to: {outputArray.shape}")

 
 
            
            labelVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", self.segmentname)

            # Create empty segmentation with predefined segment names
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"AetherSegmentation_{self.modelselector.currentText}")

            
            
            slicer.util.updateVolumeFromArray(labelVolumeNode, binaryMask)

            labelVolumeNode.CopyOrientation(outputVolumeNode)
            labelVolumeNode.SetSpacing(outputVolumeNode.GetSpacing())
            labelVolumeNode.SetOrigin(outputVolumeNode.GetOrigin())
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelVolumeNode, segmentationNode)
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(outputVolumeNode)

            # Get segmentation display node
            displayNode = segmentationNode.GetDisplayNode()

            # Loop through all segments and assign random colors
            import random
            for segmentId in segmentationNode.GetSegmentation().GetSegmentIDs():
                r, g, b = [random.random() for _ in range(3)]
                segmentationNode.GetSegmentation().GetSegment(segmentId).SetColor(r, g, b)
            # 0 = fully transparent in slice views
            displayNode.SetOpacity2DFill(0.10)
            # Thickness in pixels for slice intersections
            displayNode.SetSliceIntersectionThickness(2)


            
            slicer.mrmlScene.RemoveNode(labelVolumeNode)

            slicer.mrmlScene.RemoveNode(outputVolumeNode)

            self.outputTextBox.append(f"Output image name is named as: {outputVolumeNode.GetName()}")
            del model
            del outputArray
            del outputTensor
            del inputTensor

            for var in ["inputImage", "ctImage", "resampledCTNode", "predicted_noise", "combinedTensor",
                        "current_shape", "target_shape", "pet_shape", "ct_shape", "start_time", "elapsed"]:
                if var in locals():
                    del locals()[var]

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()









    def load_model(self):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            self.outputTextBox.append("❌ PyTorch is not installed. Please install it before running this module.")
            return None

        try:
            from monai.networks.nets import UNet,SwinUNETR
        except ImportError:
            self.outputTextBox.append("❌ MONAI is not installed. Please install it before running this module.")
            return None
        """
        Load the trained PyTorch model from model.pth.
        """
        model_path = self.modelFolderPathEdit.text + "/" + self.modelselector.currentText
          
        if not os.path.exists(model_path):
            self.outputTextBox.append(f"❌ Model file not found: {model_path}")
            return None

        self.outputTextBox.append(f"📌 Loading AI model from: {model_path}")
        
        
        # Parse user inputs
        strides = eval(self.strideComboBox.text)
        res_units = self.resUnitSpinBox.value
        num_heads = eval(self.heads.text)
        depths = eval(self.depths.text)
        do_rate= float(self.dropoutrate.text)
        channels = eval(self.channels.text)
        from monai import __version__ as monai_version
        from packaging import version
        class DenoiseUNet(nn.Module):
            def __init__(self, in_channels=1, out_channels=1,  channels=(32, 64, 128, 256,512), num_res_units=2,strides=(2, 2, 2, 2),kernel_size=3,up_kernel_size=3):
                super(DenoiseUNet, self).__init__()
        
                # Use MONAI's 3D U-Net as the denoising backbone
                self.unet = UNet(
                    strides=strides,
                    num_res_units=num_res_units,
                    kernel_size=kernel_size,
                    up_kernel_size=up_kernel_size,           
                    spatial_dims=3,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    channels=channels
                )

            def forward(self, x):
                return self.unet(x)  # ✅ Predict noise only

        class SwinDenoiser(nn.Module):
            def __init__(self,  in_channels=1, out_channels=1, feature_size=48,heads=(6,12,24,48),depths=(2,3,3,2),do_rate=0.1):
                super(SwinDenoiser, self).__init__()
                self.model = SwinUNETR(
                    num_heads = heads,
                    use_v2=True,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    feature_size=feature_size,
                    depths = depths,
                    dropout_path_rate=do_rate,
                    **({"img_size": (96, 96, 96)} if version.parse(monai_version) < version.parse("1.5") else {}),
                    use_checkpoint=True
                )

            def forward(self, x):
                return self.model(x)
        class GCFN(nn.Module):
            def __init__(self, dim):
                super(GCFN, self).__init__()
                self.norm = nn.LayerNorm(dim)

                self.fc1 = nn.Linear(dim, dim)
                self.fc2 = nn.Linear(dim, dim)
                self.fc0 = nn.Linear(dim, dim)

                self.conv1 = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim)
                self.conv2 = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim)

            def forward(self, x):
                # x: (B, C, D, H, W)
                B, C, D, H, W = x.shape
                x_ = x.permute(0, 2, 3, 4, 1).contiguous().view(B * D * H * W, C)

                x1 = self.fc1(self.norm(x_))
                x2 = self.fc2(self.norm(x_))

                x1 = x1.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
                x2 = x2.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)

                gate = F.gelu(self.conv1(x1)) * self.conv2(x2)

                gate = gate.permute(0, 2, 3, 4, 1).contiguous().view(B * D * H * W, C)
                out = self.fc0(gate).view(B, D, H, W, C).permute(0, 4, 1, 2, 3)

                return out + x
        class SwinGCFN(nn.Module):
            def __init__(self,  in_channels=1, out_channels=1, feature_size=48,heads=(6,12,24,48),depths=(2,3,3,2),do_rate=0.1):
                super(SwinGCFN, self).__init__()
                self.model = SwinUNETR(
                    num_heads = heads,
                    use_v2=True,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    feature_size=feature_size,
                    depths = depths,
                    dropout_path_rate=do_rate,
                    **({"img_size": (64, 64, 64)} if version.parse(monai_version) < version.parse("1.5") else {}),
                    use_checkpoint=True
                )
                self.gcfn = GCFN(dim=out_channels)

            def forward(self, x):
                out = self.model(x)      # (B, 1, D, H, W)
                out = self.gcfn(out)     # Apply GCFN enhancement
                return out


        use_cuda = False
        if torch.cuda.is_available():
            try:
                total_vram = torch.cuda.get_device_properties(0).total_memory
                vram_gb = total_vram / (1024 ** 3)  # Convert bytes to GB
                if vram_gb >= 1.9:
                    use_cuda = True
                    self.outputTextBox.append(f"✅ CUDA available with {vram_gb:.1f} GB VRAM — using GPU.")
                else:
                    self.outputTextBox.append(f"⚠️ CUDA available but only {vram_gb:.1f} GB VRAM — using CPU fallback.")
            except Exception as e:
                self.outputTextBox.append(f"⚠️ Could not check VRAM: {e}. Using CPU.")


        if self.forceCPUcbox.checkState():
            use_cuda = False
            self.outputTextBox.append(f"⚠️ Force CPU is selected. Using CPU.")


        # Instantiate model on correct device
        device = torch.device("cuda" if use_cuda else "cpu")


        if self.architecture.currentText=="UNET":
            if self.dualch_cbox.checkState()==False:
                model = DenoiseUNet(in_channels = 1, channels=channels, num_res_units=res_units,strides=strides,kernel_size=self.downkernelSpinBox.value,up_kernel_size=self.upkernelSpinBox.value).to(device)
            else:
                model = DenoiseUNet(in_channels = 2, channels=channels, num_res_units=res_units,strides=strides,kernel_size=self.downkernelSpinBox.value,up_kernel_size=self.upkernelSpinBox.value).to(device)
        elif self.architecture.currentText=="SwinUNETR":
            if self.dualch_cbox.checkState()==False:
                model = SwinDenoiser(in_channels = 1, feature_size=self.swinfeaturesize.value ,heads=num_heads,depths=depths,do_rate=do_rate).to(device)
            else:
                model = SwinDenoiser(in_channels = 2, feature_size=self.swinfeaturesize.value ,heads=num_heads,depths=depths,do_rate=do_rate).to(device)
        elif self.architecture.currentText=="SwinUNETR+GCFN":
            if self.dualch_cbox.checkState()==False:
                model = SwinGCFN(in_channels = 1, feature_size=self.swinfeaturesize.value ,heads=num_heads,depths=depths,do_rate=do_rate).to(device)
            else:
                model = SwinGCFN(in_channels = 2, feature_size=self.swinfeaturesize.value ,heads=num_heads,depths=depths,do_rate=do_rate).to(device)


        model.load_state_dict(torch.load(model_path, device))
        model.eval()  # Set to evaluation mode (no gradients needed)

        self.outputTextBox.append("✅ AI model loaded successfully!")
        return model
        
    def selectModelFolder(self):
        folder = qt.QFileDialog.getExistingDirectory()
        if folder:
            self.modelFolderPathEdit.setText(folder)
            self.saveModelFolderPath(folder)
            self.refreshModelSelector(folder)

    def saveModelFolderPath(self, folder):
        ini_path = os.path.join(os.path.dirname(__file__), "model_config.ini")
        config = configparser.ConfigParser()
        config["ModelFolder"] = {"path": folder}
        with open(ini_path, "w") as configfile:
            config.write(configfile)

    def loadModelFolderPath(self):
        ini_path = os.path.join(os.path.dirname(__file__), "model_config.ini")
        config = configparser.ConfigParser()
        if os.path.exists(ini_path):
            config.read(ini_path)
            folder = config.get("ModelFolder", "path", fallback="")
            if os.path.exists(folder):
                self.modelFolderPathEdit.setText(folder)
                self.refreshModelSelector(folder)

    def refreshModelSelector(self, folder):
        self.modelselector.clear()
        models = [f for f in os.listdir(folder) if f.endswith(".pth")]
        self.modelselector.addItems(models)



    def setDefaultROISize(self, roiNode):
        """
        Sets the default size and centers the ROI in the geometric center of the selected volume upon creation.
        Also ensures that the ROI size updates dynamically when changed.
        """
        if roiNode!= None:
            # Set default size
            roiNode.SetSize(400, 300, 350)

            # Get the selected volume
            inputVolumeNode = self.inputVolumeSelector.currentNode()

            # Compute the volume's geometric center in RAS coordinates
            bounds = [0] * 6
            inputVolumeNode.GetRASBounds(bounds)
            centerX = (bounds[0] + bounds[1]) / 2
            centerY = (bounds[2] + bounds[3]) / 2
            centerZ = (bounds[4] + bounds[5]) / 2

            # Set ROI center
            roiNode.SetCenter(centerX, centerY, centerZ)

            # Connect ROI size change event to dynamically update size
            roiNode.AddObserver(slicer.vtkMRMLMarkupsROINode.PointModifiedEvent, self.updateROISizeDisplay)

            # Update ROI size display immediately after creation
            self.updateROISizeDisplay()        
        
        
    def updateROISizeDisplay(self, caller=None, event=None):
        """
        Updates the ROI size display in millimeters, considering voxel spacing.
        Runs whenever the ROI size is modified.
        """
        
        roiNode = self.roiSelector.currentNode()
        if roiNode!= None:
            if self.inputVolumeSelector.currentNode()!=None :
                inputVolumeNode = self.inputVolumeSelector.currentNode()
            
                # Get the ROI size in voxel units
                size = roiNode.GetSize()  # Get size in voxel units

                # Update text box with size in mm
                self.roiSizeTextBox.setText(f"{size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm")        



