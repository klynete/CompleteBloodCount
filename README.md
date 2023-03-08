
# Object Detection using Faster R-CNN for Blood Smear Images

This repository contains code for performing object detection on a dataset of blood smear images using Jupyter notebooks. The objective of the task is to detect and classify three different cell types: red blood cells (RBCs), white blood cells (WBCs), and platelets.

We have used the Faster R-CNN algorithm to perform object detection on the images. The Faster R-CNN model is a popular deep learning model for object detection, which achieves state-of-the-art performance on a variety of object detection tasks. We trained the model on a dataset of 300 blood smear images, which has been annotated with the locations of the three different cell types.

The code in this repository includes the necessary Jupyter notebooks for training the Faster R-CNN model on the blood smear dataset, as well as for evaluating the performance of the trained model. 

Dataset can be downloaded from [link](https://github.com/MahmudulAlam/Complete-Blood-Cell-Count-Dataset). Images are 640*480, all labels for cell types and boxes coordinates are in .xml files

** Data wrangling.ipynb ** has parsing .xml files to dataframes, EDA for provided boxes and cell types.
** Data preparation and Modeling.ipynb ** has next process:
For this project we’ll use the Faster R-CNN model from the torchvision module. The pretrained Faster R-CNN model provided by PyTorch's torchvision package is pre-trained on the COCO dataset using ResNet50-FPN (Feature Pyramid Network) as its backbone architecture. COCO dataset consists of more than 330,000 images of 80 different object categories, making it one of the largest and most widely used datasets for object detection tasks.

- Dataset preparations. In our case we need to create a custom dataset class: The annotated dataset needs to be wrapped in a custom PyTorch dataset class. This class should define how to load the images and annotations, apply any necessary transformations (in our case converting the image to a tensor ensures that the image data is normalized and scaled appropriately for use with the neural network), and return the data as PyTorch tensors. For the test dataset we won’t create PyTorch tensors for annotations. Dataset class is stored in ** blood_dataset.py **
- In a pre-trained model we will need to change the number of classes we are predicting for. In our case it is 3 cell-types plus additional class for background detection with index 0. Pre-trained model should be fine-tuned on the dataset using the PyTorch training pipeline. During fine-tuning, the weights of the pre-trained model are adjusted to better fit the new dataset, resulting in a model that can accurately detect objects in the new dataset.
- Save fine-tuned model for further usage with all trained parameters.
- Run PyTorch evaluation step of the model on the prepared test dataset. The output is a list of dictionaries with PyTorch tensors for bounding boxes, labels and confidence scores for each of them per image.
- Define metric for model evaluation and ways to eliminate wrong predictions. mAP will be the main metric we’d like to maximize. For results filtering we will use different levels of score thresholding depending on cell type and also we will use non-maximum suppression function to remove redundant detections of the same object.

Final results:

|Metric |Fully retrained Faster R-CNN||Frozen parameters for backbone CNN part||
|Time to train| 676.31 min||100 min||
Results filtering
Unfiltered results
Thresholds by cell type:
Platelets: 0.55 
RBC: 0.70 
WBC: 0.8
NMS threshold: 0.45
Unfiltered results
Thresholds by cell type:
Platelets: 0.55 
RBC: 0.70 
WBC: 0.75
NMS threshold: 0.45


mAP
0.35
0.84
0.19
0.7
AP ‘RBC’
0.5
0.7
0.34
0.73
AP  'WBC’
0.35
1.0
0.1
0.81
AP  'Platelets' 
0.2
0.78
0.13
0.55


