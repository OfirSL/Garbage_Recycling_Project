# EcoVision - garbage recycling project
## Description
EcoVision was developed as part of the Israel Tech Challenege (ITC) Data Science program, final project.  
It is an AI-powered application designed to assist users in detecting and classifying various types of garbage items while providing information on proper disposal methods based on the Israeli waste system.  
The app utilizes the YOLO8n detection and classification computer vision model to accurately identify different types of garbage items.

*The primary objectives of EcoVision are:*

- Enable users to identify and categorize different types of garbage items.
- Provide users with information on the appropriate methods of disposing of each type of garbage.
- Raise awareness about waste management and encourage environmentally-friendly practices.

![alt text](https://github.com/OfirSL/Garbage_Recycling_Project/blob/b027c50c08c23a7e2b566c11022a50fc1cd146b2/res/Augmented%20model%20predictions.png)

## Streamlit Application
An overview of the EcoVision trained models is provided by the following Streamlit application:  
https://ofirsl-trash-sorting-project-yolo-streamlit-j4hy7v.streamlit.app/

## Dataset
The original dataset consists of thousands of labeled images representing various types of waste materials from different sources.
For the purpose of EcoVision and to align with the Israeli waste system, the dataset underwent certain modifications.
The combined dataset was relabeled to ensure consistency with the disposal bins and methods in Israel.

This annotated dataset served as the foundation for training two versions of the YOLO8n models:

**Original Model**: This model was trained using the dataset without any data augmentation. The images were directly used as they were labeled in the dataset.

**Augmented Model**: To enhance the diversity and robustness of the training data, the dataset underwent augmentation. The augmentation techniques applied were as follows:

- *Replacing Background*: Images with a white or single-colored background were replaced with randomly selected colorful background images. This helped introduce variations in the background and improve the model's ability to generalize to different environments.

- *Adding Noise*: Various types of noise were added randomly to the images during augmentation. This included Gaussian noise, jitter, brightness variations, and salt & pepper noise. By introducing such noise, the model became more resilient to noise interference in real-world scenarios.  
![alt text](https://github.com/OfirSL/Garbage_Recycling_Project/blob/b027c50c08c23a7e2b566c11022a50fc1cd146b2/res/augmentations_examples/g.png)

\* More data augmentation examples can be found in the streamlit app (see above).

## Acknowledgements
I would like to acknowledge Roboflow Public Datasets for providing the datasets used in the EcoVision project.  
The following datasets from Roboflow Public Datasets were used\*:
* https://universe.roboflow.com/material-identification/garbage-classification-3/dataset/1
* https://universe.roboflow.com/nhat/medicine-zknkf/dataset/2

\* partial list

