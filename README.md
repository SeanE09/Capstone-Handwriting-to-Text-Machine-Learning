# Handwriting to Text - Machine Learning

<!-- ![Image Description](DisplayIMG/Cartoon.png) -->

<div align="center">
  <img src="DisplayIMG/Cartoon.png" alt="Image Description" />
</div>

<!--  ![Image Description](DisplayIMG/HW.png) -->

### Contributors
* Sean Evans <https://www.linkedin.com/in/sean-evans/>

## Business Understanding
We were tasked with building a Neural Network model for the purposes of recognizing handwritten letters/number and converting them to text. 

## Business Applications
1. The post office - helping to identify the handwritten addresses on mail items
2. Converting historical medical records over to text files for record keeping and future meta-analysis
3. Historical Censes and government records for record keeping and future meta-analysis
4. Handwriting software for taken handwritten notes and converting them to a text file of notes

# Training Data
https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset

- Image LocationL Img/.png files (3410 files)
  - Images Details: img00#-00#.png, Width: 1200px, Height: 900px
- Labels File: labels/english.csv (71.62 kB)

# Explain Helper Function and reasoning

- Loading Function: This function takes the folder containing the images and the file with each images corresponding 'target' label. It loads the images from the location, converts them to grayscale, resizes them to a size of 64x64 pixels, and appends them to the images list with their label. Returning the images and labels lists. 
  - Uses LabelEncoder to convert categorical labels into numerical labels.
  - (images - mean) / std: Normalizes the pixel values of the images array by subtracting the mean and dividing by the standard deviation. 
  * This normalization step is commonly performed in machine learning to ensure that the input data has a similar scale, which can improve the performance of models.
  - encoded_labels = to_categorical(numerical_labels): This line applies the to_categorical function to the numerical_labels array. This converts the numerical labels into a one-hot encoded representation, which is a binary vector where each element corresponds to a unique label and has a value of 1 if the sample belongs to that class, and 0 otherwise.

- The code loads images from a specified folder, preprocesses them by converting to grayscale, resizing, and normalizing the pixel values. It also encodes the labels into numerical and one-hot encoded representations, preparing the data for further machine learning.

# Explain Data augmentation

- train_test_split: Randomly splits data leaving 20% for later testing. random_state set to 42 for reproducibility

- ImageDataGenerator from the Keras library

  - rotation_range=35: Rotating images randomly by 35% degrees. Limiting at 35% degrees so that a number or letter does not end up looking like a different number/letter. For example, an upside down 6 is a 9.
  - width_shift_range=0.1:  Shifts images horizontally by 10% of the width
  - height_shift_range=0.1: Shifts images vertically by 10% of the height
  - shear_range=0.0: Plan to look into shear more as some can replicate handwritting styles but too much could create something outside normal handwriting styles.
  - zoom_range=0.1: Applying a zoom transformation with a zoom range of 10%
  - horizontal_flip=False: Passed on applying horizontal flip for reasons mentionend in 'rotation_range'
  - vertical_flip=False: Passed on applying horizontal flip for reasons mentionend in 'rotation_range'


# Speak to model (show image of model structure)






# Python and tensorflow versions
