# wunderpen_defect_classification
HandwrittenTextClassifier is a powerful tool designed to write and classify handwritten text on various materials such as envelopes, letter pads, and more. This algorithm classifies the handwritten machine-generated images. 

# data_processing.py:

## Image Processing and Labeling Script

This script automates extracting class information from image filenames, copying images to a new directory, and creating a CSV file with labels for each image.

### How the Code Works

1. **Directory Setup**: The script checks if the target directory exists. If it does not, it creates one.

2. **Image Processing**: It walks through the source directory, identifies images that match specific criteria (filename starts with 'written' and ends with '.png'), and copies them to the target directory with a new naming format that includes a unique image ID.

3. **Class Extraction from Filenames**:
   - Uses regular expressions to find patterns in the filenames that indicate the image class.
   - Converts these patterns into indices that represent classes.

4. **Label Vector Creation**:
   - For each image, it creates a binary vector indicating the presence of classes, where each index in the vector corresponds to a class. The vector is filled with '1's at positions indicated by the class indices extracted from the filename.

5. **CSV File Generation**:
   - A CSV file is generated that lists each image's new filename along with its label vector.
   - The script prints the location of the CSV file and the total number of images processed.

### Setup and Execution

- Set the `source_directory` and `target_directory` paths according to your file structure.
- Specify the `csv_filename` for the output CSV file.
- Run the script to process the images and generate the CSV.

# training.py
## Image Classification with Transformers

This Python script demonstrates a deep learning approach for image classification using the Vision Transformer (ViT) model from the `transformers` library. It includes data preprocessing, model training, evaluation, and early stopping to prevent overfitting.

### How the Code Works

1. **Dataset Preparation**:
   - `CustomDataset`: A custom dataset class that reads image paths and labels from a CSV file, loads the images, and applies transformations.
   - The dataset is split into training and evaluation subsets.

2. **Model Setup**:
   - Loads a pre-trained Vision Transformer (ViT) model specifically designed for image classification.
   - The model is adapted to predict 12 different classes.

3. **Training Process**:
   - Utilizes the AdamW optimizer with a linear learning rate scheduler.
   - Implements a training loop that includes loss calculation and optimizer updates.
   - Evaluates the model on the validation dataset after each epoch and calculates metrics such as precision, recall, F1 score, and accuracy.

4. **Early Stopping**:
   - Monitors validation loss and stops training if there's no improvement over a set number of epochs to prevent overfitting.

5. **Model Saving**:
   - Saves the trained model if early stopping is not triggered.

# test.py 

## Image Prediction with Vision Transformer (ViT)

This Python script demonstrates how to load a pre-trained Vision Transformer (ViT) model, process an image, and predict the classes present in the image using PyTorch and the transformers library.

### How the Code Works

1. **Model Loading**:
   - Loads a pre-trained Vision Transformer (ViT) model from a specified path.
   - Sets the model to evaluation mode and transfers it to an appropriate device (GPU if available, otherwise CPU).

2. **Image Processing**:
   - Opens an image from a specified path and converts it to RGB.
   - Applies a series of transformations (resize, tensor conversion, and normalization) to prepare the image for the model.

3. **Prediction**:
   - Passes the processed image through the model to obtain logits.
   - Applies the sigmoid function to convert logits to probabilities.
   - Identifies classes with probabilities above a certain threshold and prints them.



# gradio_test.py

This Python application is designed to demonstrate image classification using a Vision Transformer (ViT) model. The application provides a user interface created with Gradio, allowing users to either upload images or navigate through a predefined set of test and support images for classification.

## Features

- **Model Loading**: Load a pretrained Vision Transformer model for image classification.
- **Image Processing**: Transform images to the required input format for the model.
- **Prediction**: Classify images and display the predicted class along with confidence levels.
- **Navigation**: Browse through a set of predefined test and support images.
- **Image Upload**: Allow users to upload images for classification.

## Prerequisites

Before running this application, ensure you have the following installed:
- Python 3.8+
- PyTorch
- Transformers
- Gradio
- PIL


#Output

![Screenshot from 2024-07-27 15-59-29](https://github.com/user-attachments/assets/c244fb5b-a6dd-4b87-b759-a95c2b165bda)
![Screenshot from 2024-07-27 15-59-18](https://github.com/user-attachments/assets/f050e12d-bee8-44ea-9caa-30df99368431)

