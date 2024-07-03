# wunderpen_defect_classification
HandwrittenTextClassifier is a powerful tool designed to write and classify handwritten text on various materials such as envelopes, letter pads, and more. This algorithm classifies the handwritten machine-generated images. 

# data_processing.py working:

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





