# Face Recognition using SVM

This project demonstrates a face recognition system using Support Vector Machines (SVM) and Principal Component Analysis (PCA). The system is trained to recognize faces of three individuals: Barack Obama, Donald Trump, and George W. Bush. The project is divided into three main steps: dataset creation, model training, and testing.

## Usage

### Dataset Creation:
1. Place the images of each individual in separate folders under the `train_data_2` directory.
2. Run the `2.0 Creating the dataset.ipynb` notebook to preprocess the images and create the dataset.

### Model Training:
1. Run the `3.0 Train the SVM.ipynb` notebook to train the SVM model using the preprocessed dataset.
2. The trained model will be saved as `SVM-Face Recognition.sav`.

### Testing:
1. Place the test images in the `test_data` directory.
2. Run the `4.0 Testing the SVM Model.ipynb` notebook to test the model on the new images.

## Dataset

The dataset consists of images of three individuals:

- Barack Obama
- Donald Trump
- George W. Bush

Each individual's images are stored in separate folders under the `train_data_2` directory. The images are preprocessed to extract and resize the face region.

## Model Training

The model training process involves the following steps:

1. **Data Loading**: Load the preprocessed dataset.
2. **PCA for Dimensionality Reduction**: Apply PCA to reduce the dimensionality of the data.
3. **SVM Training**: Train an SVM model with an RBF kernel on the reduced dataset.
4. **Model Evaluation**: Evaluate the model's performance using accuracy, precision, recall, and F1-score.

The trained model is saved as `SVM-Face Recognition.sav`.

## Testing

To test the model:

1. Place the test images in the `test_data` directory.
2. Run the `4.0 Testing the SVM Model.ipynb` notebook.

The notebook will display the test images with the predicted labels and bounding boxes around the detected faces.

## Results

The model achieved an accuracy of 100% on the test set. The classification report and confusion matrix are as follows:

### Classification Report:

               precision    recall  f1-score   support

 Barack Obama       1.00      1.00      1.00        11
 Donald Trump       1.00      1.00      1.00         9
George W Bush       1.00      1.00      1.00         9

    micro avg       1.00      1.00      1.00        29
    macro avg       1.00      1.00      1.00        29
 weighted avg       1.00      1.00      1.00        29

Confusion Matrix:
[[11  0  0]
 [ 0  9  0]
 [ 0  0  9]]

