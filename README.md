# Dogs vs. Cats Image Classifier

This repository contains a machine learning project for classifying images of dogs and cats using various neural network models and classical machine learning models. The dataset used for this project is from the [Kaggle Dogs vs. Cats competition](https://www.kaggle.com/c/dogs-vs-cats).

## Project Structure

- **data/**: Directory containing the dataset.
- **notebooks/**: Jupyter notebooks for data exploration and preprocessing.
- **models/**: Saved trained models.
- **src/**: Source code for training and evaluating models.
- **README.md**: Project documentation.

## Dataset

The dataset for this project can be downloaded from the [Kaggle Dogs vs. Cats competition](https://www.kaggle.com/c/dogs-vs-cats). It contains labeled images of dogs and cats.

## Models

The project includes the following models:

1. **Model 1**: Simple ANN with 1 hidden layer.
2. **Model 2**: ANN with 3 hidden layers.
3. **Model 3**: ANN with 2 hidden layers.
4. **Model 4**: Single-layer ANN.
5. **Model 5**: Convolutional Neural Network (CNN).
6. **Model 6**: VGG16-based feature extractor with a dense network.
7. **Model 7**: SVM (Support Vector Machine).
8. **Model 8**: Random Forest Classifier.
9. **Model 9**: Decision Tree Classifier.
10. **Model 10**: Naive Bayes Classifier.

Each model is trained and evaluated to determine the best performing one.

## Training

The models are trained on the dataset using the following steps:

1. **Data Preprocessing**:
    - Load and preprocess images.
    - Convert images to grayscale and resize.
    - Split the dataset into training and testing sets.
    - Reshape and scale the data as required by each model.
    - Use VGG16 for feature extraction.

2. **Model Training**:
    - Train each model using the training data.
    - Evaluate the model on the validation set.
    - Save the trained models.

3. **Model Selection**:
    - Compare the accuracy of all models.
    - Select the model with the highest accuracy for predictions.

## Feature Extraction using VGG16

We use the VGG16 model pre-trained on ImageNet to extract features from the images. These features are then used to train additional models.

```python
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16

# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=True)

# Create a new model to extract features from the last fully connected layer
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def Extract_features(img_arr):
    feature = model.predict(img_arr, verbose=0)
    return feature
