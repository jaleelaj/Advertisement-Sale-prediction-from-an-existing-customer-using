# Advertisement Sale Prediction from Existing Customer

This repository contains a Python script for predicting advertisement sales from an existing customer using machine learning. The prediction model is trained on historical data and uses the scikit-learn library for machine learning. The model is saved using pickle, and you can deploy it to make real-time predictions.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Model Deployment](#model-deployment)

## Prerequisites

Before you can use the provided script, you'll need to have the following installed:

- Python (3.6 or higher)
- Required Python libraries (numpy, pandas, scikit-learn, matplotlib, seaborn)

You can install the necessary libraries using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage

Follow these steps to use the script:

1. Clone or download this repository to your local machine.

2. Place your dataset file (CSV format) in the same directory as the script and update the file name in the script. The dataset should contain information about existing customers and their advertisement sales.

3. Open a terminal or command prompt and navigate to the repository's directory.

4. Run the script:

   ```bash
   python advertisement_sale_prediction.py
   ```

5. The script will load your dataset, preprocess it, train a Random Forest Classifier model, make predictions, and display the accuracy of the model along with a confusion matrix.

6. The trained model will be saved to a file named `model.pkl` in the same directory.

## Dataset

The dataset used for this prediction task should be in CSV format, with features related to customers and their advertisement sales. The script assumes that the target variable is the last column in the dataset. Make sure to clean the data and remove any duplicates.

## Model Training

The script trains a Random Forest Classifier model on the dataset. You can modify the script to use a different machine learning algorithm if desired. Ensure that you split your data into a training set and a test set for model evaluation.

## Model Deployment

To deploy the trained model for real-time predictions, you can use a web framework like Flask. The script does not cover deployment, but it does save the trained model to a file (`model.pkl`) that you can load in your deployment code. Make sure to preprocess input data in the deployment code, including scaling if required.

Feel free to adapt and extend this script to suit your specific requirements and deployment needs.
