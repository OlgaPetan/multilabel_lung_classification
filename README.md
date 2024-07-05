# Multilabel Lung Classification

__Azure AutoML__ __multilabel classification__ model on lung X-ray images

The labels for the images in the healthy and pathology folders are in the labels.csv files. The pathology images can have 1 or multiple diseases. 
All images are taken from this repo: https://nihcc.app.box.com/v/ChestXray-NIHCC

The model is trained on Azure using AutoML to do hyperparameter tuning and test different models and parameters in order to find the best model. The model is
deployed in Azure as well.

The __Streamlit app__ that hosts the model: __https://multilabel-lung-classification-azure.streamlit.app/__
