import streamlit as st
import numpy as np
from PIL import Image
import requests
import urllib.request
import json
import os
import ssl
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml.automl import SearchSpace, ClassificationMultilabelPrimaryMetrics
from azure.ai.ml.sweep import (
    Choice,
    Uniform,
    BanditPolicy,
)
from azure.ai.ml import automl
import base64
from io import BytesIO
import os

st.set_page_config(page_title = "Multilabel Lung Disease Classifier", page_icon = ":robot:") #renames the title of the page in the browser

credential = ClientSecretCredential(
    client_id=os.environ.get("AZURE_CLIENT_ID"),
    client_secret=os.environ.get("AZURE_CLIENT_SECRET"),
    tenant_id=os.environ.get("AZURE_TENANT_ID"),  
)

ml_client = None
try:
    ml_client = MLClient.from_config(credential)
except Exception as ex:
    print(ex)
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    resource_group = os.environ.get("RESOURCE_GROUP")
    workspace = os.environ.get("WORKSPACE")
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)

online_endpoint_name = os.environ.get("ONLINE_ENDPOINT_NAME")
deployment_name = os.environ.get("DEPLOYMENT_NAME")

st.markdown('<h1 style="color:black;text-align: center;">Multilabel Lung Disease Classifier</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;text-align: center;">Azure AutoML Deployed Model</h2>', unsafe_allow_html=True)
st.markdown('<h5 style="color:gray;text-align: center;">by Olga Petan</h5>', unsafe_allow_html=True)

st.markdown(
            """
            <style>
                .intro_para {
                    color: black;
                    font-size: 18px; /* Font size */
                }
            </style>
            """,
            unsafe_allow_html=True
        )
st.markdown("<div class='intro_para'>This app takes an X-ray image of the lungs as input and returns a prediction of a patient's condition. If the patient is healthy, the app does not output any pathology. If the patient is not healthy, the app outputs three useful predictions: 1. how many diseases are detected; 2. which diseases are detected; 3. a visual representation of where the diseases are detected. The model is trained on Azure using AutoML, and is deployed on Azure.</div>", unsafe_allow_html=True)
st.markdown("<div style=margin-bottom: 3px;'></div>", unsafe_allow_html=True)
def save_uploaded_image(uploaded_image, target_path):
    with open(target_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

def read_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()


uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    temp_path = "image.jpg"
    save_uploaded_image(uploaded_image, temp_path)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.image(uploaded_image, caption="Uploaded Image")
    with col3:
        st.write(' ')


    # Define explainability (XAI) parameters
    model_explainability = True
    xai_parameters = {
        "xai_algorithm": "guided_backprop",
        "visualizations": True,
        "attributions": False,
        "confidence_score_threshold_multilabel": 0.4,
    }

    request_json = {
        "input_data": {
            "columns": ["image"],
            "data": [
                json.dumps(
                    {
                        "image_base64": base64.encodebytes(read_image(temp_path)).decode(
                            "utf-8"
                        ),
                        "model_explainability": model_explainability,
                        "xai_parameters": xai_parameters,
                    }
                )
            ],
        }
    }


    request_file_name = "request_data.json"
    with open(request_file_name, "w") as request_file:
        json.dump(request_json, request_file)

    resp = ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name=deployment_name,
    request_file=request_file_name,
    )

    predictions = json.loads(resp)

    def base64_to_img(base64_img_str):
        base64_img = base64_img_str.encode("utf-8")
        decoded_img = base64.b64decode(base64_img)
        #return BytesIO(decoded_img).getvalue()
        return Image.open(BytesIO(decoded_img))
    
    labels = predictions[0]["labels"]
    probs = predictions[0]["probs"]

    filtered_labels = []
    filtered_probs = []

    for label, prob in zip(labels, probs):
        if prob > 0.4: ## this probability should be the same as "confidence_score_threshold_multilabel": 0.4, in xai_parameters 
            filtered_labels.append(label)
            filtered_probs.append(prob)

    # Display the filtered results
    filtered_data = {
        "labels": filtered_labels,
        "probs": filtered_probs
    }

    filtered_probs = [f'{prob * 100:.0f}%' for prob in filtered_probs]
    number_predictions = len(filtered_labels)
    labels = []

    for i, label in enumerate(filtered_labels):
        labels.append(label)

    if "No_Finding" in labels:
        st.markdown(
            """
            <style>
                .healthy {
                    color: #4cc76a;
                    font-size: 32px; /* Font size */
                    text-align: center;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.write("<div class='healthy'>There is no pathology detected on the X-Ray image.</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <style>
                .pathology {
                    color: #b04a43;
                    font-size: 32px; /* Font size */
                    text-align: center;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.write("<div class='pathology'>There is pathology detected on the X-Ray image.</div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

        st.write('<h4 style="color:gray;text-align: center;">What would you like to see next?</h4>', unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        
        # Button labels
        button_labels = ["How Many Diseases Are Detected", "Which Diseases Are Detected", "Where Are The Diseases Detected"]

        # Add CSS styling for equal margins
        st.markdown(
            """
            <style>
            div.stButton > button:first-child {
            background-color: #b04a43; /* Green background color */
            color: white; /* White text color */
            padding: 10px 20px; /* Padding */
            border: 2px solid #b04a43; /* Green border */
            border-radius: 8px; /* Rounded corners */
            cursor: pointer; /* Pointer cursor on hover */
            font-size: 18px; /* Font size */
            }
            div.stButton > button:hover {
            background-color: #db938f; /* Green background color */
            color: white; /* White text color */
            padding: 10px 18px; /* Padding */
            border: 2px solid #db938f; /* Green border */
            border-radius: 8px; /* Rounded corners */
            cursor: pointer; /* Pointer cursor on hover */
            font-size: 18px; /* Font size */
            
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Create a layout with 3 columns
        col1, col2, col3 = st.columns(3)

        # Button click event handling
        button_pressed = [col.button(label) for col, label in zip([col1, col2, col3], button_labels)]

        # Print text based on button clicks
        if button_pressed[0]:
            st.markdown(
            """
            <style>
                .answer_number {
                    color: black;
                    font-size: 24px; /* Font size */
                    text-align: left;
                }
            </style>
            """,
            unsafe_allow_html=True
            )
            st.write(f"<div class='answer_number'>There are {number_predictions} diseases detected.</div>", unsafe_allow_html=True)
        elif button_pressed[1]:
            labels_str = ", ".join(labels)
            st.markdown(
            """
            <style>
                .answer_labels {
                    color: black;
                    font-size: 24px; /* Font size */
                    text-align: center;
                }
            </style>
            """,
            unsafe_allow_html=True
            )
            st.write(f"<div class='answer_labels'>The diseases detected are: {labels_str}</div>", unsafe_allow_html=True)
        elif button_pressed[2]:
            visualizations = predictions[0]["visualizations"]

            num_columns = 2

            # Calculate the number of rows needed based on the number of visualizations and num_columns
            num_visualizations = len(visualizations)
            num_rows = (num_visualizations + num_columns - 1) // num_columns

            # Create a grid layout
            for row in range(num_rows):
                cols = st.columns(num_columns)

            for col_idx in range(num_columns):
                visualization_idx = row * num_columns + col_idx

                if visualization_idx < num_visualizations:
                    img = base64_to_img(visualizations[visualization_idx])
                    width, height = img.size

                    # Crop the left half of the image
                    left_half_image = img.crop((0, 0, width // 2, height))

                    # Calculate the cropping height (5% of the total height)
                    crop_height = int(0.035 * height)

                    # Crop the image from the top
                    cropped_image = left_half_image.crop((0, crop_height, width // 2, height))
                    title_html = f'<div style="text-align: center;">Label: {filtered_labels[visualization_idx]}, Confidence Score:{filtered_probs[visualization_idx]}</div>'
                    cols[col_idx].markdown(title_html, unsafe_allow_html=True)
                    cols[col_idx].image(cropped_image, use_column_width=True) 


