import os
import shutil
import streamlit as st
from ultralytics import YOLO
import cv2
import yaml
from streamlit_image_select import image_select

# Load the YOLO model
model = YOLO("yolov8m.pt")

st.title("Demo Image Detection App")


# Function to process an image
def process_image(image_path):
    # Prediction
    model.predict(image_path, save=True, save_txt=True)

    # Path of the saved image
    saved_dir = "runs/detect/predict"
    saved_image_path = os.path.join(saved_dir, os.path.basename(image_path))

    # The saved image
    predicted_img = cv2.imread(saved_image_path)
    predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2RGB)

    # The COCO class names from the yaml file
    with open("coco8.yaml", "r", encoding='utf-8') as stream:
        names = yaml.safe_load(stream)["names"]

    # The labels file and detected object names as tags
    labels_path = os.path.join(
        saved_dir, "labels", f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")
    tags = []
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding='utf-8') as file:
            lis = file.readlines()
        detected_objects = {names[int(l.split()[0])] for l in lis}

        # Colors for tags
        colors = [
            "#FF5733", "#33FF57", "#3357FF", "#F333FF",
            "#33FFF5", "#FF33F5", "#D11F13", "#F533FF"
        ]

        # List of spans
        for i, obj in enumerate(detected_objects):
            color = colors[i % len(colors)]
            tags.append(
                f'<span style="background-color: {color}; color: white; padding: 0.5em; border-radius: 0.5em; margin-right: 0.5em;">{obj}</span>'
            )

    # Remove the whole "runs" folder after displaying them
    if os.path.exists(saved_dir):
        shutil.rmtree(saved_dir)

    return predicted_img, tags


# Image selection
img = image_select(
    label="You can choose one of these images",
    images=["default_images/amesterdam.jpg",
            "default_images/dog.jpg", "default_images/barcelona.jpg"]
)

# Initialize variables
image_to_process = None

# Upload an image
uploaded_file = st.file_uploader(
    "Or Choose an image from your laptop",
    type=["jpg", "jpeg", "png"]
)

# Check if the user has uploaded an image
if uploaded_file and not os.path.exists(f'uploaded_images/{uploaded_file.name}'):
    # # Remove all files inside the uploaded_images folder
    if os.path.exists('uploaded_images'):
        for filename in os.listdir('uploaded_images'):
            file_path = os.path.join('uploaded_images', filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    # Save the uploaded file temporarily
    with open(f'uploaded_images/{uploaded_file.name}', "wb") as f:
        f.write(uploaded_file.getbuffer())

    image_to_process = f'uploaded_images/{uploaded_file.name}'

else:
    # Use the selected default image
    image_to_process = img

# Process the selected image
if image_to_process:
    original_image = cv2.imread(image_to_process)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    predicted_image, tags = process_image(image_to_process)

    # Display the original and predicted images
    st.header("Original Image")
    st.image(original_image, use_column_width=True)

    st.header("Predicted Image")
    # Concatenate spans and display them in a single line
    spans_line = " ".join(tags)
    st.markdown(spans_line, unsafe_allow_html=True)
    st.image(predicted_image, use_column_width=True)
