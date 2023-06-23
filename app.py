import streamlit as st
import cv2
import numpy as np
import pickle

# Load the trained model
model_file_path = 'model.sav'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

# Define the classes dictionary
classes = {0: 'No Tumor', 1: 'Positive Tumor'}

# Function to make predictions


def predict(image):
    # Convert image to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize image to a fixed size
    img = cv2.resize(img, (200, 200))
    # Reshape image to a 1D array and normalize pixel values
    img = img.reshape(1, -1) / 255
    # Make prediction using the model
    prediction = model.predict(img)
    # Return the predicted class
    return classes[prediction[0]]


# Streamlit app
def main():
    st.title("Brain Tumor Prediction")
    st.write(
        "Upload an image and the app will predict whether it contains a brain tumor or not.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV-compatible format.

        # The file is first converted into a byte array and then decoded using OpenCV's imdecode function.

        # uploaded_file: file object, obtained from the file uploader.
        # uploaded_file.read(): Reads the contents of the file as bytes.
        # bytearray(): creates a mutable array of bytes from the file content.
        # np.asarray(): converts the bytearray into a NumPy array.
        # The dtype=np.uint8: specifies that each element in the array should be of type unsigned 8-bit integer.
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)

        # cv2.imdecode(): function from the OpenCV library that decodes an image from a buffer. It takes two arguments: the first one is the buffer (in this case, the file_bytes array), and the second one is a flag indicating how to decode the image. The flag 1 (cv2.IMREAD_COLOR) specifies that the image should be loaded in the BGR color format with 3 color channels (Blue, Green, Red).
        image = cv2.imdecode(file_bytes, 1)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        prediction = predict(image)
        st.write("Prediction:", prediction)


# Run the app
if __name__ == '__main__':
    main()
