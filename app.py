import tensorflow as tf
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import keras

IMAGE_SHAPE = (200, 200)
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'del', 'nothing', 'space']

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Sign Language Recognition")

# Load the model
def load_model():
    return keras.layers.TFSMLayer(r'C:\Users\PIYUSH GUPTA\Documents\GitHub\Sign-Language-Recognition\ASL-Model-1', call_endpoint='serving_default')

# Function to load and preprocess image
def load_and_prep_image(image):
    # Decode image
    image = tf.image.decode_image(image, channels=3)
    
    # Resize while preserving aspect ratio
    image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
    target_height, target_width = IMAGE_SHAPE
    
    # Calculate aspect ratio and resize dimensions
    aspect_ratio = tf.cast(image_width, tf.float32) / tf.cast(image_height, tf.float32)
    target_aspect_ratio = tf.cast(target_width, tf.float32) / tf.cast(target_height, tf.float32)
    
    resize_factor = tf.cond(aspect_ratio < target_aspect_ratio,
                            lambda: target_width / tf.cast(image_width, tf.float32),
                            lambda: target_height / tf.cast(image_height, tf.float32))
    
    new_height = tf.cast(tf.cast(image_height, tf.float32) * resize_factor, tf.int32)
    new_width = tf.cast(tf.cast(image_width, tf.float32) * resize_factor, tf.int32)
    
    image = tf.image.resize(image, [new_height, new_width])
    
    # Center crop to target size
    image = tf.image.central_crop(image, central_fraction=IMAGE_SHAPE[0] / new_height)
    
    # Resize to model input shape
    image = tf.image.resize(image, size=IMAGE_SHAPE)
    
    return image

# Function to predict using the loaded model
def predict_with_model(img):
    model = load_model()
    prediction = model(tf.expand_dims(img, axis=0))
    output_tensor = prediction['output_layer']
    predicted_index = int(tf.argmax(tf.squeeze(output_tensor)).numpy())
    predicted_class = classes[predicted_index]
    return predicted_class

# Function for URL uploader
def url_uploader():
    st.text("Provide URL for Sign Recognition")
    path = st.text_input("Enter image URL to classify...", "https://i.ibb.co/4Z3CTzY/H-test.jpg")
    if path is not None:
        content = requests.get(path).content
        st.write("Predicted Sign:")
        with st.spinner("Classifying....."):
            img = load_and_prep_image(content)
            predicted_class = predict_with_model(img)
            st.write(predicted_class)
        st.write("")
        image = Image.open(BytesIO(content))
        st.image(image, caption="Classifying the Sign", use_column_width=True)

# Main function
def main():
    url_uploader()

if __name__ == "__main__":
    main()
