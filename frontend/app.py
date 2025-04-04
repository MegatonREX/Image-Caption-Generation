import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from PIL import Image

# Load the trained model and tokenizer
model = tf.keras.models.load_model("best_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load VGG16 model for feature extraction
vgg_model = VGG16(weights="imagenet")
vgg_model = tf.keras.Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)

# Define max_length based on training data
max_length = 34  # Change as per your training data

def extract_features(image_path):
    """Extract features from an image using VGG16."""
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = vgg_model.predict(image, verbose=0)
    
    return features

def idx_to_word(integer, tokenizer):
    """Convert an integer back to a word using tokenizer."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image_features, tokenizer, max_length):
    """Generate a caption for the given image."""
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=31)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    return in_text.replace("startseq", "").replace("endseq", "").strip()

# Streamlit UI
st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Save the uploaded file temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features and generate caption
    image_features = extract_features("temp_image.jpg")
    caption = predict_caption(model, image_features, tokenizer, max_length)

    # Display the generated caption
    st.subheader("Generated Caption:")
    st.write(caption)
