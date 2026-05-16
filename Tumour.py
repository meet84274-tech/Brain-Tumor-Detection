import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import os
from PIL import Image
from streamlit_option_menu import option_menu
import joblib
from gtts import gTTS

import pygame
import sklearn
import matplotlib.pyplot as plt
from scipy import stats
import pygame

def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

import streamlit as  st
st.title("🎗️Brain Tumour Predictor🎗️")

st.markdown("### This is a web application that predicts whether a brain MRI scan has a tumour or not using deep learning models.")
tumor_model=tf.keras.models.load_model('Brain_tumor_main.h5')

labels=['No Tumor','Tumor']
uploaded_file=st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

def validate_mri_image(img_arr_original, prob):
    """
    Validate if the uploaded image is likely a brain MRI scan.
    Works with both grayscale and colored images.
    Returns (is_valid, message)
    """
    # Convert to grayscale for analysis
    img_gray = np.mean(img_arr_original, axis=-1)
    
    # Check 1: Brain structure detection using edge detection
    # Brain MRI scans have specific edge patterns - they show brain tissue boundaries
    edges = cv2.Canny((img_gray * 255).astype(np.uint8), 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Brain MRI typically has moderate edge density (not too many, not too few)
    # Natural images (cats, people) have very different edge patterns
    if edge_density < 0.02 or edge_density > 0.25:
        return False, "❌ **Image Validation Failed**: This doesn't appear to be a Brain MRI scan. The image structure doesn't match brain tissue patterns."
    
    # Check 2: Circular/Oval shape detection for brain outline
    # Brain MRI scans show a roughly circular/oval brain shape
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        hull_area = cv2.contourArea(cv2.convexHull(largest_contour))
        
        # Circularity check: brain should have high solidity
        if hull_area > 0:
            solidity = contour_area / hull_area
            if solidity < 0.6:  # Brain should have compact shape
                return False, "❌ **Image Validation Failed**: Image shape doesn't match brain MRI characteristics."
    
    # Check 3: Model confidence threshold
    # If the model is highly uncertain, it's likely not a brain image
    if 0.35 < prob < 0.65:
        return False, "❌ **Image Validation Failed**: Model prediction is uncertain. This may not be a Brain MRI scan."
    
    # Check 4: Histogram analysis - MRI has specific intensity distribution
    histogram = np.histogram(img_gray.flatten(), bins=256)[0]
    histogram = histogram / histogram.sum()
    
    # Check if histogram is too uniform (blank/solid image)
    max_bin_ratio = np.max(histogram) / (np.sum(histogram) + 1e-6)
    if max_bin_ratio > 0.3:
        return False, "❌ **Image Validation Failed**: Image appears too uniform. Please upload a Brain MRI scan."
    
    # Check 5: Variance check - MRI images have good internal variation
    variance = np.var(img_gray)
    if variance < 0.005:  # Too uniform, not a brain image
        return False, "❌ **Image Validation Failed**: Image lacks intensity variation typical of Brain MRI."
    
    if variance > 0.25:  # Extremely high variance suggests natural image
        return False, "❌ **Image Validation Failed**: Image variance too high. This doesn't appear to be a Brain MRI scan."
    
    return True, "✅ Valid Brain MRI Image"

if uploaded_file is not None:
    # Load and preprocess image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_arr_original = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr_original, axis=0)

    # Predict
    preds = tumor_model.predict(img_arr)
    # ✅ CORRECT sigmoid-based logic
    prob = preds[0][0]   # probability of "yes"

    if prob >= 0.5:
        label = "yes"
        confidence = prob
    else:
        label = "no"
        confidence = 1 - prob

    # Validate if image is actually a brain MRI
    is_valid, validation_msg = validate_mri_image(img_arr_original, prob)
    
    if not is_valid:
        st.error(validation_msg)
        st.warning("🔍 Tip: Make sure you're uploading a Brain MRI scan image, not other images like cats, persons, objects, etc.")
    else:
        # Output
        st.success(validation_msg)
        st.write(f"**Prediction:** {label.upper()}")
        st.metric("Confidence", f"{confidence:.2%}")

        # Show image
        plt.figure(figsize=(6, 6))
        plt.imshow(img_arr_original)
        plt.axis("off")
        plt.title(f"Prediction: {label} ({confidence:.2%})")
        st.pyplot(plt)
    
    






