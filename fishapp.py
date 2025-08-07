import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import pandas as pd
import time

# --- CONFIGURATION ---
MODEL_FILENAME = 'model2_transferlearning.keras'  # or model1_customcnn.keras as needed
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
IMG_HEIGHT, IMG_WIDTH = 224, 224  # As per your training

# --- Correct Class Names from your printout ---
class_names = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

st.set_page_config(page_title="Fish Classifier", layout="centered")
st.title("🐟 Fish Species Classifier")
st.write("Upload a fish image and the AI model will predict its species!")

# --- Load Model ---
@st.cache_resource(show_spinner="Loading model, please wait...")
def load_my_model():
    return load_model(MODEL_PATH)
model = load_my_model()

# --- Image Upload Section ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded fish image", use_column_width=True)
    img_proc = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img_proc) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Progress bar for user experience
    my_bar = st.progress(0)
    for pct in range(0, 70, 10):
        time.sleep(0.05)
        my_bar.progress(pct+10)
    predictions = model.predict(img_array)
    time.sleep(0.2)
    my_bar.progress(100)
    st.success("Prediction complete!")

    pred_idx = int(np.argmax(predictions[0]))
    pred_prob = float(np.max(predictions[0]))
    st.balloons()

    st.markdown(f"### 🏷️ **Predicted Species:** {class_names[pred_idx]}")
    st.markdown(f"**Confidence:** {pred_prob:.2%}")

    # Show top 3 probable classes
    prob_dict = {name: float(predictions[0][i]) for i, name in enumerate(class_names)}
    top3 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    chart_df = pd.DataFrame(top3, columns=["Species", "Probability"])
    st.bar_chart(chart_df.set_index("Species"))

    with st.expander("See all class probabilities"):
        st.dataframe(
            pd.DataFrame(list(prob_dict.items()), columns=["Class", "Probability"])
            .sort_values("Probability", ascending=False), use_container_width=True
        )
else:
    st.info("Upload an image to get a prediction!")

st.caption("Project by Dhruv Tamirisa | Fish Classification Capstone")

with st.expander("About this app"):
    st.write("""
    - **Model:** VGG16 Transfer Learning (.keras)
    - **Framework:** TensorFlow/Keras
    - Preprocessing: Resize to 224x224, scale [0,1]
    """)
