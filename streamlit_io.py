import streamlit as st
from fastai.vision.all import *


st.title("Airplane Identifier")
st.text("Built by, Russell Susanto.")


def extract_plane_name(file_name):
    p = str(file_name)
    # print(p)
    plane_name_parts = p.split('/')
    print(plane_name_parts)
    final_plane_name = plane_name_parts[-2]
    return final_plane_name



breed_model = load_learner("plane_model_fastai284.pkl")


def predict(uploaded_file):
    real_img = PILImage.create(uploaded_file)
    resized_img = real_img.resize((28, 28), Image.NEAREST)
    pred_class, pred_idx, outputs = breed_model.predict(resized_img)
    return pred_class

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    prediction = predict(uploaded_file)
    st.subheader(f"Predicted Breed: {prediction}")

st.text("Built with Streamlit and FastAI, Made by russell.")