import streamlit as st
import numpy as np
from PIL import Image

from instafilter import Instafilter

#st.set_option("deprecation.showfileUploaderEncoding", False)

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

url = "https://github.com/thoppe/instafilter"
st.markdown("# [Instafilter]({url}) demo")

l_col , r_col = st.beta_columns(2)

with l_col:
    model_name = st.selectbox(
        "Choose a filter",
        sorted(Instafilter.get_models()),
        index=20,
    )
model = Instafilter(model_name, device = 'cpu')
with r_col:
    raw_image_bytes = st.file_uploader("Choose an image...")

if raw_image_bytes is not None:

    img0 = np.array(Image.open(raw_image_bytes))

    with st.spinner(text="Applying filter..."):
        # Apply the model, convert to BGR first and after
        img1 = model(img0[:, :, ::-1], is_RGB=False)[:, :, ::-1]

    l_col, r_col = st.beta_columns(2)
    with l_col:
        st.image(img1, caption = f"{model_name} filter")
    with r_col:
        st.image(img0, caption = "Original")

