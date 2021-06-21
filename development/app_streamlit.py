import base64
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO

from instafilter import Instafilter

#st.set_option("deprecation.showfileUploaderEncoding", False)
def get_image_download_link(pil_im, str_msg = 'Download result', str_format = 'JPEG'):
	"""
	Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = BytesIO()
	pil_im.save(buffered, format= str_format)
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/jpg;base64,{img_str}">{str_msg}</a>'
	return href

def Main():
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

            if st.checkbox('Download Image'):
                st.markdown(get_image_download_link(Image.fromarray(img1), str_msg = 'Click To Download Image'), 
                    unsafe_allow_html = True)
        with r_col:
            st.image(img0, caption = "Original")

if __name__ == '__main__':
	Main()
