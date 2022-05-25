import streamlit as st
from PIL import Image

import style

st.title("PyTorch Style Transfer")

img = st.sidebar.selectbox(
    "Select an image to stylize",
    ('amber.jpg','plane.png')
)

style_name = st.sidebar.selectbox(
    "Select a style",
    ('candy', 'mosaic', 'rain_princess', 'udnie')
)

model = "saved_models/{}.pth".format(style_name)
input_image = "images/content-images/{}".format(img)
output_image = "images/output-images/{}-{}".format(style_name, img)

st.write("### Source Image:")
image = Image.open(input_image)
st.image(image, width = 400)

clicked = st.button("Stylize")

if clicked:
    model = style.load_model(model)
    style.stylize(model, input_image, output_image)
    st.write("### Output Image:")
    image = Image.open(output_image)
    st.image(image, width = 400)
