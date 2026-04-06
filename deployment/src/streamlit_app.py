import streamlit as st

# 🔥 HARUS PALING ATAS
st.set_page_config(
    page_title="Sports images prediction",
    layout="wide"
)

import eda
import prediction

page = st.sidebar.selectbox(
    "Choose page",
    ("EDA", "Prediction")
)

if page == "EDA":
    eda.run()
else:
    prediction.run()
