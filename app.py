"""Minimal Persian Streamlit demo for the trained model."""
from pathlib import Path

import streamlit as st
from PIL import Image

from predict import load_classifier, predict_pil

st.set_page_config(page_title="تشخیص گربه و سگ با هوش مصنوعی", page_icon="🐾", layout="centered")
st.title("🐾 تشخیص گربه یا سگ")
st.write("یک عکس بارگذاری کن تا مدل آموزش‌دیده پیش‌بینی کند تصویر گربه است یا سگ.")
checkpoint_path = st.sidebar.text_input("مسیر مدل", "outputs/best_model.pth")

@st.cache_resource
def get_model(path: str):
    return load_classifier(path)

uploaded = st.file_uploader("عکس را انتخاب کن", type=["jpg", "jpeg", "png", "webp", "bmp"])
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="تصویر ورودی", use_container_width=True)
    if not Path(checkpoint_path).exists():
        st.error(f"مدل پیدا نشد: {checkpoint_path}\nاول train.py را اجرا کن.")
    else:
        try:
            with st.spinner("در حال بررسی تصویر..."):
                model, transform, device = get_model(checkpoint_path)
                result = predict_pil(image, model, transform, device)
            st.subheader(f"نتیجه: {result['label_fa']}")
            st.metric("اطمینان مدل", f"{result['confidence']:.1%}")
            st.write("احتمال هر کلاس")
            st.progress(result["cat_probability"], text=f"گربه: {result['cat_probability']:.1%}")
            st.progress(result["dog_probability"], text=f"سگ: {result['dog_probability']:.1%}")
            st.caption("درصد اطمینان تضمین درستی نیست؛ عکس‌های مبهم ممکن است اشتباه تشخیص داده شوند.")
        except Exception as error:
            st.error(f"خطا در بارگذاری یا اجرای مدل: {error}")
else:
    st.info("برای شروع، یک عکس گربه یا سگ انتخاب کن.")
