import streamlit as st
import os
import matplotlib.pyplot as plt
from PIL import Image
import random

def run():

    st.title("📊 Exploratory Data Analysis")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Cek 2 kemungkinan lokasi dataset
    train_path = os.path.join(BASE_DIR, "dataset", "train")

    if not os.path.exists(train_path):
        train_path = os.path.join(BASE_DIR, "..", "dataset", "train")

    if not os.path.exists(train_path):
        st.warning("Dataset folder not found.")
        return

    selected_classes = sorted(os.listdir(train_path))

    # IV.1 Distribusi Dataset
    st.header("IV.1 Distribusi Dataset")

    class_counts = {}
    for cls in selected_classes:
        class_dir = os.path.join(train_path, cls)
        if os.path.isdir(class_dir):
            class_counts[cls] = len(os.listdir(class_dir))

    fig1 = plt.figure()
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xticks(rotation=45)
    plt.title("Distribution of Images per Class")
    st.pyplot(fig1)

    st.subheader("Insight")
    st.write("""
    Distribusi antar kelas relatif seimbang sehingga tidak diperlukan
    teknik khusus untuk mengatasi imbalance.
    """)

    # IV.2 Analisis Ukuran Gambar
    st.header("IV.2 Analisis Ukuran Gambar")

    sizes = []

    for cls in selected_classes:
        class_dir = os.path.join(train_path, cls)
        images = os.listdir(class_dir)[:20]

        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path)
                sizes.append(img.size)
            except:
                continue

    st.write("Sample ukuran gambar:")
    st.write(sizes[:10])

    st.subheader("Insight")
    st.write("""
    Sebagian besar gambar memiliki ukuran 224x224 sehingga preprocessing
    menjadi lebih sederhana.
    """)

    # IV.3 Visualisasi Sample
    st.header("IV.3 Visualisasi Sample Gambar per Kelas")

    fig2 = plt.figure(figsize=(10,8))

    for i, cls in enumerate(selected_classes):
        class_dir = os.path.join(train_path, cls)
        images = os.listdir(class_dir)

        if len(images) == 0:
            continue

        img_name = random.choice(images)
        img_path = os.path.join(class_dir, img_name)

        try:
            img = Image.open(img_path)
            plt.subplot(2,3,i+1)
            plt.imshow(img)
            plt.title(cls)
            plt.axis("off")
        except:
            continue

    plt.tight_layout()
    st.pyplot(fig2)

    st.subheader("Insight")
    st.write("""
    Sample gambar menunjukkan variasi visual yang cukup baik untuk training model CNN.
    """)
