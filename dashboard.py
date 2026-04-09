import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt

from src.gradcam import get_gradcam_heatmap

model = load_model("models/brain_tumor_model.keras")

IMG_SIZE = 224

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Detection Dashboard")
st.write("Upload an MRI scan to detect tumor presence")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)

        img = np.array(image)

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

        img_input = preprocess_input(img_resized)
        img_input = np.expand_dims(img_input, axis=0)

        prediction = float(model.predict(img_input)[0][0])

        tumor_prob = float(1 - prediction)
        no_tumor_prob = float(prediction)

        st.write(f"🔍 Raw Prediction Value: {prediction:.4f}")

        threshold = 0.6

        st.subheader("🧪 Prediction Result")

        if prediction >= threshold:
            st.success(f"✅ No Tumor ({no_tumor_prob*100:.2f}% confidence)")
        else:
            st.error(f"⚠ Tumor Detected ({tumor_prob*100:.2f}% confidence)")

        st.subheader("📊 Probability Analysis")

        labels = ["Tumor", "No Tumor"]
        values = [tumor_prob, no_tumor_prob]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")

        st.pyplot(fig)

        st.subheader("🧠 Model Attention (Grad-CAM)")

        try:
            heatmap = get_gradcam_heatmap(model, img_input)

            heatmap = np.maximum(heatmap, 0)
            heatmap = heatmap / (heatmap.max() + 1e-8)

            heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

            original_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            heatmap_color = cv2.applyColorMap(
                np.uint8(255 * heatmap), cv2.COLORMAP_JET
            )

            cam_image = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)

            st.image(
                cam_image,
                caption="Grad-CAM Visualization",
                use_container_width=True
            )

        except Exception as e:
            st.warning("⚠ Grad-CAM failed")
            st.text(str(e))

        st.subheader("📋 Clinical Insight")

        if prediction < threshold:
            st.markdown("""
🔴 **Possible abnormal mass detected**

- Suggest MRI contrast scan  
- Immediate radiologist consultation recommended  
- Further diagnostic evaluation required  
""")
        else:
            st.markdown("""
🟢 **No visible tumor patterns detected**

- Brain structure appears normal  
- Routine monitoring recommended  
- No immediate clinical concern  
""")

    except Exception as e:
        st.error("❌ Error processing image")
        st.text(str(e))