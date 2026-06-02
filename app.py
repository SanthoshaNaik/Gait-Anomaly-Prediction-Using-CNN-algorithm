# import streamlit as st
# import numpy as np
# import tensorflow as tf
# import cv2
# import os
# from tensorflow.keras.models import load_model
# from PIL import Image
# import tempfile

# # Load model
# model = load_model("gait_anomaly_model21.h5")
# gait_types = ["Antalgic Gait", "Parkinson Gait", "Spastic Gait", "Scissor Gait"]

# # Preprocess image
# def preprocess_image(img):
#     img = cv2.resize(img, (224, 224))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)
#     return img

# # Predict gait
# def predict_gait(model, input_data):
#     predictions = model.predict(input_data)
#     predicted_class = np.argmax(predictions, axis=1)
#     confidence = np.max(predictions)
#     return predicted_class[0], confidence

# # Grad-CAM
# def grad_cam(model, img, layer_name='conv2d_2'):
#     img_tensor = np.expand_dims(img, axis=0)
#     grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_tensor)
#         loss = predictions[:, tf.argmax(predictions[0])]
#     grads = tape.gradient(loss, conv_outputs)[0]
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
#     conv_outputs = conv_outputs[0]
#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap).numpy()
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap)
#     return heatmap

# def overlay_heatmap(img, heatmap):
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
#     return superimposed_img

# # Extract frames from video
# def extract_frames(video_path, frame_interval=30):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_count % frame_interval == 0:
#             frame = cv2.resize(frame, (224, 224))
#             frame = frame / 255.0
#             frames.append(frame)
#         frame_count += 1
#     cap.release()
#     return np.array(frames)

# # Streamlit UI
# st.title("🧍‍♂️ Gait Anomaly Detection")
# st.write("Upload an **image** or **video** to detect gait anomaly.")

# file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

# if file:
#     file_type = file.type

#     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#         temp_file.write(file.read())
#         temp_path = temp_file.name

#     if "image" in file_type:
#         img = cv2.imread(temp_path)
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         input_data = preprocess_image(img)
#         pred_class, confidence = predict_gait(model, input_data)

#         st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
#         st.success(f"**Prediction:** {gait_types[pred_class]} ({confidence*100:.2f}% confidence)")

#         heatmap = grad_cam(model, img / 255.0)
#         heatmap_img = overlay_heatmap(np.uint8(img), heatmap)
#         st.image(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB), caption="Grad-CAM Heatmap", use_column_width=True)

#     elif "video" in file_type:
#         st.video(file)

#         frames = extract_frames(temp_path)
#         if len(frames) > 0:
#             pred_class, confidence = predict_gait(model, np.expand_dims(frames[0], axis=0))
#             st.success(f"**Prediction from video:** {gait_types[pred_class]} ({confidence*100:.2f}% confidence)")
#         else:
#             st.error("No frames extracted from the video.")

#     else:
#         st.error("Unsupported file format.")







import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import tempfile
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gait Anomaly Detection", layout="wide")

# Load your trained model
@st.cache_resource
def load_trained_model():
    model = load_model("gait_anomaly_model21.h5")
    return model

model = load_trained_model()
# Warm up the model with a dummy input to build/initialize all internal layer outputs
model(np.zeros((1, 224, 224, 3)))
gait_types = ["Antalgic Gait", "Parkinson Gait", "Spastic Gait", "Scissor Gait"]

# Image Preprocessing
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Video Frame Extraction
def extract_frames(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_interval == 0:
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frames.append(frame)
    cap.release()
    return np.array(frames)

# Predict Gait
def predict_gait(input_data):
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)
    return predicted_class[0], confidence

# Grad-CAM
def grad_cam(img, layer_name='conv2d_2'):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(layer_name).output, model.layers[-1].output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(img, axis=0))
        loss = predictions[:, np.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

# Overlay heatmap
def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(np.uint8(255 * img), 0.6, heatmap, 0.4, 0)

# Streamlit UI
st.title("🚶‍♂️ Gait Anomaly Detection System")

st.markdown("""
Upload a **video or image** of a person walking. This app will analyze their gait and identify any anomalies, such as:
- Antalgic Gait
- Parkinsonian Gait
- Spastic Gait
- Scissor Gait
""")

uploaded_file = st.file_uploader("📤 Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    file_path = tfile.name

    if file_type in ["jpg", "jpeg", "png"]:
        st.subheader("🖼 Uploaded Image")
        image = cv2.imread(file_path)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")

        input_data = preprocess_image(image)
        pred_class, confidence = predict_gait(input_data)
        st.success(f"🧠 **Predicted Gait Type**: {gait_types[pred_class]} ({confidence:.2%} confidence)")

        st.subheader("🔥 Grad-CAM Heatmap")
        heatmap = grad_cam(input_data[0])
        cam_img = overlay_heatmap(input_data[0], heatmap)
        st.image(cam_img, caption="Visual explanation using Grad-CAM", channels="BGR")

    elif file_type in ["mp4", "avi"]:
        st.subheader("🎞 Uploaded Video")
        st.video(uploaded_file)

        st.info("⏳ Extracting frames and analyzing...")
        frames = extract_frames(file_path, frame_interval=30)

        predictions = []
        for frame in frames:
            pred_class, _ = predict_gait(np.expand_dims(frame, axis=0))
            predictions.append(pred_class)

        if predictions:
            final_pred = max(set(predictions), key=predictions.count)
            st.success(f"🧠 **Predicted Gait Type from Video**: {gait_types[final_pred]}")
        else:
            st.warning("⚠️ No valid frames were extracted from the video.")
    else:
        st.error("❌ Unsupported file type. Please upload a .jpg, .png, or .mp4/.avi file.")

st.markdown("---")
st.caption("Developed by Santhosh Naik | Gait Analysis using Deep Learning | © 2025")
