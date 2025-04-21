import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Page Configuration
st.set_page_config(page_title="PneumoScan", page_icon="🫁", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .result-normal { background-color: #d4edda; color: #155724; }
    .result-pneumonia { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Load Trained Model
@st.cache_resource
def load_trained_model():
    try:
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

# Image Preprocessing
def preprocess_image(image):
    img = image.resize((150, 150))
    img_array = np.array(img)
    
    if len(img_array.shape) == 3 and img_array.shape[2] > 1:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=[0, -1])
    
    return img_array

# Prediction Function
def predict_pneumonia(image):
    model = load_trained_model()
    if model is None:
        return None
    
    img_array = preprocess_image(image)
    
    # Simulated prediction (replace with actual model prediction)
    prediction = np.array([[np.random.random(), 1 - np.random.random()]])
    
    class_idx = np.argmax(prediction[0])
    categories = ['NORMAL', 'PNEUMONIA']
    
    return {
        'class': categories[class_idx],
        'confidence': float(prediction[0][class_idx] * 100)
    }

# Main App
def main():
    st.title("PneumoScan: COVID-19 Pneumonia Detection")
    
    # Input Method Selection
    input_method = st.radio("Select Input Method", ["Upload Image", "Camera Capture"])
    
    # Image Input
    image = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
    else:
        camera_input = st.camera_input("Take a picture")
        if camera_input:
            image = Image.open(camera_input).convert('RGB')
    
    # Display Image
    if image:
        st.image(image, width=300, caption="X-ray Image")
        
        # Analyze Button
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                result = predict_pneumonia(image)
                
                if result:
                    # Result Display
                    if result['class'] == 'NORMAL':
                        st.markdown(
                            '<div class="result-normal">'
                            '<h2>✅ Normal</h2>'
                            '<p>No pneumonia signs detected.</p>'
                            '</div>', 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="result-pneumonia">'
                            '<h2>⚠️ Pneumonia Detected</h2>'
                            '<p>Potential pneumonia signs found.</p>'
                            '</div>', 
                            unsafe_allow_html=True
                        )
                    
                    # Recommendations
                    st.subheader("Recommendations")
                    st.warning("""
                    • Consult healthcare professional
                    • Additional testing may be required
                    • Professional medical advice is crucial
                    """)
    
    # Disclaimer
    st.warning("⚠️ EDUCATIONAL PURPOSE ONLY - NOT A MEDICAL DIAGNOSTIC TOOL")

if __name__ == "__main__":
    main()