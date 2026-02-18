# app.py - Main Streamlit Application

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cv2
from datetime import datetime
import os
import json
import hashlib
from pathlib import Path
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Setup data directories
DATA_DIR = Path("data/user_submissions")
DATA_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_CSV = DATA_DIR / "submissions.csv"
CLASSIFIED_DATA_DIR = Path("data/bristol_stool_dataset")
CLASSIFIED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Bristol Stool Scale Classifier",
    page_icon="💩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        height: 50px;
        font-size: 16px;
        font-weight: bold;
    }
    .camera-box {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# Bristol Scale Data
BRISTOL_DATA = {
    "Type 1": {
        "name": "Separate hard lumps",
        "description": "Separate hard lumps, like nuts (hard to pass)",
        "interpretation": "Severe Constipation",
        "color": "#FF4444",
        "transit_time": ">100 hours",
        "fiber_needs": "High",
        "recommendations": [
            "💧 Drink 8-10 glasses of water daily",
            "🌾 Increase fiber intake (25-35g per day)",
            "🏃‍♂️ Exercise regularly (30 min daily)",
            "🍎 Eat prunes, figs, and high-fiber fruits",
            "💊 Consider fiber supplements (psyllium)"
        ]
    },
    "Type 2": {
        "name": "Lumpy and sausage-like",
        "description": "Sausage-shaped but lumpy",
        "interpretation": "Mild Constipation",
        "color": "#FF8844",
        "transit_time": ">100 hours",
        "fiber_needs": "High",
        "recommendations": [
            "💧 Stay hydrated throughout the day",
            "🥗 Add more vegetables to meals",
            "🌰 Include nuts and seeds in diet",
            "🏃‍♀️ Increase physical activity",
            "📅 Establish regular bathroom routine"
        ]
    },
    "Type 3": {
        "name": "Sausage-like with cracks",
        "description": "Like a sausage but with cracks on the surface",
        "interpretation": "Normal",
        "color": "#44FF44",
        "transit_time": "~72 hours",
        "fiber_needs": "Normal",
        "recommendations": [
            "✅ Continue current healthy habits",
            "🥦 Maintain balanced fiber intake",
            "💧 Keep hydrated",
            "🏃 Regular exercise is beneficial",
            "📊 Monitor any changes in consistency"
        ]
    },
    "Type 4": {
        "name": "Smooth and soft sausage",
        "description": "Like a sausage or snake, smooth and soft",
        "interpretation": "Normal (Ideal)",
        "color": "#44FF44",
        "transit_time": "~72 hours",
        "fiber_needs": "Normal",
        "recommendations": [
            "✅ This is the ideal stool consistency!",
            "🥦 Maintain your current diet",
            "💧 Continue good hydration",
            "🏃 Keep up with regular exercise",
            "🎯 This is what to aim for"
        ]
    },
    "Type 5": {
        "name": "Soft blobs",
        "description": "Soft blobs with clear-cut edges (passed easily)",
        "interpretation": "Normal",
        "color": "#44FF44",
        "transit_time": "~48 hours",
        "fiber_needs": "Normal",
        "recommendations": [
            "✅ Within normal range",
            "🌿 Monitor for any changes",
            "💧 Maintain hydration",
            "🥗 Continue balanced diet",
            "📝 Note any dietary triggers"
        ]
    },
    "Type 6": {
        "name": "Mushy consistency",
        "description": "Fluffy pieces with ragged edges, a mushy stool",
        "interpretation": "Borderline Diarrhea",
        "color": "#FFFF44",
        "transit_time": "~36 hours",
        "fiber_needs": "Low",
        "recommendations": [
            "⚠️ Monitor frequency and duration",
            "🌿 Consider probiotics",
            "☕ Reduce caffeine if excessive",
            "🍚 Eat binding foods (rice, bananas)",
            "👨‍⚕️ Consult doctor if persistent"
        ]
    },
    "Type 7": {
        "name": "Liquid consistency",
        "description": "Watery, no solid pieces (entirely liquid)",
        "interpretation": "Diarrhea",
        "color": "#FF4444",
        "transit_time": "<24 hours",
        "fiber_needs": "Very Low",
        "recommendations": [
            "🚨 Stay hydrated with electrolytes",
            "🥣 Follow BRAT diet temporarily",
            "💊 Consider over-the-counter options",
            "🏥 Seek medical attention if severe",
            "📋 Rule out underlying conditions"
        ]
    }
}

# Model definition
class BristolClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(BristolClassifier, self).__init__()
        # Use ResNet50 as backbone without automatic weight downloads
        try:
            # Newer torchvision versions use `weights` argument; set to None to avoid downloads
            self.backbone = models.resnet50(weights=None)
        except Exception:
            # Fallback for older torchvision versions
            self.backbone = models.resnet50(pretrained=False)

        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        # Replace classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model function
@st.cache_resource
def load_model():
    """Load or create the model"""
    try:
        model = BristolClassifier()
        model = model.to(st.session_state.device)

        # Attempt to load local weights if available (preferred for reproducibility)
        possible_weights = [
            Path("model_weights.pth"),
            Path("model.pth"),
            Path("weights.pth")
        ]
        loaded = False
        for p in possible_weights:
            if p.exists():
                try:
                    state = torch.load(p, map_location=st.session_state.device)
                    model.load_state_dict(state)
                    loaded = True
                    st.info(f"Loaded model weights from {p}")
                    break
                except Exception as e:
                    st.warning(f"Found weights file {p} but failed to load: {e}")

        if not loaded:
            st.warning("No local model weights found — model will run with random/untrained weights.")

        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

# Image enhancement function
def enhance_image(image):
    """Apply image enhancement techniques"""
    # Convert PIL to numpy
    img_np = np.array(image)
    
    # Convert to RGB if grayscale
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Denoise
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    return Image.fromarray(enhanced)

# Classification function
def classify_image(model, image_tensor):
    """Run classification on preprocessed image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(st.session_state.device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
    return predicted_class + 1, probabilities[0].cpu().numpy()  # +1 because types are 1-7

# Save image function for training data
def save_submission(image, predicted_type, image_hash=None):
    """Save image submission with metadata"""
    if image_hash is None:
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
    
    # Create filename
    filename = f"{image_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = DATA_DIR / filename
    
    # Save image
    image.save(filepath)
    
    # Initialize CSV if doesn't exist
    if not SUBMISSIONS_CSV.exists():
        df = pd.DataFrame(columns=["timestamp", "filename", "predicted_type", "confirmed_type", "user_feedback", "used_for_training"])
        df.to_csv(SUBMISSIONS_CSV, index=False)
    
    # Add entry to CSV
    df = pd.read_csv(SUBMISSIONS_CSV)
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "predicted_type": f"Type {predicted_type}",
        "confirmed_type": "",
        "user_feedback": "",
        "used_for_training": False
    }
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(SUBMISSIONS_CSV, index=False)
    
    return filepath

# Read submissions
def get_submissions():
    """Get all submissions from CSV"""
    if SUBMISSIONS_CSV.exists():
        return pd.read_csv(SUBMISSIONS_CSV)
    return pd.DataFrame(columns=["timestamp", "filename", "predicted_type", "confirmed_type", "user_feedback", "used_for_training"])

# Header
st.markdown("""
    <div class="main-header">
        <h1>💩 Bristol Stool Scale Classifier</h1>
        <p>AI-Powered Computer Vision System for Stool Classification</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100?text=Bristol+Scale+AI")
    
    st.markdown("## 📸 Input Options")
    input_method = st.radio(
        "Choose input method:",
        ["📤 Upload Image", "📷 Take Photo", "🔗 Image URL"],
        key="input_method"
    )
    
    st.markdown("## ⚙️ Preprocessing Options")
    enhance_contrast = st.checkbox("Enhance Contrast", value=True)
    denoise = st.checkbox("Remove Noise", value=True)
    
    st.markdown("## 📊 Quick Reference")
    quick_ref_df = pd.DataFrame({
        "Type": list(BRISTOL_DATA.keys()),
        "Status": [data["interpretation"] for data in BRISTOL_DATA.values()]
    })
    st.dataframe(quick_ref_df, use_container_width=True, hide_index=True)
    
    st.markdown("## ℹ️ About")
    st.info(
        "This AI model uses computer vision to classify stool according to "
        "the Bristol Stool Scale. It's trained on thousands of images and "
        "provides real-time analysis with confidence scores."
    )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📷 Image Input")
    
    # Handle different input methods
    image = None
    if input_method == "📤 Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'heic']
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            
    elif input_method == "📷 Take Photo":
        st.markdown('<div class="camera-box">', unsafe_allow_html=True)
        camera_image = st.camera_input("Take a photo")
        if camera_image:
            image = Image.open(camera_image)
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:  # Image URL
        url = st.text_input("Enter image URL:")
        if url:
            try:
                import requests
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
            except:
                st.error("Failed to load image from URL")
    
    # Display image
    if image:
        st.image(image, caption="Input Image")
        
        # Enhance image if options selected
        if enhance_contrast or denoise:
            with st.spinner("Enhancing image..."):
                enhanced_image = enhance_image(image)
                st.image(enhanced_image, caption="Enhanced Image")
                image = enhanced_image

with col2:
    st.markdown("### 🔍 Analysis Results")
    
    # Analysis button
    if st.button("🚀 Analyze Image"):
        if image is None:
            st.warning("Please provide an image first!")
        else:
            # Load model
            with st.spinner("Loading AI model..."):
                model = load_model()
                st.session_state.model = model
            
            if model is None:
                st.error("Failed to load model. Please try again.")
            else:
                # Preprocess and classify
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    image_tensor = preprocess_image(image)
                    
                    # Classify
                    predicted_type, probabilities = classify_image(model, image_tensor)
                    
                    # Store in history
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.history.append({
                        "timestamp": timestamp,
                        "predicted_type": f"Type {predicted_type}",
                        "confidence": probabilities[predicted_type-1],
                        "image": image
                    })
                
                # Display results
                result = BRISTOL_DATA[f"Type {predicted_type}"]
                
                st.markdown(f"""
                    <div class="result-card">
                        <h3 style="text-align: center;">Classification Result</h3>
                        <h1 style="text-align: center; font-size: 48px;">Type {predicted_type}</h1>
                        <h4 style="text-align: center;">{result['name']}</h4>
                        <p style="text-align: center;">{result['description']}</p>
                        <div style="background-color: white; color: black; padding: 10px; border-radius: 5px; text-align: center; margin-top: 10px;">
                            <strong>Interpretation:</strong> {result['interpretation']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Confidence scores
                st.markdown("#### 📊 Confidence Scores")
                
                # Create confidence dataframe
                confidence_df = pd.DataFrame({
                    "Type": [f"Type {i+1}" for i in range(7)],
                    "Confidence": probabilities,
                    "Description": [BRISTOL_DATA[f"Type {i+1}"]["name"] for i in range(7)]
                })
                
                # Plot confidence bars
                fig = px.bar(
                    confidence_df,
                    x="Type",
                    y="Confidence",
                    color="Confidence",
                    color_continuous_scale=["red", "yellow", "green"],
                    range_color=[0, 1],
                    text=confidence_df["Confidence"].apply(lambda x: f"{x:.1%}"),
                    hover_data=["Description"]
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    yaxis_range=[0, 1],
                    yaxis_tickformat=".0%"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Transit time and fiber needs
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Transit Time", result['transit_time'])
                with col_b:
                    st.metric("Fiber Needs", result['fiber_needs'])
                
                # Save submission and feedback section
                st.markdown("---")
                st.markdown("### 🤝 Help Improve the Model")
                
                col_feedback1, col_feedback2 = st.columns(2)
                
                with col_feedback1:
                    st.markdown("**Is this classification correct?**")
                    feedback_correct = st.radio(
                        "Did the model get it right?",
                        ["Yes, correct ✅", "No, incorrect ❌"],
                        key="feedback_correct",
                        label_visibility="collapsed"
                    )
                
                with col_feedback2:
                    if feedback_correct == "No, incorrect ❌":
                        correct_type = st.selectbox(
                            "What's the correct type?",
                            [f"Type {i+1}" for i in range(7)],
                            key="correct_type"
                        )
                    else:
                        correct_type = None
                
                # Additional feedback
                user_feedback = st.text_area(
                    "Any additional comments? (optional)",
                    placeholder="E.g., image quality, lighting issues, etc.",
                    key="user_feedback_text"
                )
                
                # Save button
                if st.button("💾 Save & Help Train Model"):
                    # Save image
                    save_submission(image, predicted_type)
                    
                    # Update feedback
                    df = get_submissions()
                    last_idx = len(df) - 1
                    df.loc[last_idx, "user_feedback"] = user_feedback
                    df.loc[last_idx, "confirmed_type"] = correct_type if correct_type else f"Type {predicted_type}"
                    df.to_csv(SUBMISSIONS_CSV, index=False)
                    
                    st.success("✅ Thank you! Your image has been saved and will help improve the model.")
                    st.info("Periodically, saved images will be reviewed and used to train a better model.")


# Recommendations and details
if image and 'predicted_type' in locals():
    st.markdown("---")
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.markdown("### 💡 Recommendations")
        for rec in result['recommendations']:
            st.markdown(f"- {rec}")
    
    with col4:
        st.markdown("### 📋 Additional Insights")
        
        # AI-generated insights
        confidence_level = probabilities[predicted_type-1]
        insight = f"""
        **Pattern Recognition:** {result['description']}
        
        **Confidence Analysis:** 
        - {'✅ High confidence detection' if confidence_level > 0.8 else '⚠️ Moderate confidence' if confidence_level > 0.6 else '❌ Low confidence'}
        - Model is {confidence_level:.1%} confident in this classification
        
        **Recommendation:** {
            'This is a reliable classification. Follow the recommendations above.'
            if confidence_level > 0.8 else
            'Consider taking another photo with better lighting for more accurate results.'
        }
        """
        st.info(insight)

# History tab
st.markdown("---")
st.markdown("### 📜 Analysis History")

if st.session_state.history:
    # Create history dataframe
    history_df = pd.DataFrame([
        {
            "Timestamp": h["timestamp"],
            "Predicted Type": h["predicted_type"],
            "Confidence": f"{h['confidence']:.1%}"
        }
        for h in st.session_state.history[-10:]  # Show last 10
    ])
    
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    with col_h2:
        if st.button("📥 Download History"):
            # Convert to CSV
            csv = history_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="bristol_history.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
else:
    st.info("No analysis history yet. Start by analyzing an image!")

# Information section
with st.expander("ℹ️ About the Bristol Stool Scale"):
    st.markdown("""
    The Bristol Stool Scale is a medical aid designed to classify the form of human feces into seven categories.
    It was developed by Dr. Ken Heaton at the University of Bristol and published in the Scandinavian Journal of Gastroenterology in 1997.
    
    **The seven types are:**
    1. **Type 1:** Separate hard lumps (severe constipation)
    2. **Type 2:** Lumpy and sausage-like (mild constipation)
    3. **Type 3:** Sausage-like with cracks (normal)
    4. **Type 4:** Smooth and soft sausage (normal - ideal)
    5. **Type 5:** Soft blobs (normal)
    6. **Type 6:** Mushy consistency (borderline diarrhea)
    7. **Type 7:** Liquid consistency (diarrhea)
    
    **Important Disclaimer:** This tool is for informational and educational purposes only. 
    It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. 
    Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; padding: 10px;">
        Made with ❤️ using Streamlit and PyTorch | Version 1.0.0
    </div>
    """,
    unsafe_allow_html=True
)