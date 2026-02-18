#!/usr/bin/env python3
"""
Admin Dashboard - Review and classify user submissions for model training
"""

import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
import shutil

# Setup
DATA_DIR = Path("data/user_submissions")
SUBMISSIONS_CSV = DATA_DIR / "submissions.csv"
CLASSIFIED_DATA_DIR = Path("data/bristol_stool_dataset")
CLASSIFIED_DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Bristol Stool - Admin Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔧 Admin Dashboard - Review Submissions")

# Admin password check (optional, consider using environment variable)
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

if not st.session_state.admin_authenticated:
    st.warning("⚠️ Admin authentication required")
    password = st.text_input("Enter admin password:", type="password")
    
    if password:
        # Simple password check (in production, use environment variables)
        ADMIN_PASSWORD = st.secrets.get("admin_password", "admin123")
        if password == ADMIN_PASSWORD:
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password")
    st.stop()

# Load submissions
if SUBMISSIONS_CSV.exists():
    df = pd.read_csv(SUBMISSIONS_CSV)
else:
    df = pd.DataFrame(columns=["timestamp", "filename", "predicted_type", "confirmed_type", "user_feedback", "used_for_training"])

st.markdown(f"### 📊 Total Submissions: {len(df)}")

# Tab 1: Review Pending Submissions
tab1, tab2, tab3 = st.tabs(["Review Pending", "View All", "Training Data"])

with tab1:
    st.markdown("### Review submissions pending classification")
    
    # Filter pending submissions (no confirmed_type)
    pending = df[df["confirmed_type"].isna() | (df["confirmed_type"] == "")]
    
    if len(pending) > 0:
        st.info(f"📋 {len(pending)} submissions pending review")
        
        # Create columns for pagination
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            page = st.number_input("Page", min_value=1, max_value=max(1, len(pending)//5 + 1), value=1)
        
        start_idx = (page - 1) * 5
        end_idx = min(start_idx + 5, len(pending))
        pending_page = pending.iloc[start_idx:end_idx]
        
        for idx, row in pending_page.iterrows():
            with st.container():
                st.markdown("---")
                
                col_img, col_info = st.columns([1, 2])
                
                with col_img:
                    filepath = DATA_DIR / row["filename"]
                    if filepath.exists():
                        image = Image.open(filepath)
                        st.image(image, caption=row["filename"], use_column_width=True)
                    else:
                        st.warning("Image not found")
                
                with col_info:
                    st.markdown(f"**Predicted:** {row['predicted_type']}")
                    st.markdown(f"**Timestamp:** {row['timestamp']}")
                    if pd.notna(row["user_feedback"]) and row["user_feedback"] != "":
                        st.markdown(f"**User Feedback:** {row['user_feedback']}")
                    
                    # Classification selector
                    correct_type = st.selectbox(
                        "Correct Classification:",
                        [f"Type {i+1}" for i in range(7)],
                        key=f"correct_{idx}"
                    )
                    
                    # Save button
                    if st.button("✅ Confirm Classification", key=f"confirm_{idx}"):
                        df.loc[idx, "confirmed_type"] = correct_type
                        df.loc[idx, "used_for_training"] = False  # Will mark as used after training
                        df.to_csv(SUBMISSIONS_CSV, index=False)
                        st.success(f"✅ Marked as {correct_type}")
                        st.rerun()
    else:
        st.success("✅ All submissions have been reviewed!")

with tab2:
    st.markdown("### All Submissions")
    
    # Display all submissions
    cols_display = df[["timestamp", "filename", "predicted_type", "confirmed_type", "user_feedback"]].copy()
    st.dataframe(cols_display, use_container_width=True)
    
    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download submissions.csv",
        csv,
        "submissions.csv",
        "text/csv"
    )

with tab3:
    st.markdown("### Training Data Management")
    
    reviewed = df[df["confirmed_type"].notna() & (df["confirmed_type"] != "")]
    
    st.info(f"📊 {len(reviewed)} submissions have been classified")
    st.info(f"🚀 {len(reviewed[reviewed['used_for_training'] == True])} already used for training")
    
    # Move to training dataset
    st.markdown("#### Move to Training Dataset")
    
    untraining = reviewed[reviewed["used_for_training"] != True]
    
    if len(untraining) > 0:
        if st.button("📦 Move all reviewed images to training dataset"):
            count = 0
            for idx, row in untraining.iterrows():
                filepath = DATA_DIR / row["filename"]
                if filepath.exists():
                    # Create type directory
                    type_num = row["confirmed_type"].replace("Type ", "")
                    type_dir = CLASSIFIED_DATA_DIR / f"type_{type_num}"
                    type_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Move file
                    dest_path = type_dir / filepath.name
                    shutil.copy2(filepath, dest_path)
                    
                    # Mark as used
                    df.loc[idx, "used_for_training"] = True
                    count += 1
            
            df.to_csv(SUBMISSIONS_CSV, index=False)
            st.success(f"✅ {count} images moved to training dataset")
            st.info("💡 Run `python retrain_model.py` to train a new model with these images")
    else:
        st.info("✅ All reviewed images have been used for training")
    
    # Show training dataset structure
    st.markdown("#### Training Dataset Structure")
    
    type_counts = {}
    for i in range(1, 8):
        type_dir = CLASSIFIED_DATA_DIR / f"type_{i}"
        if type_dir.exists():
            count = len(list(type_dir.glob("*.png"))) + len(list(type_dir.glob("*.jpg")))
            type_counts[f"Type {i}"] = count
    
    if type_counts:
        type_df = pd.DataFrame(list(type_counts.items()), columns=["Type", "Images"])
        st.bar_chart(type_df.set_index("Type"))
    else:
        st.info("No training data yet")

# Footer
st.markdown("---")
st.markdown("""
### 📚 How it works:
1. Users upload images and the model predicts the type
2. User provides feedback if prediction is wrong
3. Admin reviews submissions and confirms correct classification
4. Confirmed images are moved to training dataset
5. Use `python retrain_model.py` to train an improved model
6. New model weights replace old ones for continuous improvement
""")
