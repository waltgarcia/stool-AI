# Bristol Stool Scale Classifier - Computer Vision System

An AI-powered computer vision application that classifies stool samples according to the Bristol Stool Scale using deep learning.

## Features

- 📸 **Multiple Input Methods**: Upload images, take photos with camera, or provide image URLs
- 🤖 **Advanced AI Model**: Uses ResNet50 architecture for accurate classification
- 🎨 **Image Preprocessing**: Automatic contrast enhancement and noise reduction
- 📊 **Confidence Scores**: Real-time probability distribution across all 7 types
- 💡 **Personalized Recommendations**: Health tips based on classification results
- 📈 **Analysis History**: Track and export your analysis history
- 🤝 **Continuous Learning**: Users provide feedback to improve the model
- 🔧 **Admin Dashboard**: Review and classify user submissions
- 🚀 **Model Retraining**: Automated workflow for continuous improvement

## Quick Start

### Run Locally

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the main app:**
```bash
streamlit run streamlit_app.py
```
The app will be available at `http://localhost:8501`

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click "Deploy an app"
4. Enter: `waltgarcia/stool-AI` and file `streamlit_app.py`

Your app will be live at: `[https://stool-ai-waltgarcia.streamlit.app](https://blank-app-vdw2s266yel.streamlit.app/)`

## 🔄 Training Pipeline

This project includes a workflow to continuously improve the model:

### User Feedback Collection
- Users upload images and get predictions
- They can confirm or correct the classification
- Images are automatically saved for training

### Admin Review
```bash
streamlit run admin_dashboard.py
```
- Review user submissions
- Manually classify if needed
- Move approved images to training dataset

### Model Retraining
```bash
python retrain_model.py
```
- Trains new model with user-provided data
- Automatically saves best weights
- Model improves with more data

See [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md) for detailed instructions.

## 📁 Project Structure

```
stool-AI/
├── streamlit_app.py           # Main application
├── admin_dashboard.py         # Admin panel for reviewing submissions
├── retrain_model.py          # Script to train improved models
├── train_model.py            # Original training script
│
├── data/
│   ├── user_submissions/     # User uploaded images
│   └── bristol_stool_dataset/ # Training dataset (type_1 through type_7)
│
├── model_weights.pth         # Current model weights
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🏥 Bristol Stool Scale

The Bristol Stool Scale classifies stool into 7 types:

| Type | Name | Classification |
|------|------|-----------------|
| 1 | Separate hard lumps | Severe Constipation |
| 2 | Lumpy and sausage-like | Mild Constipation |
| 3 | Sausage-like with cracks | Normal |
| 4 | Smooth and soft sausage | Normal (Ideal) |
| 5 | Soft blobs | Normal |
| 6 | Mushy consistency | Borderline Diarrhea |
| 7 | Liquid consistency | Diarrhea |

## ⚙️ Technologies

- **Frontend**: Streamlit
- **ML Framework**: PyTorch
- **Computer Vision**: torchvision, OpenCV
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib

## 📊 Model Architecture

- **Backbone**: ResNet50
- **Input Size**: 224×224 pixels
- **Output Classes**: 7 (Bristol Stool Types)
- **Preprocessing**: Normalization + CLAHE enhancement

## 🚀 Deployment

The app is deployed on Streamlit Community Cloud and accessible at:
```
https://stool-ai-waltgarcia.streamlit.app
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for **informational and educational purposes only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a physician or qualified health provider with questions about medical conditions.

## 👤 Author

Walt Garcia - [GitHub](https://github.com/waltgarcia)

## 🤝 Contributing

Contributions are welcome! You can help by:
1. Using the app and providing feedback
2. Uploading images to improve the model
3. Reporting bugs or suggesting features
4. Sharing the app with others

---

**Made with ❤️ using Streamlit and PyTorch**
