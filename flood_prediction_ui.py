import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
# Use InceptionV3 only - it's the only model that works properly
# Baseline CNN and ResNet50 are biased due to class imbalance in training
MODEL_THRESHOLDS = {
    'Baseline CNN': 0.50,      # May not work well - trained on imbalanced data
    'ResNet50': 0.50,          # May not work well - trained on imbalanced data  
    'InceptionV3': 0.50        # Works best - use this model's predictions
}
DEFAULT_THRESHOLD = 0.50

# Page configuration
st.set_page_config(
    page_title="Flood Detection System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .flooded {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .non-flooded {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_paths = {
        # 'Baseline CNN': 'best_baseline_model.keras',  # Disabled - biased model
        # 'ResNet50': 'best_resnet50_model.keras',      # Disabled - biased model
        'InceptionV3': 'best_inceptionv3_model.keras'   # Only use InceptionV3 - it works
    }
    
    for model_name, model_path in model_paths.items():
        try:
            if Path(model_path).exists():
                models[model_name] = keras.models.load_model(model_path)
                st.sidebar.success(f"✓ {model_name} loaded")
            else:
                st.sidebar.warning(f"⚠ {model_name} not found at {model_path}")
        except Exception as e:
            st.sidebar.error(f"✗ Error loading {model_name}: {str(e)}")
    
    return models

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert RGBA to RGB if needed
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Resize image
    img_resized = cv2.resize(img_array, target_size)
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_resized

def get_prediction_explanation(prediction, confidence):
    """Generate explanation for the prediction"""
    if prediction == "Flooded":
        if confidence > 0.8:
            explanation = "The model is highly confident that this image shows flooded conditions. Key indicators may include: standing water, submerged areas, water-covered surfaces, or flood-related damage."
        elif confidence > 0.6:
            explanation = "The model indicates flooded conditions with moderate confidence. The image likely shows some signs of flooding such as water accumulation or flood-affected areas."
        else:
            explanation = "The model suggests flooded conditions but with lower confidence. There may be some ambiguous features that could indicate flooding."
    else:
        if confidence > 0.8:
            explanation = "The model is highly confident that this image shows non-flooded conditions. The area appears to be dry with no visible signs of flooding or water accumulation."
        elif confidence > 0.6:
            explanation = "The model indicates non-flooded conditions with moderate confidence. The image likely shows normal, dry conditions without significant water presence."
        else:
            explanation = "The model suggests non-flooded conditions but with lower confidence. The image may have some ambiguous features."
    
    return explanation

def create_confidence_visualization(predictions_dict):
    """Create visualization of model predictions"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of predictions
    model_names = list(predictions_dict.keys())
    flooded_probs = [pred['flooded_prob'] for pred in predictions_dict.values()]
    non_flooded_probs = [pred['non_flooded_prob'] for pred in predictions_dict.values()]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0].bar(x - width/2, flooded_probs, width, label='Flooded', color='#f44336', alpha=0.8)
    axes[0].bar(x + width/2, non_flooded_probs, width, label='Non-Flooded', color='#4caf50', alpha=0.8)
    axes[0].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Probability', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Predictions Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Add value labels on bars
    for i, (f, nf) in enumerate(zip(flooded_probs, non_flooded_probs)):
        axes[0].text(i - width/2, f + 0.02, f'{f:.2%}', ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width/2, nf + 0.02, f'{nf:.2%}', ha='center', va='bottom', fontsize=9)
    
    # Confidence scores
    confidences = [pred['confidence'] for pred in predictions_dict.values()]
    colors = ['#4caf50' if pred['prediction'] == 'Non-Flooded' else '#f44336' 
              for pred in predictions_dict.values()]
    
    bars = axes[1].barh(model_names, confidences, color=colors, alpha=0.8)
    axes[1].set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Model Confidence Scores', fontsize=14, fontweight='bold')
    axes[1].set_xlim([0, 1])
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        axes[1].text(conf + 0.01, i, f'{conf:.2%}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🌊 Flood Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Model Information")
    st.sidebar.markdown("### Available Models")
    
    # Load models
    models = load_models()
    
    if not models:
        st.error("❌ No models found! Please ensure model files are in the current directory.")
        st.stop()
    
    st.sidebar.markdown(f"**{len(models)} model(s) loaded**")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Instructions")
    st.sidebar.markdown("""
    1. Upload a satellite/aerial image
    2. Wait for predictions
    3. View results and explanations
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a satellite or aerial image to classify"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Preprocess image
            img_batch, img_resized = preprocess_image(image)
            
            # Make predictions
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    predictions = {}
                    
                    for model_name, model in models.items():
                        try:
                            # Get prediction
                            # Model outputs probability of class 1 (Non-Flooded)
                            # During training: 0 = Flooded, 1 = Non-Flooded
                            pred_proba = model.predict(img_batch, verbose=0)[0][0]
                            non_flooded_prob = float(pred_proba)  # Probability of class 1 (Non-Flooded)
                            flooded_prob = float(1 - pred_proba)  # Probability of class 0 (Flooded)
                            
                            # Determine prediction with model-specific threshold
                            # Different models have different biases due to training differences
                            # Use model-specific threshold to account for this
                            threshold = MODEL_THRESHOLDS.get(model_name, DEFAULT_THRESHOLD)
                            
                            # Standard prediction logic
                            # With properly trained models (using class weights), use standard 0.5 threshold
                            prediction = "Flooded" if flooded_prob > threshold else "Non-Flooded"
                            confidence = max(flooded_prob, non_flooded_prob)
                            
                            predictions[model_name] = {
                                'prediction': prediction,
                                'flooded_prob': flooded_prob,
                                'non_flooded_prob': non_flooded_prob,
                                'confidence': confidence
                            }
                        except Exception as e:
                            st.error(f"Error predicting with {model_name}: {str(e)}")
                    
                    # Store predictions in session state
                    st.session_state['predictions'] = predictions
                    st.session_state['image'] = img_resized
    
    with col2:
        st.subheader("Prediction Results")
        
        if 'predictions' in st.session_state and st.session_state['predictions']:
            predictions = st.session_state['predictions']
            
            # Ensemble prediction (average of all models)
            # Use weighted average threshold based on individual model thresholds
            # This accounts for different model biases
            model_weights = []
            model_probs = []
            for model_name, pred_data in predictions.items():
                threshold = MODEL_THRESHOLDS.get(model_name, DEFAULT_THRESHOLD)
                # Weight inversely proportional to threshold (models with lower thresholds get more weight)
                weight = 1.0 / (threshold + 0.1)  # Add small value to avoid division by zero
                model_weights.append(weight)
                model_probs.append(pred_data['flooded_prob'])
            
            # Weighted average of flooded probabilities
            if model_weights:
                avg_flooded_prob = np.average(model_probs, weights=model_weights)
                # Use average threshold for ensemble
                avg_threshold = np.mean([MODEL_THRESHOLDS.get(name, DEFAULT_THRESHOLD) 
                                       for name in predictions.keys()])
            else:
                avg_flooded_prob = np.mean([p['flooded_prob'] for p in predictions.values()])
                avg_threshold = DEFAULT_THRESHOLD
            
            ensemble_prediction = "Flooded" if avg_flooded_prob > avg_threshold else "Non-Flooded"
            ensemble_confidence = max(avg_flooded_prob, 1 - avg_flooded_prob)
            
            # Display ensemble result
            st.markdown("### Ensemble Prediction")
            pred_class = "flooded" if ensemble_prediction == "Flooded" else "non-flooded"
            st.markdown(
                f'<div class="prediction-box {pred_class}">'
                f'<p><strong>Prediction:</strong> {ensemble_prediction}</p>'
                f'<p><strong>Confidence:</strong> {ensemble_confidence:.2%}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Explanation
            explanation = get_prediction_explanation(ensemble_prediction, ensemble_confidence)
            st.markdown("### Explanation")
            st.info(explanation)
            
            # Individual model results
            st.markdown("### Individual Model Results")
            
            for model_name, pred_data in predictions.items():
                with st.expander(f"{model_name} - {pred_data['prediction']}"):
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Flooded Probability", f"{pred_data['flooded_prob']:.2%}")
                    with col_b:
                        st.metric("Non-Flooded Probability", f"{pred_data['non_flooded_prob']:.2%}")
                    with col_c:
                        st.metric("Confidence", f"{pred_data['confidence']:.2%}")
            
            # Visualization
            st.markdown("### Visualizations")
            fig = create_confidence_visualization(predictions)
            st.pyplot(fig)
            
            # Statistics
            st.markdown("### Statistics")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Models Agreeing", 
                         f"{sum(1 for p in predictions.values() if p['prediction'] == ensemble_prediction)}/{len(predictions)}")
            
            with col_stat2:
                avg_conf = np.mean([p['confidence'] for p in predictions.values()])
                st.metric("Average Confidence", f"{avg_conf:.2%}")
            
            with col_stat3:
                std_conf = np.std([p['confidence'] for p in predictions.values()])
                st.metric("Confidence Std Dev", f"{std_conf:.4f}")
            
            # Model agreement analysis
            st.markdown("### Model Agreement Analysis")
            agreement_count = sum(1 for p in predictions.values() if p['prediction'] == ensemble_prediction)
            agreement_pct = (agreement_count / len(predictions)) * 100
            
            if agreement_pct == 100:
                st.success(f"✅ All models agree: {ensemble_prediction}")
            elif agreement_pct >= 66:
                st.warning(f"⚠️ Most models agree ({agreement_pct:.0f}%): {ensemble_prediction}")
            else:
                st.error(f"❌ Models disagree. Only {agreement_pct:.0f}% agree on {ensemble_prediction}")
            
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see predictions")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "Flood Detection System | Deep Learning Project | CSC3218"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

