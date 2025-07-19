# ==========================================
# 🫁 LUNG NODULE DETECTION AI - INTERFACE WEB AMÉLIORÉE
# 🎯 Application Streamlit avec design professionnel
# ==========================================

import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import pandas as pd
from pathlib import Path

# Configuration
WEB_CONFIG = {
    'model_name': 'Lung Nodule Detection AI',
    'version': '2.0 - Amélioré',
    'performance': {
        'f1_score': 0.821,
        'precision': 0.765,
        'recall': 0.886,
        'grade': 'A+ (98.5%)'
    },
    'confidence_threshold': 0.70,
    'image_size': (640, 640),
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225]
}

# Configuration Streamlit avec design amélioré
st.set_page_config(
    page_title="🫁 Lung Nodule Detection AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour améliorer le design
st.markdown("""
<style>
    /* Améliorer l'apparence générale */
    .main {
        padding-top: 1rem;
    }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Cards pour les sections */
    .info-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .success-card {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }

    .warning-card {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }

    /* Métriques */
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }

    /* Boutons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Instructions */
    .instructions {
        background-color: #e3f2fd;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 1rem;
    }

    /* Upload zone */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Chemins
MODELS_PATH = Path("../models")
IMPROVED_MODELS_PATH = MODELS_PATH / "improved"


def create_model(num_classes=2):
    """Créer le modèle Faster R-CNN"""
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Transformation
transform = A.Compose([
    A.Resize(WEB_CONFIG['image_size'][0], WEB_CONFIG['image_size'][1]),
    A.Normalize(
        mean=WEB_CONFIG['normalize_mean'],
        std=WEB_CONFIG['normalize_std']
    ),
    ToTensorV2()
])


@st.cache_resource
def load_trained_model():
    """Charger le modèle entraîné"""
    model = create_model(num_classes=2)
    model_info = {'name': 'Modèle par défaut', 'epoch': 'N/A', 'val_loss': 'N/A'}

    possible_paths = [
        IMPROVED_MODELS_PATH / "best_improved_model.pth",
        IMPROVED_MODELS_PATH / "final_improved_model.pth",
        MODELS_PATH / "best_model.pth"
    ]

    model_loaded = False
    for path in possible_paths:
        if path.exists():
            try:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model_info = {
                    'name': path.name,
                    'epoch': checkpoint.get('epoch', 'N/A'),
                    'val_loss': checkpoint.get('val_loss', 'N/A')
                }
                model_loaded = True
                break
            except Exception as e:
                continue

    if not model_loaded:
        st.warning("⚠️ Modèle entraîné non trouvé. Utilisation du modèle par défaut.")

    model.to(device)
    model.eval()
    return model, model_info


def predict_image(model, image, confidence_threshold=0.7):
    """Prédire les nodules sur une image"""
    model.eval()

    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)

    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()

    valid_indices = scores >= confidence_threshold
    filtered_boxes = boxes[valid_indices]
    filtered_scores = scores[valid_indices]

    return filtered_boxes, filtered_scores


def draw_predictions(image, boxes, scores, confidence_threshold):
    """Dessiner les prédictions avec style amélioré"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Fond sombre pour contraster avec l'image médicale
    fig.patch.set_facecolor('#2E2E2E')
    ax.set_facecolor('#2E2E2E')

    ax.imshow(image, cmap='gray')
    ax.set_title(f"🔍 Détection de Nodules Pulmonaires (Seuil: {confidence_threshold:.0%})",
                 fontsize=16, fontweight='bold', color='white', pad=20)

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # Couleurs plus contrastées et visibles
        if score >= 0.9:
            color = '#FF0000'  # Rouge vif
            text_color = 'white'
            confidence_level = 'HAUTE'
        elif score >= 0.8:
            color = '#FF6600'  # Orange vif
            text_color = 'white'
            confidence_level = 'MOYENNE'
        else:
            color = '#00FF00'  # Vert vif
            text_color = 'black'  # Texte noir sur vert pour meilleure lisibilité
            confidence_level = 'FAIBLE'

        # Rectangle avec bordure très visible
        rect = patches.Rectangle((x1, y1), width, height,
                                 linewidth=4, edgecolor=color,
                                 facecolor='none', linestyle='-')
        ax.add_patch(rect)

        # Label avec fond coloré et texte contrasté
        label = f"Nodule #{i + 1}\n{confidence_level} ({score:.0%})"
        ax.text(x1, y1 - 25, label,
                bbox=dict(boxstyle="round,pad=0.6", facecolor=color, alpha=0.95,
                          edgecolor='black', linewidth=2),
                fontsize=11, fontweight='bold', color=text_color,
                ha='left', va='bottom')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    return fig


def main():
    """Interface principale avec design amélioré"""

    # Charger le modèle
    model, model_info = load_trained_model()

    # Header principal avec design moderne
    st.markdown(f"""
    <div class="main-header">
        <h1>🫁 {WEB_CONFIG['model_name']}</h1>
        <h3>{WEB_CONFIG['version']}</h3>
        <p style="font-size: 1.2em; margin-top: 1rem;">
            🏆 F1-Score: {WEB_CONFIG['performance']['f1_score']:.1%} | 
            Precision: {WEB_CONFIG['performance']['precision']:.1%} | 
            Recall: {WEB_CONFIG['performance']['recall']:.1%}
        </p>
        <p style="font-size: 1.1em;">
            🎓 Note: {WEB_CONFIG['performance']['grade']} | 
            🤖 Modèle: {model_info['name']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar avec design amélioré
    with st.sidebar:
        st.markdown("### ⚙️ Paramètres de Détection")

        confidence_threshold = st.slider(
            "🎯 Seuil de Confiance",
            min_value=0.1,
            max_value=0.99,
            value=WEB_CONFIG['confidence_threshold'],
            step=0.05,
            help="Plus le seuil est élevé, plus les détections seront certaines"
        )

    

    # Interface principale
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📤 Upload d'Image")

        uploaded_file = st.file_uploader(
            "Choisissez une radiographie pulmonaire",
            type=['png', 'jpg', 'jpeg'],
            help="Formats supportés: PNG, JPG, JPEG (max 200MB)"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            # Affichage de l'image avec style
            st.markdown("#### 🖼️ Image Originale")
            st.image(image, caption=f"📁 {uploaded_file.name}", use_container_width=True)

            # Bouton de détection avec style
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 Analyser l'Image", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyse en cours..."):
                    start_time = time.time()
                    boxes, scores = predict_image(model, image, confidence_threshold)
                    processing_time = time.time() - start_time

                    # Afficher les résultats dans la colonne 2
                    with col2:
                        st.markdown("### 🎯 Résultats de l'Analyse")

                        if len(boxes) > 0:
                            # Message de succès
                            st.markdown(f"""
                            <div class="success-card">
                                <h4>✅ {len(boxes)} nodule(s) détecté(s)</h4>
                                <p>Analyse terminée avec succès!</p>
                            </div>
                            """, unsafe_allow_html=True)

                            # Métriques en cards modernes
                            metric_col1, metric_col2, metric_col3 = st.columns(3)

                            with metric_col1:
                                st.markdown(f"""
                                <div class="metric-container">
                                    <div class="metric-value">{len(boxes)}</div>
                                    <div class="metric-label">Nodules</div>
                                </div>
                                """, unsafe_allow_html=True)

                            with metric_col2:
                                st.markdown(f"""
                                <div class="metric-container">
                                    <div class="metric-value">{max(scores):.2f}</div>
                                    <div class="metric-label">Confiance Max</div>
                                </div>
                                """, unsafe_allow_html=True)

                            with metric_col3:
                                st.markdown(f"""
                                <div class="metric-container">
                                    <div class="metric-value">{processing_time:.1f}s</div>
                                    <div class="metric-label">Temps</div>
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown("<br>", unsafe_allow_html=True)

                            # Visualisation des résultats
                            st.markdown("#### 📊 Visualisation des Détections")
                            fig = draw_predictions(np.array(image), boxes, scores, confidence_threshold)
                            st.pyplot(fig, use_container_width=True)

                            # Tableau des détections avec style
                            st.markdown("#### 📋 Détails des Détections")
                            detection_data = []
                            for i, (box, score) in enumerate(zip(boxes, scores)):
                                x1, y1, x2, y2 = box
                                if score >= 0.9:
                                    level = "🔴 Haute"
                                elif score >= 0.8:
                                    level = "🟠 Moyenne"
                                else:
                                    level = "🟢 Faible"

                                detection_data.append({
                                    "🏷️ Nodule": f"#{i + 1}",
                                    "📊 Confiance": f"{score:.1%}",
                                    "📈 Niveau": level,
                                    "📍 Position": f"({x1:.0f}, {y1:.0f})",
                                    "📐 Taille": f"{x2 - x1:.0f}×{y2 - y1:.0f}px"
                                })

                            df = pd.DataFrame(detection_data)
                            st.dataframe(df, use_container_width=True, hide_index=True)

                        else:
                            # Message d'avertissement stylé
                            st.markdown(f"""
                            <div class="warning-card">
                                <h4>⚠️ Aucun nodule détecté</h4>
                                <p>Essayez de réduire le seuil de confiance à {confidence_threshold - 0.1:.1f} 
                                ou vérifiez que l'image est une radiographie pulmonaire.</p>
                            </div>
                            """, unsafe_allow_html=True)



if __name__ == "__main__":
    main()