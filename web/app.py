# -*- coding: utf-8 -*-
"""
🌐 APPLICATION WEB - LUNG NODULE DETECTION AI
🎯 Interface Streamlit pour modèle A+ (F1: 82.1%)
🏥 Détection automatique de nodules pulmonaires
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io
import json
import time
from datetime import datetime
from pathlib import Path

# Configuration globale
CONFIG = {
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

# Cache pour le modèle
@st.cache_resource
def load_model():
    """Charger le modèle avec cache Streamlit pour optimiser les performances"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_model(num_classes=2):
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    model = create_model()

    # Chemins possibles pour le modèle (ordre de priorité)
    model_paths = [
        Path("../models/improved/best_improved_model.pth"),
        Path("models/improved/best_improved_model.pth"),
        Path("../models/best_model.pth"),
        Path("models/best_model.pth"),
        Path("best_improved_model.pth"),
        Path("best_model.pth")
    ]

    model_loaded = False
    model_info = {}

    for path in model_paths:
        if path.exists():
            try:
                checkpoint = torch.load(path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model_loaded = True

                # Déterminer le type de modèle
                if "improved" in str(path):
                    model_info = {
                        'type': 'improved',
                        'performance': CONFIG['performance'],
                        'path': str(path)
                    }
                    st.success(f"✅ Modèle amélioré chargé: {path.name}")
                else:
                    model_info = {
                        'type': 'baseline',
                        'performance': {'f1_score': 0.615, 'precision': 0.667, 'recall': 0.571, 'grade': 'C (79.9%)'},
                        'path': str(path)
                    }
                    st.info(f"ℹ️ Modèle baseline chargé: {path.name}")
                break
            except Exception as e:
                st.warning(f"⚠️ Erreur lors du chargement de {path}: {e}")
                continue

    if not model_loaded:
        st.error("❌ Aucun modèle trouvé. Assurez-vous qu'un fichier de modèle (.pth) est disponible.")
        st.info("📁 Placez le fichier dans un de ces emplacements: " + ", ".join(str(p) for p in model_paths))
        st.stop()

    model.to(device)
    model.eval()
    return model, device, model_info

@st.cache_data
def preprocess_image(image_array, target_size=(640, 640)):
    """Préprocesser une image pour la prédiction"""
    # Assurer format RGB
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_rgb = image_array
    elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
        image_rgb = image_array[:, :, :3]  # RGBA vers RGB
    else:
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

    # Redimensionner
    image_resized = cv2.resize(image_rgb, target_size)

    # Normalisation
    transform = A.Compose([
        A.Normalize(mean=CONFIG['normalize_mean'], std=CONFIG['normalize_std']),
        ToTensorV2()
    ])

    transformed = transform(image=image_resized)
    image_tensor = transformed['image'].unsqueeze(0)

    return image_tensor, image_resized

def predict_and_visualize(model, device, image, confidence_threshold=0.70):
    """Effectuer prédiction et créer visualisation"""
    start_time = time.time()

    # Préprocessing
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image

    image_tensor, processed_image = preprocess_image(image_array, CONFIG['image_size'])
    image_tensor = image_tensor.to(device)

    # Prédiction
    with torch.no_grad():
        predictions = model(image_tensor)

    # Post-processing
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()

    # Filtrer par confiance
    valid_mask = scores >= confidence_threshold
    filtered_boxes = boxes[valid_mask]
    filtered_scores = scores[valid_mask]

    # Trier par confiance décroissante
    if len(filtered_scores) > 0:
        sort_indices = np.argsort(filtered_scores)[::-1]
        filtered_boxes = filtered_boxes[sort_indices]
        filtered_scores = filtered_scores[sort_indices]

    # Créer image avec détections
    img_with_boxes = processed_image.copy()

    for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
        x1, y1, x2, y2 = box.astype(int)

        # Couleur basée sur la confiance
        if score > 0.8:
            color = (255, 0, 0)  # Rouge pour haute confiance
        elif score > 0.6:
            color = (255, 165, 0)  # Orange pour confiance modérée
        else:
            color = (255, 255, 0)  # Jaune pour confiance faible

        # Dessiner rectangle
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)

        # Label avec confiance
        label = f'Nodule {i+1}: {score:.1%}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        # Fond pour le texte
        cv2.rectangle(img_with_boxes,
                     (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1),
                     color, -1)

        # Texte
        cv2.putText(img_with_boxes, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    processing_time = time.time() - start_time

    return {
        'image_with_boxes': img_with_boxes,
        'count': len(filtered_boxes),
        'scores': filtered_scores,
        'boxes': filtered_boxes,
        'processing_time': processing_time,
        'original_image': processed_image
    }

def create_results_report(results, model_info):
    """Créer un rapport détaillé des résultats"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_info': model_info,
        'detection_results': {
            'count': int(results['count']),
            'processing_time': float(results['processing_time']),
            'detections': []
        }
    }

    if results['count'] > 0:
        for i, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)

            detection = {
                'id': i + 1,
                'confidence': float(score),
                'bounding_box': {
                    'x1': float(x1), 'y1': float(y1),
                    'x2': float(x2), 'y2': float(y2),
                    'area': float(area)
                },
                'risk_level': 'high' if score > 0.8 else 'medium' if score > 0.6 else 'low'
            }
            report['detection_results']['detections'].append(detection)

        # Statistiques globales
        report['detection_results']['statistics'] = {
            'max_confidence': float(np.max(results['scores'])),
            'avg_confidence': float(np.mean(results['scores'])),
            'min_confidence': float(np.min(results['scores']))
        }

    return report

# Interface Streamlit principale
def main():
    # Configuration de la page
    st.set_page_config(
        page_title="🫁 Lung Nodule Detection AI",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS personnalisé
    st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .detection-alert {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-alert {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Titre principal
    st.markdown('<h1 class="main-title">🫁 Lung Nodule Detection AI</h1>', unsafe_allow_html=True)

    # Charger le modèle
    model, device, model_info = load_model()

    # Sous-titre avec performance
    perf = model_info['performance']
    st.markdown(f'<p class="subtitle">🎯 Modèle {perf["grade"]} - F1 Score: {perf["f1_score"]:.1%}</p>', unsafe_allow_html=True)

    # Sidebar avec informations
    with st.sidebar:
        st.header("🤖 Informations du Modèle")

        # Type de modèle
        if model_info['type'] == 'improved':
            st.success("✅ Modèle Amélioré")
        else:
            st.info("ℹ️ Modèle Baseline")

        # Métriques de performance
        st.markdown("### 📊 Performance")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("F1-Score", f"{perf['f1_score']:.1%}")
            st.metric("Precision", f"{perf['precision']:.1%}")
        with col2:
            st.metric("Recall", f"{perf['recall']:.1%}")
            if model_info['type'] == 'improved':
                st.metric("Amélioration", "+33.4%")

        # Configuration
        st.markdown("### ⚙️ Paramètres")
        confidence_threshold = st.slider(
            "Seuil de Confiance",
            min_value=0.1,
            max_value=0.95,
            value=CONFIG['confidence_threshold'],
            step=0.05,
            help="Plus élevé = moins de détections mais plus précises"
        )

        show_confidence = st.checkbox("Afficher scores de confiance", value=True)
        show_details = st.checkbox("Afficher détails techniques", value=False)

        # Informations techniques
        if show_details:
            st.markdown("### 🔧 Détails Techniques")
            st.text(f"Device: {device}")
            st.text(f"Image Size: {CONFIG['image_size']}")
            st.text(f"Modèle: {model_info['path']}")

        # Objectifs du projet
        st.markdown("### 🎯 Objectifs")
        if model_info['type'] == 'improved':
            st.markdown("- ✅ **Recall:** 88.6% (obj: 75%)")
            st.markdown("- ✅ **F1-Score:** 82.1% (obj: 77%)")
            st.markdown("- ⚠️ **Precision:** 76.5% (obj: 80%)")
        else:
            st.markdown("- ❌ **F1-Score:** 61.5% (obj: 77%)")
            st.markdown("- ⚠️ **Precision:** 66.7% (obj: 80%)")
            st.markdown("- ❌ **Recall:** 57.1% (obj: 75%)")

    # Interface principale
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload d'Image")

        # Zone d'upload
        uploaded_file = st.file_uploader(
            "Choisir une image de scanner thoracique",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Formats supportés: JPG, PNG, BMP, TIFF (max 200MB)"
        )

        # Options d'exemple
        st.markdown("### 🖼️ Ou utiliser une image d'exemple")

        # Créer images de démonstration si pas d'exemples
        col_ex1, col_ex2 = st.columns(2)

        with col_ex1:
            if st.button("📸 Exemple 1", help="Image avec nodules"):
                # Créer une image de démonstration
                demo_img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
                # Ajouter quelques cercles pour simuler des nodules
                cv2.circle(demo_img, (200, 150), 20, (255, 255, 255), -1)
                cv2.circle(demo_img, (350, 300), 15, (200, 200, 200), -1)
                st.session_state['demo_image'] = demo_img
                st.success("✅ Exemple 1 chargé")

        with col_ex2:
            if st.button("📸 Exemple 2", help="Image sans nodules"):
                demo_img = np.random.randint(40, 180, (512, 512, 3), dtype=np.uint8)
                st.session_state['demo_image'] = demo_img
                st.success("✅ Exemple 2 chargé")

        # Afficher image chargée
        image_to_analyze = None

        if uploaded_file is not None:
            image_to_analyze = Image.open(uploaded_file)
            st.markdown("### 🖼️ Image Chargée")
            st.image(image_to_analyze, caption=f"📁 {uploaded_file.name}", use_column_width=True)

        elif 'demo_image' in st.session_state:
            image_to_analyze = Image.fromarray(st.session_state['demo_image'])
            st.markdown("### 🖼️ Image d'Exemple")
            st.image(image_to_analyze, caption="📸 Image de démonstration", use_column_width=True)

        # Bouton d'analyse
        if image_to_analyze is not None:
            if st.button("🔍 Analyser l'Image", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyse en cours... Veuillez patienter."):
                    try:
                        results = predict_and_visualize(
                            model, device, image_to_analyze, confidence_threshold
                        )
                        st.session_state['analysis_results'] = results
                        st.session_state['analysis_done'] = True
                        st.success("✅ Analyse terminée!")
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                        st.info("💡 Essayez avec une autre image ou vérifiez le format.")
        else:
            st.info("👆 Veuillez charger une image ou sélectionner un exemple pour commencer l'analyse.")

    with col2:
        st.header("🔍 Résultats de l'Analyse")

        if 'analysis_results' in st.session_state and st.session_state.get('analysis_done', False):
            results = st.session_state['analysis_results']

            # Afficher image avec détections
            st.markdown("### 🎯 Détections")
            st.image(
                results['image_with_boxes'],
                caption="Nodules détectés avec boîtes englobantes",
                use_column_width=True
            )

            # Résumé des résultats
            if results['count'] > 0:
                # Alerte de détection
                max_confidence = np.max(results['scores'])
                avg_confidence = np.mean(results['scores'])

                st.markdown(f"""
                <div class="detection-alert">
                <h3>🚨 {results['count']} Nodule(s) Détecté(s)</h3>
                <p><strong>⏱️ Temps de traitement:</strong> {results['processing_time']:.2f} secondes</p>
                <p><strong>🎯 Seuil utilisé:</strong> {confidence_threshold:.0%}</p>
                <p><strong>🔥 Confiance maximale:</strong> {max_confidence:.1%}</p>
                <p><strong>📊 Confiance moyenne:</strong> {avg_confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)

                # Détails par détection
                st.markdown("### 📋 Détails des Détections")

                for i, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
                    x1, y1, x2, y2 = box.astype(int)
                    area = (x2 - x1) * (y2 - y1)

                    # Niveau de risque
                    if score > 0.8:
                        risk_level = "🔴 Élevé"
                        risk_color = "red"
                    elif score > 0.6:
                        risk_level = "🟠 Modéré"
                        risk_color = "orange"
                    else:
                        risk_level = "🟡 Faible"
                        risk_color = "gold"

                    with st.expander(f"Nodule {i+1} - Confiance: {score:.1%}", expanded=i==0):
                        col_a, col_b, col_c = st.columns(3)

                        with col_a:
                            st.metric("Confiance", f"{score:.1%}")
                            st.metric("Niveau de Risque", risk_level)

                        with col_b:
                            st.metric("Position X", f"{x1} - {x2}")
                            st.metric("Position Y", f"{y1} - {y2}")

                        with col_c:
                            st.metric("Largeur", f"{x2-x1} px")
                            st.metric("Hauteur", f"{y2-y1} px")

                        st.info(f"📐 Aire: {area:,} pixels²")

                # Recommandations
                st.markdown("### 💡 Recommandations")
                if max_confidence > 0.8:
                    st.warning("⚠️ **Détection à haute confiance:** Consultation médicale recommandée pour validation.")
                elif max_confidence > 0.6:
                    st.info("ℹ️ **Détection modérée:** Suivi médical conseillé.")
                else:
                    st.success("✅ **Détection à faible confiance:** Probablement bénin, mais surveillance recommandée.")

            else:
                # Aucune détection
                st.markdown(f"""
                <div class="success-alert">
                <h3>✅ Aucun Nodule Détecté</h3>
                <p>Le modèle n'a pas identifié de nodule suspect avec le seuil de confiance actuel ({confidence_threshold:.0%}).</p>
                <p><strong>⏱️ Temps de traitement:</strong> {results['processing_time']:.2f} secondes</p>
                <p><em>💡 Si vous suspectez la présence de nodules, essayez de réduire le seuil de confiance.</em></p>
                </div>
                """, unsafe_allow_html=True)

            # Export des résultats
            st.markdown("### 📥 Exporter les Résultats")

            col_export1, col_export2 = st.columns(2)

            with col_export1:
                # Rapport JSON
                report = create_results_report(results, model_info)
                json_str = json.dumps(report, indent=2, ensure_ascii=False)

                st.download_button(
                    "📄 Télécharger Rapport JSON",
                    json_str,
                    file_name=f"lung_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Rapport détaillé au format JSON"
                )

            with col_export2:
                # Image annotée
                img_buffer = io.BytesIO()
                Image.fromarray(results['image_with_boxes']).save(img_buffer, format='PNG')

                st.download_button(
                    "🖼️ Télécharger Image Annotée",
                    img_buffer.getvalue(),
                    file_name=f"lung_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    help="Image avec détections annotées"
                )

        else:
            # Interface d'accueil
            st.info("👆 Chargez une image ou sélectionnez un exemple pour commencer l'analyse")

            # Informations sur le modèle
            st.markdown("### 🤖 À propos de ce Modèle")

            if model_info['type'] == 'improved':
                st.success("""
                ✅ **Modèle Amélioré Chargé**

                - **Architecture:** Faster R-CNN avec ResNet-50 FPN
                - **Performance:** A+ (98.5%) - F1 Score: 82.1%
                - **Spécialisation:** Détection de nodules pulmonaires
                - **Optimisations:** Seuil de confiance, augmentation de données, hyperparamètres
                """)
            else:
                st.info("""
                ℹ️ **Modèle Baseline Chargé**

                - **Architecture:** Faster R-CNN avec ResNet-50 FPN
                - **Performance:** C (79.9%) - F1 Score: 61.5%
                - **Spécialisation:** Détection de nodules pulmonaires
                """)

            st.markdown("### 📊 Capacités du Système")
            st.markdown("""
            - 🔍 **Détection automatique** de nodules pulmonaires
            - 📏 **Localisation précise** avec boîtes englobantes
            - 🎯 **Scores de confiance** pour chaque détection
            - ⚡ **Traitement rapide** (< 3 secondes par image)
            - 📱 **Interface intuitive** et accessible
            """)

            st.markdown("### ⚠️ Avertissements Médicaux")
            st.warning("""
            **IMPORTANT:** Cet outil est destiné à des fins de démonstration et de recherche uniquement.

            - 🏥 Ne remplace pas un diagnostic médical professionnel
            - 👨‍⚕️ Consultez toujours un radiologue qualifié
            - 📋 Les résultats doivent être validés cliniquement
            - 🔬 Utilisez comme aide au diagnostic, non comme diagnostic final
            """)

    # Footer informatif
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>🏥 Lung Nodule Detection AI - Version 2.0</strong></p>
        <p>🎓 Projet d'Intelligence Artificielle Médicale</p>
        <p>⚡ Propulsé par PyTorch et Streamlit</p>
        <p style="font-size: 0.8em; margin-top: 10px;">
            <em>Cet outil est développé à des fins éducatives et de recherche.<br>
            Toujours consulter un professionnel de santé pour un diagnostic médical.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
