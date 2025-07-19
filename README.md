# 🫁 Lung Nodule Detection AI

![Version](https://img.shields.io/badge/version-2.0%20Amélioré-blue)
![Python](https://img.shields.io/badge/python-3.9-green)
![Deep Learning](https://img.shields.io/badge/deep%20learning-PyTorch-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## 📋 Description

Un système d'intelligence artificielle basé sur Faster R-CNN pour la détection automatique de nodules pulmonaires dans les images CT (tomodensitométrie). Ce projet utilise des techniques avancées de deep learning pour identifier avec précision les nodules pulmonaires, qui peuvent être des indicateurs précoces de cancer du poumon.

## 🖼️ Démonstration

### Interface utilisateur
![Interface principale](web/test_images/apercu_images_test.png)
*Interface principale de l'application web avec ajustement du seuil de confiance*

### Exemple de détection
![Exemple de détection](test_images_streamlit/test_02_9_jpg.rf.6c2fe24736498530f0d421b484c0b2b7.png)
*Détection d'un nodule pulmonaire avec marquage et niveau de confiance*

### Métriques de performance
![Métriques de performance](results/validation/detection_metrics.png)
*Graphiques des métriques de précision et rappel selon différents seuils de confiance*

## 🚀 Performance du modèle

- **F1-Score**: 82.1%
- **Précision**: 76.5%
- **Rappel**: 88.6%
- **Grade**: A+ (98.5%)
- **Seuil de confiance optimal**: 0.70

## 🗂️ Structure du projet

```
ai-lung-nodule-detection-fastrcnn/
├── data/                   # Données d'entraînement et validation
│   ├── annotations/        # Annotations des nodules (format JSON)
│   ├── processed/          # Images prétraitées
│   └── raw/                # Images CT brutes (non incluses dans le repo)
├── models/                 # Modèles entraînés
│   ├── improved/           # Modèles améliorés avec métriques
│   └── README.md           # Instructions pour télécharger les modèles
├── notebooks/              # Notebooks Jupyter pour l'analyse et l'entraînement
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_inference.ipynb
│   ├── 05_model_validation.ipynb
│   ├── 06_web_interface.ipynb
│   └── 07_model_improvement.ipynb
├── results/                # Résultats d'évaluation
│   └── validation/         # Rapports de validation et métriques
├── web/                    # Interface web Streamlit
│   ├── lung_detection_app.py  # Application principale
│   └── test_images/        # Images de test pour la démonstration
└── requirements.txt        # Dépendances du projet
```

## 💻 Installation

### Prérequis
- Python 3.9+
- CUDA compatible GPU (recommandé pour l'inférence rapide)

### Configuration

1. Cloner le repository
```bash
git clone https://github.com/votre-username/ai-lung-nodule-detection-fastrcnn.git
cd ai-lung-nodule-detection-fastrcnn
```

2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. Installer les dépendances
```bash
pip install -r requirements.txt
```

4. Télécharger les modèles pré-entraînés

Les fichiers de modèles (.pth) ne sont pas inclus dans ce dépôt en raison de leur taille. Vous pouvez les télécharger via ce lien:
[Télécharger les modèles pré-entraînés](https://huggingface.co/models/your-username/lung-nodule-detection)

Placez les fichiers téléchargés dans le répertoire `models/improved/`.

## 🔧 Utilisation

### Interface web

L'interface web permet une utilisation simple et intuitive du modèle:

```bash
cd web
streamlit run lung_detection_app.py
```

L'application sera accessible à l'adresse: http://localhost:8501

### Utilisation via notebooks

Les notebooks Jupyter dans le dossier `notebooks/` fournissent des exemples détaillés pour:
- Explorer les données
- Prétraiter les images CT
- Entraîner le modèle Faster R-CNN
- Effectuer des inférences
- Valider les performances
- Améliorer le modèle

## 🧠 Architecture du modèle

Ce projet utilise un modèle Faster R-CNN avec backbone ResNet-50 FPN (Feature Pyramid Network), adapté pour la détection d'objets de tailles variées. L'architecture a été optimisée pour détecter des nodules pulmonaires qui peuvent être très petits et subtils dans les images CT.

Caractéristiques principales:
- Backbone: ResNet-50
- Taille d'image d'entrée: 640×640 pixels
- Classes: 2 (nodule, arrière-plan)
- Augmentation de données: rotations, zoom, modifications de contraste

## 📊 Évaluation et métriques

Les performances du modèle ont été évaluées sur un ensemble de données de validation indépendant. Les métriques clés incluent:

- **Précision vs. Seuil de confiance**: Mesure la précision des prédictions à différents seuils
- **Rappel vs. Seuil de confiance**: Évalue la capacité du modèle à détecter tous les nodules
- **F1-Score vs. Seuil de confiance**: Balance entre précision et rappel
- **Courbes Précision-Rappel**: Représentation du compromis précision-rappel

Ces métriques sont disponibles dans le dossier `results/validation/`.

## 📚 Citation

Si vous utilisez ce projet dans votre recherche ou application, veuillez le citer comme suit:

```
@software{lung_nodule_detection_ai,
  author = {Votre Nom},
  title = {Lung Nodule Detection AI: Fast R-CNN pour la détection automatique de nodules pulmonaires},
  year = {2025},
  url = {https://github.com/votre-username/ai-lung-nodule-detection-fastrcnn}
}
```

## 📝 Licence

Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## ✉️ Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur ce repository ou à me contacter directement à [votre-email@example.com].
