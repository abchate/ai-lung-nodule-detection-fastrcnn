# 🫁 Lung Nodule Detection Project

## 📊 Résultats d'Entraînement

- **Modèle**: Faster R-CNN (ResNet50 + FPN)
- **Dataset**: Lung Nodule Detection (format YOLO → COCO)
- **Époques**: 10
- **Meilleur Validation Loss**: 0.1188
- **Temps d'entraînement**: 73.7 minutes (CPU)

## 🎯 Performance

- **Loss final**: 0.1428 (train) | 0.1188 (val)
- **Amélioration**: 80.1%
- **Qualité**: Très bon modèle (loss < 0.2)

## 🏥 Capacités de Détection

- Détecte plusieurs nodules par image
- Confidence typique: 30-80%+
- Système médical complet avec rapports
- Interface de visualisation intégrée

## 📁 Fichiers Générés

```
../models/
├── best_model.pth          # Meilleur modèle entraîné
├── training_config.json    # Configuration complète
├── training_losses.json    # Historique des losses
└── README.md              # Cette documentation
```

## 🚀 Utilisation

```python
# Charger le modèle
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Utiliser le système de détection
detector = LungNoduleDetector(model, CONFIG, device)
results = detector.detect_nodules(image_path)
detector.generate_report(results)
```

## ⚠️ Important

Ce système est un outil d'aide au diagnostic.
L'interprétation finale doit toujours être faite par un médecin qualifié.
