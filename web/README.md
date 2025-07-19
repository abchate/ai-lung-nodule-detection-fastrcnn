# 🫁 Lung Nodule Detection AI - Interface Web

## 🎯 Description
Interface web interactive pour la détection automatique de nodules pulmonaires utilisant un modèle Faster R-CNN optimisé.

## 🏆 Performance du Modèle
- **Grade:** A+ (98.5%)
- **F1-Score:** 82.1%
- **Precision:** 76.5%
- **Recall:** 88.6%

## 🚀 Installation et Lancement

### 1. Prérequis
```bash
Python 3.8+
pip ou conda
```

### 2. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancement de l'application
```bash
# Méthode 1: Script de lancement (recommandé)
./launch.sh

# Méthode 2: Commande directe
streamlit run app.py

# Méthode 3: Port personnalisé
streamlit run app.py --server.port 8502
```

### 4. Accès à l'application
Ouvrez votre navigateur et allez à: `http://localhost:8501`

## 📁 Structure des Fichiers
```
web/
├── app.py                 # Application Streamlit principale
├── requirements.txt       # Dépendances Python
├── launch.sh             # Script de lancement
└── README.md             # Ce fichier
```

## 🔧 Utilisation

### Upload d'Image
1. Cliquez sur "Choisir une image de scanner thoracique"
2. Sélectionnez un fichier JPG, PNG, BMP ou TIFF
3. Ou utilisez les exemples de démonstration

### Configuration
- **Seuil de Confiance:** Ajustez dans la sidebar (0.1 à 0.95)
- **Affichage:** Options pour les scores et détails techniques

### Analyse
1. Cliquez sur "🔍 Analyser l'Image"
2. Attendez le traitement (< 3 secondes)
3. Visualisez les résultats avec détections annotées

### Export
- **Rapport JSON:** Données détaillées des détections
- **Image Annotée:** Image avec boîtes englobantes

## 🎯 Fonctionnalités

### Interface Utilisateur
- ✅ Design responsive et intuitif
- ✅ Sidebar avec informations du modèle
- ✅ Upload par glisser-déposer
- ✅ Images d'exemple intégrées
- ✅ Visualisation en temps réel

### Détection Intelligente
- ✅ Seuil de confiance ajustable
- ✅ Codes couleur selon le niveau de risque
- ✅ Statistiques détaillées par détection
- ✅ Métriques de performance en temps réel

### Export et Reporting
- ✅ Rapport JSON structuré
- ✅ Images annotées téléchargeables
- ✅ Timestamps et métadonnées
- ✅ Informations techniques

## ⚠️ Avertissements Médicaux

**IMPORTANT:** Cet outil est destiné à des fins de démonstration et de recherche uniquement.

- 🏥 Ne remplace pas un diagnostic médical professionnel
- 👨‍⚕️ Consultez toujours un radiologue qualifié
- 📋 Les résultats doivent être validés cliniquement
- 🔬 Utilisez comme aide au diagnostic, non comme diagnostic final

## 🐛 Dépannage

### Problèmes courants

**1. Erreur "Aucun modèle trouvé"**
```
Solution: Placez le fichier best_improved_model.pth dans:
- ../models/improved/
- models/improved/
- ./
```

**2. Erreur de dépendances**
```bash
pip install --upgrade -r requirements.txt
```

**3. Problème de performance**
```
- Vérifiez la disponibilité du GPU
- Réduisez la taille des images
- Fermez les autres applications
```

**4. Interface ne se charge pas**
```bash
# Vérifiez le port
netstat -an | grep 8501

# Essayez un autre port
streamlit run app.py --server.port 8502
```

## 📊 Support Technique

### Formats d'images supportés
- **JPG/JPEG:** Recommandé pour scanners
- **PNG:** Haute qualité
- **BMP:** Format natif
- **TIFF:** Format médical standard

### Spécifications recommandées
- **Résolution:** 512x512 à 2048x2048 pixels
- **Taille:** < 50 MB par image
- **Format:** RGB ou Grayscale
- **Qualité:** Non compressé de préférence

## 🔄 Mises à jour

Pour mettre à jour l'application:
```bash
git pull origin main
pip install --upgrade -r requirements.txt
```

## 📞 Contact

Pour support technique ou questions:
- 📧 Projet d'IA Médicale
- 🐛 Rapporter un bug: Créez une issue
- 💡 Suggestions d'améliorations: Bienvenues!

---

**🎓 Développé dans le cadre d'un projet d'Intelligence Artificielle Médicale**
