#!/bin/bash
# Script de lancement de l'application Lung Nodule Detection

echo "🚀 Lancement de l'application Lung Nodule Detection AI"
echo "=" * 50

# Vérifier si streamlit est installé
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit n'est pas installé"
    echo "💡 Installez avec: pip install streamlit"
    exit 1
fi

# Vérifier les dépendances
echo "🔍 Vérification des dépendances..."
python -c "import torch, torchvision, cv2, PIL, albumentations" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Certaines dépendances sont manquantes"
    echo "💡 Installez avec: pip install -r requirements.txt"
    exit 1
fi

echo "✅ Toutes les dépendances sont présentes"

# Lancer l'application
echo "🌐 Lancement de l'application web..."
echo "🔗 L'application sera disponible à: http://localhost:8501"
echo ""

streamlit run app.py --server.port 8501 --server.address localhost
