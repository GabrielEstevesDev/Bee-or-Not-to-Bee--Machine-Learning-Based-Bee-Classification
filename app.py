import pandas as pd
import joblib
import os
import gradio as gr
from fastapi import FastAPI
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from pathlib import Path

# 1. Initialisation de l'application FastAPI
app = FastAPI()

# --- CONFIGURATION ---
MODEL_PATH = "logistic_regression_model.pkl"
IMG_DIR = "img"       # Dossier contenant vos images JPG
MASK_DIR = "masks"    # Dossier contenant vos masques TIF

# Liste EXACTE des 26 features attendues par le modèle
FEATURE_COLUMNS = [
    'area_ratio', 'mask_pixels', 'roundness', 'ellipse_fit_quality', 
    'vertical_symmetry', 'horizontal_symmetry', 'min_R', 'mean_R', 
    'std_R', 'min_G', 'mean_G', 'std_G', 'med_G', 'mean_B', 'std_B', 
    'med_B', 'lbp_0', 'lbp_2', 'lbp_3', 'lbp_4', 'lbp_5', 'lbp_6', 
    'lbp_7', 'lbp_8', 'lbp_9', 'edge_density'
]

# Paramètres d'extraction
LBP_RADIUS   = 1
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD   = 'uniform'
EDGE_THRESH  = 30

model = None
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Modèle chargé avec succès.")
except Exception as e:
    print(f"❌ ERREUR CRITIQUE : Impossible de charger le modèle. {e}")

# --- FONCTIONS UTILITAIRES ---

def create_auto_mask(img_bgr):
    """Génère un masque automatiquement si le TIF est absent."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(gray)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [main_contour], -1, 255, thickness=cv2.FILLED)
    return final_mask

def extract_features_full(image_path):
    """
    Extrait les features et retourne aussi un message sur l'origine du masque.
    Retourne: (features, status_message, error_message)
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None, "Impossible de lire le fichier image."
    
    mask = None
    mask_source_msg = "" 
    
    filename = os.path.basename(image_path)
    try:
        potential_id = filename.split('.')[0].split('-')[0]
        potential_mask_path = os.path.join(MASK_DIR, f"binary_{potential_id}.tif")
        
        if os.path.exists(potential_mask_path):
            mask = cv2.imread(potential_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask_source_msg = f"✅ Masque TIF original trouvé (binary_{potential_id}.tif)."
    except Exception:
        pass 

    if mask is None:
        mask = create_auto_mask(img)
        mask_source_msg = "⚠️ Pas de masque TIF trouvé. Masque généré automatiquement."

    if mask is None or mask.sum() == 0:
        return None, None, "Échec de la création du masque."

    # --- Extraction des Features ---
    feats = {}
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bug_mask = mask == 255
    bug_pix = img_rgb[bug_mask]
    
    if bug_pix.size == 0:
        return None, None, "Le masque ne couvre aucun pixel."

    # Forme
    mask_pixels = bug_mask.sum()
    total_pixels = mask.size
    feats['area_ratio'] = mask_pixels / total_pixels
    feats['mask_pixels'] = int(mask_pixels)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        feats['roundness'] = min((4 * np.pi * area) / (perimeter ** 2), 1.0) if area > 0 and perimeter > 0 else 0
            
        if len(main_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(main_contour)
                (_, axes, _) = ellipse
                major, minor = max(axes), min(axes)
                ellipse_area = np.pi * (major/2) * (minor/2)
                feats['ellipse_fit_quality'] = min(area / ellipse_area, 1.0) if ellipse_area > 0 else 0
            except:
                feats['ellipse_fit_quality'] = 0
        else:
            feats['ellipse_fit_quality'] = 0
            
        rect = cv2.boundingRect(main_contour)
        x, y, w, h = rect
        
        # Symétrie Verticale
        cx = x + w // 2
        l = mask[y:y+h, x:cx]
        r = mask[y:y+h, cx:x+w]
        r = np.fliplr(r)
        min_w = min(l.shape[1], r.shape[1])
        feats['vertical_symmetry'] = np.logical_and(l[:, :min_w], r[:, :min_w]).sum() / np.logical_or(l[:, :min_w], r[:, :min_w]).sum() if min_w > 0 else 0
            
        # Symétrie Horizontale
        cy = y + h // 2
        t = mask[y:cy, x:x+w]
        b = mask[cy:y+h, x:x+w]
        b = np.flipud(b)
        min_h = min(t.shape[0], b.shape[0])
        feats['horizontal_symmetry'] = np.logical_and(t[:min_h, :], b[:min_h, :]).sum() / np.logical_or(t[:min_h, :], b[:min_h, :]).sum() if min_h > 0 else 0
    else:
        feats.update({'roundness': 0, 'ellipse_fit_quality': 0, 'vertical_symmetry': 0, 'horizontal_symmetry': 0})

    # Couleur
    for i, c in enumerate(['R', 'G', 'B']):
        channel = bug_pix[:, i]
        feats[f'min_{c}'] = float(channel.min())
        feats[f'mean_{c}'] = float(channel.mean())
        feats[f'std_{c}'] = float(channel.std())
        feats[f'med_{c}'] = float(np.median(channel))

    # Texture
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, LBP_METHOD)
    hist, _ = np.histogram(lbp[bug_mask], bins=np.arange(0, LBP_N_POINTS + 3), density=True)
    for i, v in enumerate(hist):
        feats[f'lbp_{i}'] = float(v)

    # Edges
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(sx, sy)
    feats['edge_density'] = float((mag[bug_mask] > EDGE_THRESH).mean())

    try:
        features_vector = [feats[col] for col in FEATURE_COLUMNS]
        return np.array([features_vector]), mask_source_msg, None
    except KeyError as e:
        return None, None, f"Erreur interne : Feature manquante '{e}'."


def predict_insect(image_path):
    if model is None:
        return "Erreur : Modèle non chargé."
    
    features, status_msg, error = extract_features_full(image_path)
    
    if error:
        return f"Erreur : {error}"
    
    try:
        pred = model.predict(features)[0]
        probs = model.predict_proba(features)[0]
    except Exception as e:
        return f"Erreur lors de la prédiction : {e}"
    
    res = f"{status_msg}\n\n"
    res += f"**Résultat Prédit : {pred}**\n\nConfiance par classe :"
    
    sorted_probs = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)
    for cls, p in sorted_probs:
        res += f"\n- {cls}: {p*100:.1f}%"
        
    return res

# --- INTERFACE GRADIO ---

example_files = []
if os.path.exists(IMG_DIR):
    files = sorted(os.listdir(IMG_DIR))
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            example_files.append(os.path.join(IMG_DIR, f))

with gr.Blocks() as demo:
    gr.Markdown("# 🐞 Classification d'Insectes")
    gr.Markdown("Cliquez sur un exemple ci-dessous ou uploadez votre propre image.")
    
    with gr.Row():
        with gr.Column():
            # 1. Préparation de l'image (cachée)
            input_img = gr.Image(type="filepath", label="Votre Image", height=400, render=False)
            
            # 2. Affichage des exemples en PREMIER
            if example_files:
                # Calcul du nombre total d'images pour les mettre toutes sur la même "page"
                total_examples = len(example_files)
                
                gr.Examples(
                    examples=example_files,
                    inputs=input_img,
                    label="👇 SÉLECTION RAPIDE (Images du Dataset)",
                    examples_per_page=total_examples # <--- C'EST ICI LA CLÉ (affiche tout sans pagination)
                )
            
            # 3. Affichage de l'upload en DESSOUS
            input_img.render()
            
            btn = gr.Button("Analyser & Prédire", variant="primary")
                
        with gr.Column():
            output_txt = gr.Textbox(label="Résultat de l'analyse", lines=12)

    btn.click(fn=predict_insect, inputs=input_img, outputs=output_txt)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)