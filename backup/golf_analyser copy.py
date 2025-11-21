# Importation des bibliothèques nécessaires
import cv2  # OpenCV pour le traitement vidéo
import mediapipe as mp  # MediaPipe pour la détection de pose
import os  # OS pour la gestion des fichiers
# Initialisation des modules MediaPipe pour la détection de pose    
mp_pose = mp.solutions.pose  # Module de détection de pose
pose = mp_pose.Pose()  # Création de l'objet de détection de pose
mp_draw = mp.solutions.drawing_utils  # Outils de dessin pour les landmarks
videoPath = "/Users/joequenneville/Movies/OBS/Replay 2025-11-10 22-35-59.mp4"  # Chemin de la vidéo à analyser
# Ouverture de la vidéo
cap = cv2.VideoCapture(videoPath)  # Chargement de la vidéo

# Préparation du writer pour sauvegarder la vidéo traitée
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec vidéo H.264 pour meilleure compatibilité QuickTime
fps = cap.get(cv2.CAP_PROP_FPS)  # Images par seconde
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Largeur de la vidéo
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Hauteur de la vidéo
base, ext = os.path.splitext(videoPath)  # Séparation du nom et de l'extension
outputPath = f"{base}_AI{ext}"  # Chemin de sortie pour la vidéo annotée
out = cv2.VideoWriter(outputPath, fourcc, fps, (width, height))  # Initialisation du writer

# Boucle de traitement des frames vidéo 
while cap.isOpened():  # Tant que la vidéo est ouverte
    success, frame = cap.read()  # Lecture d'une frame
    if not success:  # Si la lecture échoue, on quitte la boucle
        break
    # Conversion de l'image de BGR à RGB pour MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Conversion des couleurs
    # Traitement de l'image pour détecter les poses
    result = pose.process(rgb)  # Détection des landmarks
    # Si des landmarks de pose sont détectés, les dessiner sur la frame
    if result.pose_landmarks:
        # Dessin des landmarks de pose sur la frame
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # Affichage de la frame avec les annotations
    cv2.imshow("Analyse swing", frame)  # Affichage de la vidéo annotée
    # Sauvegarde de la frame annotée dans le fichier de sortie
    out.write(frame)  # Écriture de la frame dans le fichier MP4
    # Sortie de la boucle si la touche 'Esc' est pressée
    if cv2.waitKey(1) & 0xFF == 27:
        break
# Libération des ressources
cap.release()  # Fermeture de la vidéo
out.release()  # Fermeture du fichier vidéo de sortie
# Fermeture de toutes les fenêtres OpenCV
cv2.destroyAllWindows()  # Fermeture des fenêtres d'affichage
