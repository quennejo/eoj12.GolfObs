# Importation des bibliothèques nécessaires
import cv2  # OpenCV pour le traitement vidéo
import mediapipe as mp  # MediaPipe pour la détection de pose
import os  # OS pour la gestion des fichiers
from ultralytics import YOLO  # YOLO pour la détection de la balle de golf
from ultralytics import YOLOWorld  # YOLO-World pour la détection open-vocabulary
import numpy as np  # NumPy pour les opérations sur les matrices
import ssl
import urllib.request

# Désactiver la vérification SSL (pour résoudre l'erreur de certificat)
ssl._create_default_https_context = ssl._create_unverified_context

# Initialisation des modules MediaPipe pour la détection de pose    
mp_pose = mp.solutions.pose  # Module de détection de pose
pose = mp_pose.Pose()  # Création de l'objet de détection de pose
mp_draw = mp.solutions.drawing_utils  # Outils de dessin pour les landmarks
videoPath = "/Users/joequenneville/Movies/OBS/Replay 2025-11-10 22-35-59.mp4"  # Chemin de la vidéo à analyser
#videoPath = "/Users/joequenneville/Movies/OBS/Replay 2025-11-10 22-31-29.mp4"
#videoPath = "/Users/joequenneville/Movies/OBS/Replay 2025-11-10 22-16-09.mp4"
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

# Définition du nombre maximum de frames à garder
max_frames = 300  # Nombre de frames à garder après le début du swing
frame_count = 0

# Initialisation du détecteur YOLO pour la balle de golf
class YoloGolfBallDetector:
    """
    Détecteur de balle de golf utilisant YOLOv8.
    Nécessite un modèle YOLO entraîné pour la balle de golf (personnalisé ou adapté).
    """
    def __init__(self, model_path='yolov8n.pt', ball_class_name='golf ball'):
        self.model = YOLO(model_path)
        self.ball_class_name = ball_class_name

    def detect_ball(self, frame):
        results = self.model(frame)
        ball_boxes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = r.names[cls]
                if name == self.ball_class_name:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    ball_boxes.append((x1, y1, x2, y2))
        return ball_boxes

# Initialisation du détecteur YOLO-World pour le bâton de golf
class YoloWorldGolfClubDetector:
    """
    Détecteur de bâton de golf utilisant YOLO-World (open-vocabulary).
    Peut détecter "golf club" sans entraînement spécifique.
    """
    def __init__(self, model_path='yolov8s-worldv2.pt'):
        self.model = YOLOWorld(model_path)
        # Définir les classes à détecter
        self.model.set_classes(["golf club", "golf stick"])
    
    def detect_club(self, frame):
        results = self.model(frame, conf=0.3)
        club_boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                club_boxes.append((x1, y1, x2, y2))
        return club_boxes

# Initialisation du détecteur de balle de golf
# Correction : utiliser le nom de classe YOLO 'sports ball' au lieu de 'golf ball'
#yolo_ball_detector = YoloGolfBallDetector(model_path='yolov8n.pt', ball_class_name='sports ball')

# Initialisation du détecteur de bâton de golf avec YOLO-World
yolo_club_detector = YoloWorldGolfClubDetector(model_path='yolov8s-worldv2.pt')

ball_point = None
shoulder_point = None
ball_detected = False
club_line = None  # Pour stocker la ligne du club détecté
club_length = 450  # Longueur estimée du club en pixels
club_trajectory = []  # Trajectoire de la tête du club (historique des positions)
club_trajectory_raw = []  # Trajectoire brute avant filtrage
previous_club_head = None  # Position précédente de la tête du club
max_movement_per_frame = 150  # Déplacement maximum autorisé entre deux frames (en pixels)

# Boucle de traitement des frames vidéo 
while cap.isOpened():  # Tant que la vidéo est ouverte
    success, frame = cap.read()  # Lecture d'une frame
    if not success or frame_count >= max_frames:  # Si la lecture échoue ou si le nombre de frames est atteint, on quitte la boucle
        break
    frame_count += 1
    # Conversion de l'image de BGR à RGB pour MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Conversion des couleurs
    # Traitement de l'image pour détecter les poses
    result = pose.process(rgb)  # Détection des landmarks

    # Détection de la balle de golf avec YOLO au début du swing
    if ball_detected ==False and frame_count < 100:
        #ball_boxes = yolo_ball_detector.detect_ball(frame)
        
        # Si YOLO ne trouve pas la balle, essayer HoughCircles
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                    param1=50, param2=30, minRadius=5, maxRadius=30)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                bx, by, radius = c
                # Vérifier que le centre est sur une zone verte
                patch_size = 10
                x1g = max(bx - patch_size, 0)
                y1g = max(by - patch_size, 0)
                x2g = min(bx + patch_size, frame.shape[1])
                y2g = min(by + patch_size, frame.shape[0])
                patch = frame[y1g:y2g, x1g:x2g]
                hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                mask_green = cv2.inRange(hsv_patch, np.array([35, 40, 40]), np.array([85, 255, 255]))
                green_ratio = cv2.countNonZero(mask_green) / (patch.shape[0] * patch.shape[1])
                # Vérifier aussi que le centre est blanc
                mask_white = cv2.inRange(hsv_patch, np.array([0, 0, 180]), np.array([180, 50, 255]))
                white_ratio = cv2.countNonZero(mask_white) / (patch.shape[0] * patch.shape[1])
                if green_ratio > 0.5 and white_ratio > 0.3 and bx> 400 and bx <850:
                    ball_point = (bx, by)
                    ball_point= (ball_point[0]+10, ball_point[1]+10) # ajustement de la position de la balle
                    ball_detected = True
                    break
            else:
                ball_point = None
                ball_detected = False
        else:
            ball_point = None
            ball_detected = False
   # club_boxes = yolo_club_detector.detect_club(frame)         
    #if club_boxes:
        # Prendre le premier bâton détecté et calculer le centre bas
        #x1, y1, x2, y2 = club_boxes[0]
        # Dessiner un rectangle noir autour du bâton de golf détecté
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)    
    if frame_count == 1:
        # Détection du poignet et du pouce gauche
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            h, w, _ = frame.shape
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            left_thumb = landmarks[mp_pose.PoseLandmark.LEFT_THUMB]
            left_wrist_pos = (int(left_wrist.x * w), int(left_wrist.y * h))
            left_thumb_pos = (int(left_thumb.x * w), int(left_thumb.y * h))
        else:
            shoulder_point = None
        
        # Détection du bâton de golf avec YOLO-World à la frame 1
        club_boxes = yolo_club_detector.detect_club(frame)
        if club_boxes:
            x1, y1, x2, y2 = club_boxes[0]
            club_bottom = (x1, y2+40)  # Centre bas du rectangle
            # Calculer la longueur du club à partir du pouce gauche
            club_length = int(np.sqrt((club_bottom[0] - left_thumb_pos[0])**2 + (club_bottom[1] - left_thumb_pos[1])**2))
            club_line = (left_thumb_pos, club_bottom)  # Ligne du pouce gauche au bas du club 
    
    # Extrapolation du club à partir de plusieurs points des mains et poignets
    if result.pose_landmarks and club_length > 0:
        landmarks = result.pose_landmarks.landmark
        h, w, _ = frame.shape
        
        # Récupérer tous les points des mains et poignets
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        left_thumb = landmarks[mp_pose.PoseLandmark.LEFT_THUMB]
        left_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
        left_pinky = landmarks[mp_pose.PoseLandmark.LEFT_PINKY]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        right_thumb = landmarks[mp_pose.PoseLandmark.RIGHT_THUMB]
        right_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]
        right_pinky = landmarks[mp_pose.PoseLandmark.RIGHT_PINKY]
        
        # Convertir en coordonnées pixel
        left_wrist_pos = np.array([left_wrist.x * w, left_wrist.y * h])
        left_thumb_pos = np.array([left_thumb.x * w, left_thumb.y * h])
        left_index_pos = np.array([left_index.x * w, left_index.y * h])
        left_pinky_pos = np.array([left_pinky.x * w, left_pinky.y * h])
        right_wrist_pos = np.array([right_wrist.x * w, right_wrist.y * h])
        right_thumb_pos = np.array([right_thumb.x * w, right_thumb.y * h])
        right_index_pos = np.array([right_index.x * w, right_index.y * h])
        right_pinky_pos = np.array([right_pinky.x * w, right_pinky.y * h])
        
        # Calculer le centre moyen des mains (grip du club)
        # On donne plus de poids aux pouces et index qui tiennent le club
        hand_points = []
        weights = []
        
        # Main gauche (poids plus élevés pour pouce et index)
        if left_thumb.visibility > 0.5:
            hand_points.append(left_thumb_pos)
            weights.append(2.0)
        if left_index.visibility > 0.5:
            hand_points.append(left_index_pos)
            weights.append(1.5)
        if left_wrist.visibility > 0.5:
            hand_points.append(left_wrist_pos)
            weights.append(1.0)
        if left_pinky.visibility > 0.5:
            hand_points.append(left_pinky_pos)
            weights.append(0.8)
        
        # Main droite
        if right_thumb.visibility > 0.5:
            hand_points.append(right_thumb_pos)
            weights.append(2.0)
        if right_index.visibility > 0.5:
            hand_points.append(right_index_pos)
            weights.append(1.5)
        if right_wrist.visibility > 0.5:
            hand_points.append(right_wrist_pos)
            weights.append(1.0)
        if right_pinky.visibility > 0.5:
            hand_points.append(right_pinky_pos)
            weights.append(0.8)
        
        if len(hand_points) > 0:
            # Calculer le centre pondéré des mains (grip)
            hand_points = np.array(hand_points)
            weights = np.array(weights)
            grip_center = np.average(hand_points, axis=0, weights=weights)
            
            # Calculer la direction principale en utilisant tous les points disponibles
            # On utilise les pouces comme point de référence pour la direction
            if left_thumb.visibility > 0.5 and right_thumb.visibility > 0.5:
                # Direction basée sur la moyenne des deux pouces
                thumb_center = (left_thumb_pos + right_thumb_pos) / 2
            elif left_thumb.visibility > 0.5:
                thumb_center = left_thumb_pos
            elif right_thumb.visibility > 0.5:
                thumb_center = right_thumb_pos
            else:
                # Fallback: utiliser le centre du grip
                thumb_center = grip_center
            
            # Calculer la direction du club
            # On prend en compte la direction des poignets vers les pouces
            wrist_center = (left_wrist_pos + right_wrist_pos) / 2
            dx = thumb_center[0] - wrist_center[0]
            dy = thumb_center[1] - wrist_center[1]
            angle = np.arctan2(dy, dx)
            
            # Extrapoler la position de la tête du club à partir du centre des pouces
            club_head_x = int(thumb_center[0] + club_length * np.cos(angle))
            club_head_y = int(thumb_center[1] + club_length * np.sin(angle))
            club_head = (club_head_x, club_head_y)
            
            # Point de départ du club (centre du grip)
            grip_start = (int(grip_center[0]), int(grip_center[1]))
        else:
            # Fallback si aucun point n'est visible
            club_head = None
            grip_start = None
        
        # Mettre à jour la ligne du club si les points sont valides
        if club_head is not None and grip_start is not None:
            # Contrôle d'erreur : vérifier que le mouvement n'est pas trop important
            is_valid = True
            if previous_club_head is not None:
                # Calculer la distance entre la position actuelle et la précédente
                distance = np.sqrt((club_head[0] - previous_club_head[0])**2 + 
                                 (club_head[1] - previous_club_head[1])**2)
                
                # Si le mouvement est trop important, c'est probablement une erreur
                if distance > max_movement_per_frame:
                    is_valid = False
                    # Utiliser la position précédente au lieu de la nouvelle
                    club_head = previous_club_head
            
            if is_valid:
                previous_club_head = club_head
            
            club_line = (grip_start, club_head)
            
            # Ajouter la position de la tête du club à la trajectoire brute
            club_trajectory_raw.append(club_head)
        else:
            # Si aucun point n'est visible, ne pas ajouter à la trajectoire
            if previous_club_head is not None:
                club_trajectory_raw.append(previous_club_head)  # Répéter la dernière position
        
        # Limiter la taille de la trajectoire brute
        if len(club_trajectory_raw) > 50:
            club_trajectory_raw.pop(0)
        
        # Appliquer un lissage (moyenne mobile) sur la trajectoire
        # On utilise une fenêtre glissante pour lisser
        window_size = 5  # Taille de la fenêtre de lissage
        if len(club_trajectory_raw) >= window_size:
            # Calculer la moyenne des dernières positions
            recent_points = club_trajectory_raw[-window_size:]
            smoothed_x = int(np.mean([p[0] for p in recent_points]))
            smoothed_y = int(np.mean([p[1] for p in recent_points]))
            smoothed_point = (smoothed_x, smoothed_y)
            club_trajectory.append(smoothed_point)
        elif len(club_trajectory_raw) > 0:
            # Si pas assez de points pour lisser, utiliser le point brut
            club_trajectory.append(club_trajectory_raw[-1])
        
        # Limiter la taille de la trajectoire lissée (garder les 50 dernières positions)
        if len(club_trajectory) > 50:
            club_trajectory.pop(0)
        
        # Dessiner la trajectoire du club (traînée verte lissée)
        if len(club_trajectory) > 1:
            # Dessiner une courbe lissée en utilisant cv2.polylines pour une belle courbe
            points = np.array(club_trajectory, dtype=np.int32)
            
            # Dessiner la trajectoire avec un dégradé d'épaisseur
            for i in range(1, len(club_trajectory)):
                if club_trajectory[i-1] is None or club_trajectory[i] is None:
                    continue
                # Épaisseur qui augmente vers les positions récentes
                thickness = int(1 + (i / len(club_trajectory)) * 4)
                # Opacité qui augmente aussi (simulée par l'épaisseur)
                cv2.line(frame, club_trajectory[i-1], club_trajectory[i], (0, 255, 0), thickness)

    # Tracer la ligne rouge fixe entre la balle et les mains (au lieu de l'épaule gauche)
    if ball_detected and ball_point and result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        h, w, _ = frame.shape
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_hand = (int(left_wrist.x * w), int(left_wrist.y * h))
        right_hand = (int(right_wrist.x * w), int(right_wrist.y * h))
        # Milieu des mains
        hands_mid = (int((left_hand[0] + right_hand[0]) / 2), int((left_hand[1] + right_hand[1]) / 2))
        
        #cv2.circle(frame, ball_point, 8, (0,255,0), 2)  # cercle vert sur la balle détectée
        cv2.circle(frame, ball_point, 16, (0,0,0), 3)  # cercle noir autour de la balle
        #cv2.line(frame, ball_point, hands_mid, (0, 0, 255), 4)
    
    # Dessiner la ligne bleue du bâton de golf détecté avec YOLO-World
    if club_line is not None:
        cv2.line(frame, club_line[0], club_line[1], (255, 0, 0), 3)  # ligne bleue pour le bâton

    # Afficher tous les landmarks du squelette avec des couleurs différentes
    if result.pose_landmarks:
        # Dessiner toutes les connexions du squelette
        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=0),  # Pas de cercles sur les connections
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=0)  # Lignes vertes
        )
        
        landmarks = result.pose_landmarks.landmark
        h, w, _ = frame.shape
        
        # Dictionnaire de couleurs pour chaque landmark (BGR format)
        landmark_colors = {
            # Visage
            mp_pose.PoseLandmark.NOSE: (255, 0, 0),  # Bleu
            mp_pose.PoseLandmark.LEFT_EYE_INNER: (255, 100, 100),
            mp_pose.PoseLandmark.LEFT_EYE: (255, 150, 150),
            mp_pose.PoseLandmark.LEFT_EYE_OUTER: (255, 200, 200),
            mp_pose.PoseLandmark.RIGHT_EYE_INNER: (100, 100, 255),
            mp_pose.PoseLandmark.RIGHT_EYE: (150, 150, 255),
            mp_pose.PoseLandmark.RIGHT_EYE_OUTER: (200, 200, 255),
            mp_pose.PoseLandmark.LEFT_EAR: (255, 0, 255),  # Magenta
            mp_pose.PoseLandmark.RIGHT_EAR: (255, 0, 128),
            mp_pose.PoseLandmark.MOUTH_LEFT: (128, 0, 255),
            mp_pose.PoseLandmark.MOUTH_RIGHT: (200, 0, 200),
            
            # Épaules
            mp_pose.PoseLandmark.LEFT_SHOULDER: (0, 255, 255),  # Cyan
            mp_pose.PoseLandmark.RIGHT_SHOULDER: (0, 200, 255),
            
            # Coudes
            mp_pose.PoseLandmark.LEFT_ELBOW: (0, 255, 128),  # Vert-cyan
            mp_pose.PoseLandmark.RIGHT_ELBOW: (0, 200, 100),
            
            # Poignets
            mp_pose.PoseLandmark.LEFT_WRIST: (203, 192, 255),  # Rose (comme demandé)
            mp_pose.PoseLandmark.RIGHT_WRIST: (180, 150, 255),
            
            # Mains
            mp_pose.PoseLandmark.LEFT_PINKY: (255, 128, 0),  # Orange
            mp_pose.PoseLandmark.RIGHT_PINKY: (255, 100, 0),
            mp_pose.PoseLandmark.LEFT_INDEX: (255, 200, 0),  # Jaune-orange
            mp_pose.PoseLandmark.RIGHT_INDEX: (255, 180, 0),
            mp_pose.PoseLandmark.LEFT_THUMB: (0, 255, 255),  # Jaune (comme demandé)
            mp_pose.PoseLandmark.RIGHT_THUMB: (0, 255, 200),
            
            # Hanches
            mp_pose.PoseLandmark.LEFT_HIP: (128, 255, 0),  # Vert clair
            mp_pose.PoseLandmark.RIGHT_HIP: (100, 255, 0),
            
            # Genoux
            mp_pose.PoseLandmark.LEFT_KNEE: (0, 128, 255),  # Bleu clair
            mp_pose.PoseLandmark.RIGHT_KNEE: (0, 100, 255),
            
            # Chevilles
            mp_pose.PoseLandmark.LEFT_ANKLE: (255, 0, 200),  # Rose vif
            mp_pose.PoseLandmark.RIGHT_ANKLE: (255, 0, 150),
            
            # Pieds
            mp_pose.PoseLandmark.LEFT_HEEL: (200, 0, 255),  # Violet
            mp_pose.PoseLandmark.RIGHT_HEEL: (180, 0, 255),
            mp_pose.PoseLandmark.LEFT_FOOT_INDEX: (150, 0, 255),
            mp_pose.PoseLandmark.RIGHT_FOOT_INDEX: (120, 0, 255),
        }
        
        # Dessiner tous les landmarks avec leurs couleurs
        for landmark_id, color in landmark_colors.items():
            landmark = landmarks[landmark_id]
            if landmark.visibility > 0.5:  # Seulement si le landmark est visible
                pos = (int(landmark.x * w), int(landmark.y * h))
                cv2.circle(frame, pos, 6, color, -1)  # Cercle rempli
                cv2.circle(frame, pos, 6, (0, 0, 0), 2)  # Contour noir
       
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
