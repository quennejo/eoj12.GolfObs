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
#videoPath = "/Users/joequenneville/Movies/OBS/Replay 2025-11-10 22-35-59.mp4"  # Chemin de la vidéo à analyser
videoPath = "/Users/joequenneville/Movies/OBS/Replay 2025-11-10 22-31-29.mp4"
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
    if frame_count == 5:
        # Détection de l'épaule gauche 
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            h, w, _ = frame.shape
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_hand = (int(left_wrist.x * w), int(left_wrist.y * h))
            right_hand = (int(right_wrist.x * w), int(right_wrist.y * h))
            # Milieu des mains
            hands_mid = (int((left_hand[0] + right_hand[0]) / 2),  int((left_hand[1] + right_hand[1]) / 2))
        else:
            shoulder_point = None
        
        # Détection du bâton de golf avec YOLO-World à la frame 5
        club_boxes = yolo_club_detector.detect_club(frame)
        if club_boxes:
            # Prendre le premier bâton détecté et calculer le centre bas
            x1, y1, x2, y2 = club_boxes[0]
            club_bottom =  (x1, y2+40)  # Centre bas du rectangle
            club_start = (hands_mid[0]+ 450,(hands_mid[1]+40)-450)  # Milieu en haut du rectangle`
            club_start = (club_start[0], club_start[1]-20)
            club_line = (club_start, club_bottom)  # Ligne du milieu des mains au bas du club 
            # Dessiner un rectangle noir autour du bâton de golf détecté
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)

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
        
        cv2.circle(frame, ball_point, 8, (0,255,0), 2)  # cercle vert sur la balle détectée
        cv2.circle(frame, ball_point, 16, (0,0,0), 3)  # cercle noir autour de la balle
        #cv2.line(frame, ball_point, hands_mid, (0, 0, 255), 4)
    
    # Dessiner la ligne bleue du bâton de golf détecté avec YOLO-World
    if club_line is not None:
        cv2.line(frame, club_line[0], club_line[1], (255, 0, 0), 3)  # ligne bleue pour le bâton

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
