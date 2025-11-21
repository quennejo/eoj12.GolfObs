"""
Analyseur de swing de golf avec détection automatique.

Fonctionnalités:
- Détection de balle de golf (HoughCircles)
- Détection du club de golf (YOLO-World)
- Tracking du squelette (MediaPipe Pose)
- Annotations visuelles en temps réel
"""

import cv2
import mediapipe as mp
import os
import ssl
import numpy as np
from ultralytics import YOLOWorld

# Désactiver la vérification SSL
ssl._create_default_https_context = ssl._create_unverified_context


class YoloWorldGolfClubDetector:
    """Détecteur de club de golf utilisant YOLO-World (open-vocabulary)."""
    
    def __init__(self, model_path='yolov8s-worldv2.pt'):
        """
        Initialise le détecteur YOLO-World.
        
        Args:
            model_path: Chemin vers le modèle YOLO-World
        """
        self.model = YOLOWorld(model_path)
        self.model.set_classes(["golf club", "golf stick"])
    
    def detect_club(self, frame, confidence=0.3):
        """
        Détecte le club de golf dans une frame.
        
        Args:
            frame: Image à analyser
            confidence: Seuil de confiance (0.0 - 1.0)
            
        Returns:
            Liste de bounding boxes [(x1, y1, x2, y2), ...]
        """
        results = self.model(frame, conf=confidence)
        club_boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                club_boxes.append((x1, y1, x2, y2))
        return club_boxes


class GolfSwingAnalyzer:
    """Analyseur de swing de golf avec détection automatique."""
    
    # Constantes de configuration
    BALL_DETECTION_FRAMES = 100
    CLUB_DETECTION_FRAME = 5
    BALL_X_MIN = 400
    BALL_X_MAX = 850
    BALL_GREEN_RATIO_MIN = 0.5
    BALL_WHITE_RATIO_MIN = 0.3
    
    def __init__(self, video_path, max_frames=300, output_path=None):
        """
        Initialise l'analyseur de swing de golf.
        
        Args:
            video_path: Chemin vers la vidéo à analyser
            max_frames: Nombre maximum de frames à traiter
            output_path: Chemin de sortie (optionnel, auto-généré si None)
        """
        self.video_path = video_path
        self.max_frames = max_frames
        
        # Initialisation MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialisation détecteur YOLO-World
        self.yolo_club_detector = YoloWorldGolfClubDetector()
        
        # Variables de tracking
        self.ball_point = None
        self.ball_detected = False
        self.club_line = None
        self.frame_count = 0
        
        # Configuration vidéo
        self._setup_video_capture()
        self._setup_video_writer(output_path)
    
    def _setup_video_capture(self):
        """Configure la capture vidéo."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def _setup_video_writer(self, output_path):
        """Configure l'écriture de la vidéo de sortie."""
        if output_path is None:
            base, ext = os.path.splitext(self.video_path)
            output_path = f"{base}_AI{ext}"
        
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.out = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            self.fps, 
            (self.width, self.height)
        )
    
    def _get_hands_center(self, landmarks):
        """
        Calcule le centre des deux mains.
        
        Args:
            landmarks: Landmarks MediaPipe Pose
            
        Returns:
            Tuple (x, y) du centre des mains
        """
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        
        left_hand = (int(left_wrist.x * self.width), int(left_wrist.y * self.height))
        right_hand = (int(right_wrist.x * self.width), int(right_wrist.y * self.height))
        
        return (
            int((left_hand[0] + right_hand[0]) / 2),
            int((left_hand[1] + right_hand[1]) / 2)
        )
    
    def detect_ball_hough(self, frame):
        """
        Détecte la balle de golf avec HoughCircles.
        
        Args:
            frame: Image à analyser
            
        Returns:
            Tuple (x, y) de la position de la balle ou None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=30
        )
        
        if circles is None:
            return None
        
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            bx, by, radius = circle
            
            # Vérification de la zone autour de la balle
            patch_size = 10
            x1 = max(bx - patch_size, 0)
            y1 = max(by - patch_size, 0)
            x2 = min(bx + patch_size, frame.shape[1])
            y2 = min(by + patch_size, frame.shape[0])
            
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            
            # Conversion HSV
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            
            # Masque vert (gazon)
            mask_green = cv2.inRange(
                hsv_patch,
                np.array([35, 40, 40]),
                np.array([85, 255, 255])
            )
            green_ratio = cv2.countNonZero(mask_green) / (patch.shape[0] * patch.shape[1])
            
            # Masque blanc (balle)
            mask_white = cv2.inRange(
                hsv_patch,
                np.array([0, 0, 180]),
                np.array([180, 50, 255])
            )
            white_ratio = cv2.countNonZero(mask_white) / (patch.shape[0] * patch.shape[1])
            
            # Validation
            if (green_ratio > self.BALL_GREEN_RATIO_MIN and
                white_ratio > self.BALL_WHITE_RATIO_MIN and
                self.BALL_X_MIN < bx < self.BALL_X_MAX):
                return (bx + 10, by + 10)
        
        return None
    
    def detect_club_initial(self, frame, hands_mid):
        """
        Détecte le club à une frame spécifique avec YOLO-World.
        
        Args:
            frame: Image à analyser
            hands_mid: Position du centre des mains
        """
        club_boxes = self.yolo_club_detector.detect_club(frame)
        if not club_boxes:
            return
        
        x1, y1, x2, y2 = club_boxes[0]
        
        # Calcul des positions du club
        club_bottom = (x1, y2 + 40)
        club_start = (
            hands_mid[0] + 450,
            (hands_mid[1] + 40) - 450
        )
        club_start = (club_start[0], club_start[1] - 20)
        
        self.club_line = (club_start, club_bottom)
        
        # Dessiner le rectangle de détection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
    
    def _draw_ball(self, frame):
        """Dessine la balle détectée sur la frame."""
        if self.ball_point:
            cv2.circle(frame, self.ball_point, 8, (0, 255, 0), 2)
            cv2.circle(frame, self.ball_point, 16, (0, 0, 0), 3)
    
    def _draw_club(self, frame):
        """Dessine la ligne du club sur la frame."""
        if self.club_line:
            cv2.line(frame, self.club_line[0], self.club_line[1], (255, 0, 0), 3)
    
    def _draw_skeleton(self, frame, pose_landmarks):
        """Dessine le squelette sur la frame."""
        if pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
    
    def process_frame(self, frame):
        """
        Traite une frame individuelle.
        
        Args:
            frame: Image à traiter
            
        Returns:
            Frame annotée
        """
        # Conversion BGR -> RGB pour MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        
        # Détection de la balle (frames 1-100)
        if not self.ball_detected and self.frame_count < self.BALL_DETECTION_FRAMES:
            ball_pos = self.detect_ball_hough(frame)
            if ball_pos:
                self.ball_point = ball_pos
                self.ball_detected = True
        
        # Détection initiale du club
        if self.frame_count == self.CLUB_DETECTION_FRAME and result.pose_landmarks:
            hands_mid = self._get_hands_center(result.pose_landmarks.landmark)
            self.detect_club_initial(frame, hands_mid)
        
        # Annotations visuelles
        self._draw_ball(frame)
        self._draw_club(frame)
        self._draw_skeleton(frame, result.pose_landmarks)
        
        return frame
    
    def _read_and_process_video(self, show_preview=True):
        """
        Lit et traite la vidéo frame par frame.
        
        Args:
            show_preview: Afficher la prévisualisation en temps réel
            
        Yields:
            Tuple (frame_number, processed_frame) pour chaque frame traitée
        """
        while self.cap.isOpened():
            success, frame = self.cap.read()
            
            # Vérifier la fin de la vidéo ou limite de frames
            if not success or self.frame_count >= self.max_frames:
                break
            
            self.frame_count += 1
            
            # Traiter la frame
            processed_frame = self.process_frame(frame)
            
            # Afficher la prévisualisation si demandé
            if show_preview:
                cv2.imshow("Analyse swing", processed_frame)
                if cv2.waitKey(1) & 0xFF == 27:  # Touche Esc
                    print("⚠️  Analyse interrompue par l'utilisateur")
                    break
            
            yield self.frame_count, processed_frame
    
    def run(self, show_preview=True):
        """
        Lance l'analyse de la vidéo.
        
        Args:
            show_preview: Afficher la prévisualisation en temps réel
            
        Returns:
            Chemin de la vidéo de sortie
        """
        print(f"🎬 Analyse de la vidéo : {self.video_path}")
        print(f"📊 Résolution : {self.width}x{self.height} @ {self.fps:.2f} FPS")
        print(f"⏱️  Frames à traiter : {self.max_frames}")
        
        try:
            # Utiliser le générateur pour lire et traiter la vidéo
            for frame_num, processed_frame in self._read_and_process_video(show_preview):
                # Sauvegarder la frame
                self.out.write(processed_frame)
                
                # Afficher la progression tous les 30 frames
                if frame_num % 30 == 0:
                    progress = (frame_num / self.max_frames) * 100
                    print(f"📹 Progression : {frame_num}/{self.max_frames} ({progress:.1f}%)")
        
        finally:
            self._cleanup()
        
        print(f"✅ Vidéo annotée sauvegardée : {self.output_path}")
        return self.output_path
    
    def _cleanup(self):
        """Libère les ressources."""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        if hasattr(self, 'out') and self.out:
            self.out.release()
        cv2.destroyAllWindows()
    
    def __enter__(self):
        """Support du context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup automatique avec context manager."""
        self._cleanup()
        return False


# Point d'entrée principal
if __name__ == "__main__":
    import sys
    
    # Configuration par défaut
    DEFAULT_VIDEO = "/Users/joequenneville/Movies/OBS/Replay 2025-11-10 22-31-29.mp4"
    
    # Utiliser l'argument en ligne de commande ou la valeur par défaut
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    
    # Analyse avec context manager
    try:
        with GolfSwingAnalyzer(video_path, max_frames=300) as analyzer:
            output_path = analyzer.run(show_preview=True)
            print(f"\n🎉 Analyse terminée avec succès !")
            print(f"📂 Fichier de sortie : {output_path}")
    except Exception as e:
        print(f"\n❌ Erreur lors de l'analyse : {e}")
        sys.exit(1)
