import os
import glob
import shutil
import time
import threading
import queue
import numpy as np
import sounddevice as sd
from obsws_python import ReqClient

# ------------------ CONFIG ------------------
HOST = "localhost"
PORT = 4455
PASSWORD = "46Zic0AZYig9iktT"

WATCH_DIR = os.path.expanduser("~/Movies/OBS")  # Dossier des replays OBS
SCENE = "SwingMonitor"
POLL_INTERVAL = 0.1  # secondes entre chaque vérification

# Configuration audio
AUDIO_DEVICE = None  # None = microphone par défaut
SAMPLE_RATE = 44100  # Hz
BLOCK_DURATION = 0.1  # secondes (100ms)
VOLUME_THRESHOLD = 30  # Seuil de volume pour détecter un swing
SWING_COOLDOWN = 5  # secondes minimum entre deux détections
# -------------------------------------------


class SoundTrigger:
    """Classe pour détecter les swings de golf par analyse audio."""
    
    def __init__(self, threshold=VOLUME_THRESHOLD, sample_rate=SAMPLE_RATE, 
                 host=HOST, port=PORT, password=PASSWORD, on_swing_callback=None):
        """
        Initialise le détecteur de son.
        
        Args:
            threshold: Seuil de volume pour détecter un swing
            sample_rate: Fréquence d'échantillonnage en Hz
            host: Hôte OBS WebSocket
            port: Port OBS WebSocket
            password: Mot de passe OBS WebSocket
            on_swing_callback: Fonction à appeler quand un swing est détecté
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.is_running = False
        self.stream = None
        self.on_swing_callback = on_swing_callback
        self.last_swing_time = 0
        
        # Connexion OBS
        self.client = ReqClient(host=host, port=port, password=password)
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback appelé par sounddevice pour chaque bloc audio."""
        if status:
            print(f"⚠️  Statut audio : {status}")
        
        # Calculer le volume RMS (Root Mean Square)
        volume_norm = np.linalg.norm(indata) * 10
        
        # Détecter un pic de volume (swing)
        if volume_norm > self.threshold:
            current_time = time.time()
            
            # Vérifier le cooldown pour éviter les faux positifs
            if current_time - self.last_swing_time >= SWING_COOLDOWN:
                print(f"🎤 SWING DÉTECTÉ ! Volume: {volume_norm:.2f}")
                self.last_swing_time = current_time
                
                # Sauvegarder immédiatement le replay buffer d'OBS
                try:
                    self.client.save_replay_buffer()
                    print("💾 Replay buffer sauvegardé dans OBS")
                    
                    # Appeler le callback si défini
                    if self.on_swing_callback:
                        self.on_swing_callback(current_time)
                        
                except Exception as e:
                    print(f"❌ Erreur lors de la sauvegarde du replay : {e}")
    
    def start(self):
        """Démarre la détection audio."""
        if self.is_running:
            print("⚠️  Le détecteur audio est déjà en cours d'exécution")
            return
        
        try:
            # Lister les périphériques audio disponibles
            print("\n🎙️  Périphériques audio disponibles :")
            print(sd.query_devices())
            print()
            
            # Démarrer le stream audio
            self.stream = sd.InputStream(
                device=AUDIO_DEVICE,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * BLOCK_DURATION),
                callback=self.audio_callback
            )
            self.stream.start()
            self.is_running = True
            print(f"✅ Détection audio démarrée (seuil: {self.threshold})")
            
        except Exception as e:
            print(f"❌ Erreur lors du démarrage de la détection audio : {e}")
            self.is_running = False
    
    def stop(self):
        """Arrête la détection audio."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.is_running = False
            print("🛑 Détection audio arrêtée")
    
    def check_replay_buffer_status(self):
        """
        Vérifie et démarre le replay buffer d'OBS si nécessaire.
        
        Returns:
            bool: True si le replay buffer est actif
        """
        try:
            status = self.client.get_replay_buffer_status()
            if status.output_active:
                print("✅ Replay buffer OBS actif")
                return True
            else:
                print("⚠️  Replay buffer OBS inactif, démarrage...")
                self.client.start_replay_buffer()
                time.sleep(2)
                print("✅ Replay buffer OBS démarré")
                return True
        except Exception as e:
            print(f"❌ Erreur avec le replay buffer : {e}")
            print("ℹ️  Assurez-vous que le Replay Buffer est configuré dans OBS")
            return False
            print(sd.query_devices())
            print()
            
            # Démarrer le stream audio
            self.stream = sd.InputStream(
                device=AUDIO_DEVICE,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * BLOCK_DURATION),
                callback=self.audio_callback
            )
            self.stream.start()
            self.is_running = True
            print(f"✅ Détection audio démarrée (seuil: {self.threshold})")
            
        except Exception as e:
            print(f"❌ Erreur lors du démarrage de la détection audio : {e}")
            self.is_running = False
    
    def stop(self):
        """Arrête la détection audio."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.is_running = False
            print("🛑 Détection audio arrêtée")


def check_replay_buffer_status():
    """
    Vérifie et démarre le replay buffer d'OBS si nécessaire.
    
    Returns:
        bool: True si le replay buffer est actif, False sinon
    """
    try:
        status = client.get_replay_buffer_status()
        if status.output_active:
            print("✅ Replay buffer OBS actif")
            return True
        else:
            print("⚠️  Replay buffer OBS inactif, démarrage...")
            client.start_replay_buffer()
            time.sleep(2)  # Attendre que le buffer démarre
            print("✅ Replay buffer OBS démarré")
            return True
    except Exception as e:
        print(f"❌ Erreur avec le replay buffer : {e}")
        print("ℹ️  Assurez-vous que le Replay Buffer est configuré dans OBS")
        return False


def trigger_replay(video_path):
    """
    Déclenche le replay dans OBS.
    
    Args:
        video_path: Chemin vers la vidéo de replay
    """
    try:
        print(f"🎬 Déclenchement du replay : {video_path}")
        
        # Basculer sur la scène
        client.set_current_program_scene(SCENE)
        time.sleep(1)
        
        # Désactiver le live, activer le replay
        client.set_scene_item_enabled(scene_name=SCENE, item_id=1, enabled=False)
        client.set_scene_item_enabled(scene_name=SCENE, item_id=2, enabled=True)
        client.set_scene_item_enabled(scene_name=SCENE, item_id=3, enabled=True)
        time.sleep(0.1)
        
        # Mettre à jour la source média
        client.set_input_settings(
            name=REPLAY_SOURCE,
            settings={"local_file": video_path},
            overlay=True
        )
        client.set_input_settings(
            name=REPLAY_TEXT_SOURCE,
            settings={"text": "Replay"},
            overlay=True
        )
        
        # Attendre la durée du replay
        time.sleep(REPLAY_SWING_DURATION)
        
        # Retour au live
        print("📹 Retour au live")
        client.set_input_settings(
            name=REPLAY_TEXT_SOURCE,
            settings={"text": ""},
            overlay=True
        )
        client.set_scene_item_enabled(scene_name=SCENE, item_id=1, enabled=True)
        client.set_scene_item_enabled(scene_name=SCENE, item_id=2, enabled=False)
        client.set_scene_item_enabled(scene_name=SCENE, item_id=3, enabled=False)
        
        print("✅ Replay terminé")
        
    except Exception as e:
        print(f"❌ Erreur lors du déclenchement du replay : {e}")


def watch_for_new_videos():
    """Surveille l'apparition de nouvelles vidéos dans le dossier."""
    already_seen = set(glob.glob(os.path.join(WATCH_DIR, "*.mp4")))
    print(f"🔍 Surveillance du dossier : {WATCH_DIR}")
    
    while True:
        time.sleep(POLL_INTERVAL)
        current_files = set(glob.glob(os.path.join(WATCH_DIR, "*.mp4")))
        new_files = current_files - already_seen
        
        if new_files:
            latest = max(new_files, key=os.path.getmtime)
            print(f"✅ Nouveau replay détecté : {latest}")
            
            # Vérifier s'il y a eu une détection de swing récente
            if not swing_detected_queue.empty():
                swing_detected_queue.get()  # Consommer l'événement
                trigger_replay(latest)
            else:
                print("ℹ️  Pas de swing détecté récemment, replay ignoré")
            
            already_seen.update(new_files)


def main():
    """Fonction principale."""
    print("=" * 60)
    print("🏌️  GOLF REPLAY - DÉTECTION AUDIO DE SWING")
    print("=" * 60)
    print(f"📂 Dossier surveillé : {WATCH_DIR}")
    print(f"🎤 Seuil de détection : {VOLUME_THRESHOLD}")
    print(f"⏱️  Cooldown entre swings : {SWING_COOLDOWN}s")
    print(f"🎬 Durée du replay : {REPLAY_SWING_DURATION}s")
    print("=" * 60)
    print()
    
    # Vérifier et démarrer le replay buffer d'OBS
    print("🔧 Vérification du Replay Buffer OBS...")
    if not check_replay_buffer_status():
        print("❌ Impossible de démarrer le Replay Buffer")
        print("📝 Étapes à suivre dans OBS :")
        print("   1. Paramètres → Sortie → Onglet 'Enregistrement'")
        print("   2. Activer 'Replay Buffer'")
        print("   3. Configurer la durée du buffer (ex: 30 secondes)")
        return
    
    # Initialiser le détecteur audio
    detector = AudioSwingDetector(threshold=VOLUME_THRESHOLD)
    
    try:
        # Démarrer la détection audio dans un thread séparé
        detector.start()
        
        # Surveiller les nouvelles vidéos dans le thread principal
        watch_for_new_videos()
        
    except KeyboardInterrupt:
        print("\n⚠️  Interruption par l'utilisateur")
    finally:
        detector.stop()
        print("👋 Arrêt du programme")


if __name__ == "__main__":
    main()




