"""
SoundTrigger - Détection de swing de golf par analyse audio.
"""


import time
import numpy as np
import sounddevice as sd
from obsws_python import ReqClient


class SoundTrigger:
    """Classe pour détecter les swings de golf par analyse audio."""
    
    def __init__(self, threshold=30, sample_rate=44100, cooldown=5,
                 host="localhost", port=4455, password="46Zic0AZYig9iktT", 
                 second_after_swing=2, on_swing_callback=None):
        """
        Initialise le détecteur de son.
        
        Args:
            threshold: Seuil de volume pour détecter un swing
            sample_rate: Fréquence d'échantillonnage en Hz
            cooldown: Temps minimum entre deux détections (secondes)
            host: Hôte OBS WebSocket
            port: Port OBS WebSocket
            password: Mot de passe OBS WebSocket
            second_after_swing: Secondes à attendre après détection avant sauvegarde
            on_swing_callback: Fonction à appeler quand un swing est détecté
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.cooldown = cooldown
        self.is_running = False
        self.stream = None
        self.on_swing_callback = on_swing_callback
        self.last_swing_time = 0
        self.second_after_swing = second_after_swing
        
        # Connexion OBS
        self.client = ReqClient(host=host, port=port, password=password)
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback appelé par sounddevice pour chaque bloc audio."""
        if status:
            print(f"⚠️  Statut audio : {status}", flush=True)
        
        # Calculer le volume RMS (Root Mean Square)
        volume_norm = np.linalg.norm(indata) * 10
        
        # Debug: afficher le volume périodiquement
        #if int(time.time() * 10) % 50 == 0:  # Toutes les ~5 secondes
         #   print(f"📊 Volume actuel: {volume_norm:.2f} (seuil: {self.threshold})", flush=True)
        
        # Détecter un pic de volume (swing)
        if volume_norm > self.threshold:
            current_time = time.time()
            
            # Vérifier le cooldown pour éviter les faux positifs
            if current_time - self.last_swing_time >= self.cooldown:
                print(f"\n{'='*60}", flush=True)
                print(f"🎤 SWING DÉTECTÉ ! Volume: {volume_norm:.2f}", flush=True)
                print(f"{'='*60}\n", flush=True)
                self.last_swing_time = current_time
                
                # Attendre le nombre de secondes configuré
                print(f"⏳ Attente de {self.second_after_swing}s pour capture complète...", flush=True)
                time.sleep(self.second_after_swing)
                
                # Sauvegarder immédiatement le replay buffer d'OBS
                try:
                    print("💾 Sauvegarde du replay buffer OBS...", flush=True)
                    self.client.save_replay_buffer()
                    print("✅ Replay buffer sauvegardé dans OBS", flush=True)
                    
                    # Appeler le callback si défini
                    if self.on_swing_callback:
                        self.on_swing_callback(current_time)
                        
                except Exception as e:
                    print(f"❌ Erreur lors de la sauvegarde du replay : {e}", flush=True)
    
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
                device=None,  # Microphone par défaut
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                callback=self.audio_callback
            )
            self.stream.start()
            self.is_running = True
            print(f"✅ Détection audio démarrée (seuil: {self.threshold})")
            
        except Exception as e:
            print(f"❌ Erreur lors du démarrage de la détection audio : {e}")
            self.is_running = False
            raise
    
    def stop(self):
        """Arrête la détection audio."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.is_running = False
            print("🛑 Détection audio arrêtée")
    
    def check_replay_buffer(self):
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
            return False
