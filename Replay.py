"""
Replay - Gestion de l'affichage des replays dans OBS.
"""

import time
import glob
import os
from obsws_python import ReqClient
from SwingAnalyser import GolfSwingAnalyzer



class Replay:
    """Classe pour gérer l'affichage des replays dans OBS."""
    
    def __init__(self, watch_dir=None, scene="SwingMonitor", 
                 replay_source="replay", replay_text_source="replay_text",
                 replay_duration=30, host="localhost", port=4455, 
                 password="46Zic0AZYig9iktT", analyze_swing=True, max_frames=300):  
        """
        Initialise le gestionnaire de replay.
        
        Args:
            watch_dir: Dossier à surveiller pour les replays
            scene: Nom de la scène OBS
            replay_source: Nom de la source replay dans OBS
            replay_text_source: Nom de la source texte dans OBS
            replay_duration: Durée d'affichage du replay (secondes)
            host: Hôte OBS WebSocket
            port: Port OBS WebSocket
            password: Mot de passe OBS WebSocket
            analyze_swing: Si True, analyse la vidéo avec SwingAnalyser
        """
        self.watch_dir =  os.path.expanduser("~/Movies/OBS")
        self.scene = scene
        self.replay_source = replay_source
        self.replay_text_source = replay_text_source
        self.replay_duration = replay_duration
        self.analyze_swing = analyze_swing
        self.max_frames = max_frames
        
        # Connexion OBS
        self.client = ReqClient(host=host, port=port, password=password)
        
        # Cacher les fichiers existants dans le dossier surveillé
        # EXCLURE les fichiers _AI.mp4 qui sont générés par l'analyse
        all_mp4_files = set(glob.glob(os.path.join(self.watch_dir, "*.mp4")))
        self.cached_files = {f for f in all_mp4_files if not f.endswith("_AI.mp4")}
        self.already_seen = self.cached_files.copy()
        
        # Flag pour éviter les replays concurrents
        self.replay_in_progress = False
    
    def analyze_video(self, video_path):
        """
        Analyse la vidéo avec GolfSwingAnalyzer.
        
        Args:
            video_path: Chemin vers la vidéo à analyser
            
        Returns:
            str: Chemin vers la vidéo analysée (avec _AI suffix)
        """
        try:
            print(f"🔍 Analyse du swing en cours...", flush=True)
            print(f"   📁 Fichier : {os.path.basename(video_path)}", flush=True)
            print(f"   🎯 Max frames : {self.max_frames}", flush=True)
            
            # Créer l'analyseur
            analyzer = GolfSwingAnalyzer(
                video_path=video_path,
                max_frames=self.max_frames
            )
            
            # Lancer l'analyse (sans prévisualisation)
            print(f"   🚀 Démarrage de l'analyse...", flush=True)
            output_path = analyzer.run(show_preview=False)
            
            print(f"✅ Analyse terminée : {os.path.basename(output_path)}", flush=True)
            return output_path
            
        except Exception as e:
            print(f"⚠️  Erreur lors de l'analyse : {e}", flush=True)
            print(f"   → Utilisation de la vidéo originale", flush=True)
            return video_path
    
    def trigger_replay(self, video_path):
        """
        Déclenche l'affichage d'un replay dans OBS.
        
        Args:
            video_path: Chemin vers la vidéo de replay
        """
        try:
            # Vérifier si un replay est déjà en cours
            if self.replay_in_progress:
                print(f"⚠️  Un replay est déjà en cours, ignorer cette demande", flush=True)
                return
            
            # Marquer le replay comme en cours
            self.replay_in_progress = True
            
            print(f"\n🎬 Déclenchement du replay : {os.path.basename(video_path)}", flush=True)
            
            # Analyser la vidéo si demandé
            if self.analyze_swing:
                print(f"🧠 Mode analyse activé", flush=True)
                video_to_play = self.analyze_video(video_path)
            else:
                print(f"⚡ Mode lecture directe (sans analyse)", flush=True)
                video_to_play = video_path
            
            print(f"📺 Basculement sur la scène OBS '{self.scene}'...", flush=True)
            # Basculer sur la scène
            self.client.set_current_program_scene(self.scene)
            time.sleep(1)
            
            # Désactiver le live (item 1), activer le replay (item 2) et texte (item 3)
            self.client.set_scene_item_enabled(scene_name=self.scene, item_id=1, enabled=False)
            self.client.set_scene_item_enabled(scene_name=self.scene, item_id=2, enabled=True)
            self.client.set_scene_item_enabled(scene_name=self.scene, item_id=3, enabled=True)
            time.sleep(0.1)
            
            # Mettre à jour la source média
            self.client.set_input_settings(
                name=self.replay_source,
                settings={"local_file": video_to_play},
                overlay=True
            )
            self.client.set_input_settings(
                name=self.replay_text_source,
                settings={"text": "Replay"},
                overlay=True
            )
            
            print(f"⏱️  Affichage du replay pendant {self.replay_duration}s...")
            
            # Attendre la durée du replay
            time.sleep(self.replay_duration)
            
            # Retour au live
            print("📹 Retour au live")
            self.client.set_input_settings(
                name=self.replay_text_source,
                settings={"text": ""},
                overlay=True
            )
            self.client.set_scene_item_enabled(scene_name=self.scene, item_id=1, enabled=True)
            self.client.set_scene_item_enabled(scene_name=self.scene, item_id=2, enabled=False)
            self.client.set_scene_item_enabled(scene_name=self.scene, item_id=3, enabled=False)
            
            print("✅ Replay terminé\n")
            
        except Exception as e:
            print(f"❌ Erreur lors du replay : {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            # Toujours libérer le flag, même en cas d'erreur
            self.replay_in_progress = False
    
    def on_swing_detected(self, timestamp):
        """
        Callback appelé quand un swing est détecté.
        Vérifie s'il y a une nouvelle vidéo et la traite.
        
        Args:
            timestamp: Timestamp de la détection du swing
        """
        print(f"\n🎯 Swing détecté ! (timestamp: {timestamp})", flush=True)
        
        # Si un replay est déjà en cours, ignorer cette détection
        if self.replay_in_progress:
            print(f"⚠️  Un replay est déjà en cours, cette détection sera ignorée", flush=True)
            return
        
        print(f"🔍 Recherche de la vidéo...", flush=True)
        
        # Attendre que le fichier soit complètement écrit par OBS
        # On va essayer plusieurs fois avec un délai croissant
        max_attempts = 5
        wait_times = [1, 2, 2, 3, 3]  # Attendre 1s, puis 2s, puis 2s, etc.
        
        for attempt in range(max_attempts):
            print(f"   🔄 Tentative {attempt + 1}/{max_attempts}...", flush=True)
            time.sleep(wait_times[attempt])
            
            # Chercher la nouvelle vidéo
            new_video = self.check_for_new_video()
            
            if new_video:
                print(f"\n{'='*60}", flush=True)
                print(f"📹 Nouvelle vidéo détectée : {os.path.basename(new_video)}", flush=True)
                print(f"{'='*60}\n", flush=True)
                
                # Marquer la vidéo comme vue IMMÉDIATEMENT
                self.already_seen.add(new_video)
                print(f"✅ Vidéo marquée comme vue avant démarrage du replay", flush=True)
                
                # Déclencher le replay (qui vérifiera aussi le flag replay_in_progress)
                self.trigger_replay(new_video)
                return  # Succès, on sort
        
        # Si on arrive ici, aucune vidéo n'a été trouvée après toutes les tentatives
        print(f"⚠️  Aucune nouvelle vidéo trouvée après {max_attempts} tentatives", flush=True)
        print(f"   Vérifiez que :", flush=True)
        print(f"   • Le Replay Buffer OBS est actif", flush=True)
        print(f"   • Les replays sont bien sauvegardés dans {self.watch_dir}", flush=True)
    
    def check_for_new_video(self):
        """
        Vérifie s'il y a une nouvelle vidéo dans le dossier surveillé.
        Retourne le fichier le plus récent parmi les nouveaux fichiers.
        EXCLUT les fichiers *_AI.mp4 qui sont générés par l'analyse.
        
        Returns:
            str ou None: Chemin de la nouvelle vidéo ou None
        """
        # Vérifier que le dossier existe
        if not os.path.exists(self.watch_dir):
            print(f"⚠️  Le dossier surveillé n'existe pas : {self.watch_dir}", flush=True)
            return None
        
        # Obtenir tous les fichiers .mp4 actuels (EXCLURE les fichiers _AI.mp4)
        all_mp4_files = set(glob.glob(os.path.join(self.watch_dir, "*.mp4")))
        current_files = {f for f in all_mp4_files if not f.endswith("_AI.mp4")}
        
        # Trouver les nouveaux fichiers (non encore vus)
        new_files = current_files - self.already_seen
        
        print(f"🔍 Vérification : {len(current_files)} fichiers totaux (excl. _AI), {len(self.already_seen)} déjà vus, {len(new_files)} nouveaux", flush=True)
        
        if new_files:
            # Trouver le fichier le plus récent parmi les nouveaux
            latest = max(new_files, key=os.path.getmtime)
            
            print(f"🆕 Nouvelle vidéo trouvée: {os.path.basename(latest)}", flush=True)
            print(f"   📅 Date de modification : {time.ctime(os.path.getmtime(latest))}", flush=True)
            
            # NE PAS marquer comme vu ici - ce sera fait dans on_swing_detected()
            # pour éviter les doubles replays
            return latest
        
        return None
    
    def keep_alive(self):
        """
        Garde le programme en vie en attendant les callbacks.
        Simple boucle infinie avec affichage périodique.
        """
        print(f"💤 En attente des swings...", flush=True)
        print(f"   Appuyez sur Ctrl+C pour arrêter", flush=True)
        print()
        
        try:
            count = 0
            while True:
                time.sleep(10)  # Attendre 10 secondes
                count += 1
                #print(f"💓 Système actif... ({count * 10}s écoulées)", flush=True)
        except KeyboardInterrupt:
            print("\n⚠️  Arrêt demandé", flush=True)
