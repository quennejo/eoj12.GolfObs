"""
Main - Orchestrateur principal du système de replay de golf.

Workflow:
1. Démarre SoundTrigger pour la détection audio
2. Quand un swing est détecté → sauvegarde le replay buffer
3. Callback déclenche Replay.on_swing_detected()
4. Affiche automatiquement le replay dans OBS
"""

import sys
import time
import threading

# Forcer l'affichage immédiat des print (pas de buffering)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from SoundTrigger import SoundTrigger
from Replay import Replay


def main():
    """Fonction principale du système de replay de golf."""
    
    print("=" * 70)
    print("🏌️  GOLF REPLAY SYSTEM")
    print("=" * 70)
    print()
    
    # Configuration
    THRESHOLD = 60 # Seuil de volume pour détecter un swing
    COOLDOWN = 20 # Temps minimum entre deux détections (secondes)
    REPLAY_DURATION = 20 # Durée du replay en secondes avant
    REPLAY_MAX_FRAMES = 200  # Nombre maximum de frames à analyser
    SECOND_AFTER_SWING = 2  # Secondes après le swing à inclure dans le replay
    WATCH_DIR = "~/Movies/OBS" # Dossier surveillé
    SCENE = "SwingMonitor" # Scène OBS
    ANALYZE_SWING = True  # Activer l'analyse du swing
    
    
    print(f"🎤 Seuil de détection   : {THRESHOLD}")
    print(f"⏱️  Cooldown             : {COOLDOWN}s")
    print(f"🎬 Durée du replay      : {REPLAY_DURATION}s")
    print(f"🎮 Scène OBS            : {SCENE}")
    print(f"📂 Dossier surveillé    : {WATCH_DIR}")
    print(f"🔍 Analyse du swing     : {'Activée' if ANALYZE_SWING else 'Désactivée'}")
    print()
    
    try:
        # 1. Créer l'instance Replay
        print("📹 Initialisation du gestionnaire de Replay...")
        replay = Replay(
            watch_dir=WATCH_DIR,
            scene=SCENE,
            replay_duration=REPLAY_DURATION,
            analyze_swing=ANALYZE_SWING,
            max_frames=REPLAY_MAX_FRAMES
        )
        print("✅ Replay initialisé")
        print()
        
        # 2. Créer l'instance SoundTrigger avec callback vers Replay.on_swing_detected
        print("🎤 Initialisation du détecteur de son...")
        sound_trigger = SoundTrigger(
            threshold=THRESHOLD,
            cooldown=COOLDOWN,
            second_after_swing=SECOND_AFTER_SWING,
            on_swing_callback=replay.on_swing_detected  # ✅ Passer la méthode comme callback
        )
        
        # 3. Vérifier le replay buffer
        print("🔧 Vérification du Replay Buffer OBS...")
        if not sound_trigger.check_replay_buffer():
            print("\n❌ Configuration requise :")
            print("   1. Ouvrir OBS → Paramètres → Sortie")
            print("   2. Onglet 'Enregistrement'")
            print("   3. Activer 'Replay Buffer'")
            print("   4. Configurer la durée (30-60 secondes)")
            return
        print()
        
        # 4. Démarrer la détection audio
        print("🚀 Démarrage du système...")
        sound_trigger.start()
        print()
        
        print("🎯 Système prêt !")
        print("   → Frappez la balle pour déclencher un replay")
        print("   → Appuyez sur Ctrl+C pour arrêter")
        print()
        
        # 5. Garder le programme en vie (attend les callbacks)
        replay.keep_alive()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Nettoyage
        if 'sound_trigger' in locals():
            sound_trigger.stop()
        print("\n👋 Système arrêté")
        print("=" * 70)


if __name__ == "__main__":
    main()
    
