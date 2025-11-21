#!/usr/bin/env python3
"""
Test du système Replay uniquement (sans détection audio).
"""

import sys
import os

# Forcer l'affichage immédiat
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from Replay import Replay

print("=" * 70)
print("🧪 TEST DU SYSTÈME REPLAY")
print("=" * 70)
print()

# Configuration
WATCH_DIR = os.path.expanduser("~/Movies/OBS")
SCENE = "SwingMonitor"
REPLAY_DURATION = 20
ANALYZE_SWING = False  # Désactiver l'analyse pour le test
MAX_FRAMES = 200

print(f"📂 Dossier surveillé : {WATCH_DIR}")
print(f"🎮 Scène OBS        : {SCENE}")
print(f"⏱️  Durée replay     : {REPLAY_DURATION}s")
print(f"🔍 Analyse          : {'Activée' if ANALYZE_SWING else 'Désactivée'}")
print()

# Vérifier que le dossier existe
if not os.path.exists(WATCH_DIR):
    print(f"❌ Le dossier {WATCH_DIR} n'existe pas!")
    print(f"   Création du dossier...")
    os.makedirs(WATCH_DIR, exist_ok=True)
    print(f"   ✅ Dossier créé")
print()

# Lister les vidéos existantes
import glob
videos = glob.glob(os.path.join(WATCH_DIR, "*.mp4"))
print(f"📹 Vidéos dans le dossier : {len(videos)}")
for v in videos[:5]:  # Afficher les 5 dernières
    print(f"   • {os.path.basename(v)}")
if len(videos) > 5:
    print(f"   ... et {len(videos) - 5} autres")
print()

try:
    print("🚀 Création de l'instance Replay...")
    replay = Replay(
        watch_dir=WATCH_DIR,
        scene=SCENE,
        replay_duration=REPLAY_DURATION,
        analyze_swing=ANALYZE_SWING,
        max_frames=MAX_FRAMES
    )
    print("✅ Replay créé avec succès")
    print()
    
    # Test de connexion OBS
    print("🔧 Test de connexion à OBS...")
    try:
        # Essayer d'obtenir la liste des scènes
        from obsws_python import ReqClient
        client = ReqClient(host="localhost", port=4455, password="46Zic0AZYig9iktT")
        scenes = client.get_scene_list()
        print(f"✅ Connexion OBS OK - {len(scenes.scenes)} scènes trouvées")
        
        # Vérifier si la scène existe
        scene_names = [s['sceneName'] for s in scenes.scenes]
        if SCENE in scene_names:
            print(f"✅ Scène '{SCENE}' trouvée")
        else:
            print(f"⚠️  Scène '{SCENE}' non trouvée!")
            print(f"   Scènes disponibles : {', '.join(scene_names)}")
        
    except Exception as e:
        print(f"❌ Erreur de connexion OBS : {e}")
        print(f"   Vérifiez que :")
        print(f"   • OBS est ouvert")
        print(f"   • WebSocket est activé (Outils → WebSocket Server Settings)")
        print(f"   • Port : 4455, Password : 46Zic0AZYig9iktT")
    print()
    
    print("👀 Démarrage de la surveillance...")
    print("   → Ajoutez un fichier .mp4 dans ~/Movies/OBS pour tester")
    print("   → Appuyez sur Ctrl+C pour arrêter")
    print()
    
    # Surveiller pendant 30 secondes pour le test
    replay.watch_and_replay(poll_interval=0.5, timeout=30)
    
except KeyboardInterrupt:
    print("\n\n⚠️  Test arrêté par l'utilisateur")
except Exception as e:
    print(f"\n❌ Erreur : {e}")
    import traceback
    traceback.print_exc()
finally:
    print("\n👋 Test terminé")
    print("=" * 70)
