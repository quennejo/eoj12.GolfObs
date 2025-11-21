#!/usr/bin/env python3
"""
Guide de diagnostic complet du système de replay.
"""

import sys
import os
import glob
sys.stdout.reconfigure(line_buffering=True)

print("=" * 70)
print("🔍 DIAGNOSTIC COMPLET DU SYSTÈME")
print("=" * 70)
print()

# 1. Vérifier l'environnement Python
print("1️⃣  ENVIRONNEMENT PYTHON")
print(f"   Python : {sys.version}")
print(f"   Exécutable : {sys.executable}")
print()

# 2. Vérifier les imports
print("2️⃣  MODULES REQUIS")
modules = [
    ("obsws_python", "Contrôle OBS"),
    ("sounddevice", "Détection audio"),
    ("numpy", "Calculs numériques"),
    ("cv2", "Traitement vidéo"),
    ("mediapipe", "Détection squelette"),
]

for module_name, description in modules:
    try:
        __import__(module_name)
        print(f"   ✅ {module_name:20s} - {description}")
    except ImportError:
        print(f"   ❌ {module_name:20s} - {description} (MANQUANT!)")
print()

# 3. Vérifier les fichiers du projet
print("3️⃣  FICHIERS DU PROJET")
project_files = [
    "main.py",
    "SoundTrigger.py",
    "Replay.py",
    "SwingAnalyser.py"
]

for file in project_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"   ✅ {file:25s} ({size:,} bytes)")
    else:
        print(f"   ❌ {file:25s} (MANQUANT!)")
print()

# 4. Vérifier le dossier de surveillance
print("4️⃣  DOSSIER DE SURVEILLANCE")
watch_dir = os.path.expanduser("~/Movies/OBS")
print(f"   Chemin : {watch_dir}")

if os.path.exists(watch_dir):
    print(f"   ✅ Dossier existe")
    videos = glob.glob(os.path.join(watch_dir, "*.mp4"))
    print(f"   📹 Vidéos présentes : {len(videos)}")
    if videos:
        latest = max(videos, key=os.path.getmtime)
        import time
        age_seconds = time.time() - os.path.getmtime(latest)
        print(f"   📅 Dernière vidéo : {os.path.basename(latest)}")
        print(f"      (il y a {age_seconds/60:.1f} minutes)")
else:
    print(f"   ❌ Dossier n'existe pas!")
print()

# 5. Tester la connexion OBS
print("5️⃣  CONNEXION OBS")
try:
    from obsws_python import ReqClient
    client = ReqClient(host="localhost", port=4455, password="46Zic0AZYig9iktT")
    print(f"   ✅ Connexion établie")
    
    # Obtenir des infos
    version = client.get_version()
    print(f"   📌 OBS Version : {version.obs_version}")
    print(f"   📌 WebSocket Version : {version.obs_web_socket_version}")
    
    # Lister les scènes
    scenes = client.get_scene_list()
    print(f"   🎬 Scènes : {len(scenes.scenes)}")
    for scene in scenes.scenes:
        marker = "👉" if scene['sceneName'] == "SwingMonitor" else "  "
        print(f"      {marker} {scene['sceneName']}")
    
    # Vérifier le replay buffer
    try:
        status = client.get_replay_buffer_status()
        if status.output_active:
            print(f"   ✅ Replay Buffer : ACTIF")
        else:
            print(f"   ⚠️  Replay Buffer : INACTIF")
    except Exception as e:
        print(f"   ❌ Replay Buffer : Erreur - {e}")
    
except Exception as e:
    print(f"   ❌ Échec de connexion : {e}")
    print(f"      Vérifiez que :")
    print(f"      • OBS est ouvert")
    print(f"      • WebSocket Server est activé")
    print(f"      • Port 4455, Password : 46Zic0AZYig9iktT")
print()

# 6. Tester l'audio
print("6️⃣  PÉRIPHÉRIQUES AUDIO")
try:
    import sounddevice as sd
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    print(f"   🎤 Périphérique par défaut : {devices[default_input]['name']}")
    print(f"   📊 Canaux : {devices[default_input]['max_input_channels']}")
    print(f"   🔊 Fréquence : {devices[default_input]['default_samplerate']} Hz")
except Exception as e:
    print(f"   ❌ Erreur audio : {e}")
print()

print("=" * 70)
print("✅ DIAGNOSTIC TERMINÉ")
print()
print("📝 RECOMMANDATIONS :")
print("   1. Si OBS n'est pas connecté : Ouvrez OBS et activez WebSocket")
print("   2. Si Replay Buffer est inactif : Activez-le dans OBS → Paramètres")
print("   3. Si des modules manquent : pip install <module>")
print("   4. Si pas de vidéos : Testez manuellement avec 'Sauvegarder le replay'")
print("=" * 70)
