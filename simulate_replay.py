#!/usr/bin/env python3
"""
Test complet : copier une vidéo dans le dossier pour simuler un replay.
"""
import sys
import os
import shutil
import time
import glob

sys.stdout.reconfigure(line_buffering=True)

WATCH_DIR = os.path.expanduser("~/Movies/OBS")

print("=" * 70)
print("🧪 TEST DE SIMULATION DE REPLAY")
print("=" * 70)
print()

# Vérifier qu'il y a des vidéos à copier
videos = glob.glob(os.path.join(WATCH_DIR, "*.mp4"))
if not videos:
    print("❌ Aucune vidéo trouvée dans ~/Movies/OBS")
    print("   Veuillez enregistrer un replay depuis OBS d'abord")
    sys.exit(1)

# Prendre la dernière vidéo
source_video = max(videos, key=os.path.getmtime)
print(f"📹 Vidéo source : {os.path.basename(source_video)}")
print()

# Créer une copie avec un nouveau nom
timestamp = int(time.time())
test_video = os.path.join(WATCH_DIR, f"test_replay_{timestamp}.mp4")

print(f"📋 Copie de la vidéo pour simulation...")
shutil.copy2(source_video, test_video)
print(f"✅ Copie créée : {os.path.basename(test_video)}")
print()

print("🎬 Cette vidéo devrait maintenant être détectée par le système Replay")
print(f"   Si le système tourne, il va la traiter automatiquement")
print()

# Attendre un peu
print("⏳ Attente de 5 secondes...")
time.sleep(5)

# Vérifier que le fichier existe toujours
if os.path.exists(test_video):
    print(f"✅ Fichier toujours présent")
    print(f"   Taille : {os.path.getsize(test_video) / 1024 / 1024:.2f} MB")
else:
    print(f"⚠️  Fichier disparu (peut-être traité ?)")

print()
print("=" * 70)
