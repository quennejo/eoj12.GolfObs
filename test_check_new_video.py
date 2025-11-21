#!/usr/bin/env python3
"""
Test de la fonction check_for_new_video - Crée un fichier de test et vérifie la détection.
"""

import sys
import os
import time
import shutil

sys.stdout.reconfigure(line_buffering=True)

from Replay import Replay

print("=" * 70)
print("🧪 TEST DE check_for_new_video()")
print("=" * 70)
print()

WATCH_DIR = os.path.expanduser("~/Movies/OBS")

# Créer l'instance Replay
print("📹 Création de l'instance Replay...")
replay = Replay(
    watch_dir=WATCH_DIR,
    scene="SwingMonitor",
    replay_duration=20,
    analyze_swing=False,
    max_frames=200
)
print(f"✅ Replay créé - {len(replay.already_seen)} fichiers déjà présents\n")

# Test 1 : Vérifier qu'il n'y a pas de nouvelle vidéo
print("📊 Test 1 : Vérification initiale (aucun nouveau fichier)")
result = replay.check_for_new_video()
if result is None:
    print("✅ Correct : Aucun nouveau fichier détecté\n")
else:
    print(f"⚠️  Inattendu : Fichier trouvé : {result}\n")

# Test 2 : Créer un nouveau fichier de test
print("📊 Test 2 : Création d'un fichier de test")
import glob
existing_videos = glob.glob(os.path.join(WATCH_DIR, "*.mp4"))

if existing_videos:
    # Copier une vidéo existante pour le test
    source = existing_videos[0]
    timestamp = int(time.time())
    test_file = os.path.join(WATCH_DIR, f"test_replay_{timestamp}.mp4")
    
    print(f"   📋 Source : {os.path.basename(source)}")
    print(f"   📄 Création de : {os.path.basename(test_file)}")
    shutil.copy2(source, test_file)
    print(f"   ✅ Fichier créé ({os.path.getsize(test_file) / 1024 / 1024:.2f} MB)")
    print()
    
    # Attendre un peu
    time.sleep(0.5)
    
    # Test 3 : Vérifier la détection
    print("📊 Test 3 : Détection du nouveau fichier")
    result = replay.check_for_new_video()
    
    if result:
        print(f"✅ Succès : Fichier détecté !")
        print(f"   📁 Chemin : {result}")
        print(f"   📄 Nom : {os.path.basename(result)}")
        print(f"   ⏰ Date : {time.ctime(os.path.getmtime(result))}")
        
        # Vérifier que c'est bien notre fichier de test
        if os.path.basename(result) == os.path.basename(test_file):
            print(f"   ✅ C'est bien le fichier de test créé")
        else:
            print(f"   ⚠️  C'est un autre fichier : {os.path.basename(result)}")
    else:
        print(f"❌ Échec : Aucun fichier détecté")
    print()
    
    # Test 4 : Vérifier qu'un deuxième appel ne retourne rien
    print("📊 Test 4 : Deuxième vérification (fichier déjà vu)")
    result2 = replay.check_for_new_video()
    if result2 is None:
        print("✅ Correct : Le fichier est maintenant marqué comme vu\n")
    else:
        print(f"⚠️  Inattendu : Fichier retourné à nouveau : {result2}\n")
    
    # Nettoyage
    print("🧹 Nettoyage...")
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"   ✅ Fichier de test supprimé")
    
else:
    print("❌ Aucune vidéo existante dans le dossier pour créer un test")
    print(f"   Veuillez enregistrer un replay depuis OBS d'abord")

print()
print("=" * 70)
print("✅ TEST TERMINÉ")
print("=" * 70)
