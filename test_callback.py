#!/usr/bin/env python3
"""
Test du système avec callback : simule un swing sans vraiment déclencher l'audio.
"""

import sys
import time

sys.stdout.reconfigure(line_buffering=True)

from Replay import Replay

print("=" * 70)
print("🧪 TEST DU SYSTÈME CALLBACK")
print("=" * 70)
print()

# Configuration
WATCH_DIR = "~/Movies/OBS"
SCENE = "SwingMonitor"
REPLAY_DURATION = 20
ANALYZE_SWING = False  # Désactiver pour test rapide
MAX_FRAMES = 200

print("📹 Création de l'instance Replay...")
replay = Replay(
    watch_dir=WATCH_DIR,
    scene=SCENE,
    replay_duration=REPLAY_DURATION,
    analyze_swing=ANALYZE_SWING,
    max_frames=MAX_FRAMES
)
print("✅ Replay créé")
print()

print("🎯 Simulation d'un swing détecté...")
print("   (appel manuel de on_swing_detected)")
print()

# Simuler un swing détecté
current_time = time.time()
replay.on_swing_detected(current_time)

print()
print("=" * 70)
print("✅ TEST TERMINÉ")
print()
print("Si une vidéo a été détectée, elle devrait s'afficher dans OBS")
print("Sinon, vérifiez :")
print("  • Qu'il y a des fichiers .mp4 dans ~/Movies/OBS")
print("  • Que OBS est ouvert et connecté")
print("  • Que la scène 'SwingMonitor' existe")
print("=" * 70)
