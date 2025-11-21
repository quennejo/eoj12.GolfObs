#!/usr/bin/env python3
"""
Script de test pour vérifier que les prints s'affichent correctement.
"""

import sys
import time

# Forcer l'affichage immédiat
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("=" * 70, flush=True)
print("🧪 TEST DU SYSTÈME DE PRINTS", flush=True)
print("=" * 70, flush=True)
print(flush=True)

# Test 1 : Prints basiques
print("📝 Test 1 : Prints basiques", flush=True)
for i in range(5):
    print(f"   → Test {i+1}/5", flush=True)
    time.sleep(0.2)
print("   ✅ Test 1 OK", flush=True)
print(flush=True)

# Test 2 : Import des modules
print("📦 Test 2 : Import des modules", flush=True)
try:
    from SoundTrigger import SoundTrigger
    print("   ✅ SoundTrigger importé", flush=True)
except Exception as e:
    print(f"   ❌ Erreur SoundTrigger : {e}", flush=True)

try:
    from Replay import Replay
    print("   ✅ Replay importé", flush=True)
except Exception as e:
    print(f"   ❌ Erreur Replay : {e}", flush=True)

try:
    from SwingAnalyser import GolfSwingAnalyzer
    print("   ✅ SwingAnalyser importé", flush=True)
except Exception as e:
    print(f"   ❌ Erreur SwingAnalyser : {e}", flush=True)

print(flush=True)

# Test 3 : Instanciation des classes
print("🏗️  Test 3 : Instanciation des classes", flush=True)
try:
    print("   → Création de SoundTrigger...", flush=True)
    sound = SoundTrigger(threshold=60, cooldown=20, second_after_swing=2)
    print("   ✅ SoundTrigger créé", flush=True)
    print(f"      • Threshold: {sound.threshold}", flush=True)
    print(f"      • Cooldown: {sound.cooldown}s", flush=True)
    print(f"      • Second after swing: {sound.second_after_swing}s", flush=True)
except Exception as e:
    print(f"   ❌ Erreur création SoundTrigger : {e}", flush=True)
    import traceback
    traceback.print_exc()

print(flush=True)

try:
    print("   → Création de Replay...", flush=True)
    replay = Replay(
        watch_dir="~/Movies/OBS",
        scene="SwingMonitor",
        replay_duration=20,
        analyze_swing=True,
        max_frames=200
    )
    print("   ✅ Replay créé", flush=True)
    print(f"      • Watch dir: {replay.watch_dir}", flush=True)
    print(f"      • Scene: {replay.scene}", flush=True)
    print(f"      • Duration: {replay.replay_duration}s", flush=True)
    print(f"      • Analyze: {replay.analyze_swing}", flush=True)
except Exception as e:
    print(f"   ❌ Erreur création Replay : {e}", flush=True)
    import traceback
    traceback.print_exc()

print(flush=True)

# Test 4 : Vérification du replay buffer
print("🔧 Test 4 : Vérification du Replay Buffer OBS", flush=True)
try:
    if sound.check_replay_buffer():
        print("   ✅ Replay Buffer OBS OK", flush=True)
    else:
        print("   ⚠️  Replay Buffer OBS non disponible", flush=True)
except Exception as e:
    print(f"   ❌ Erreur vérification : {e}", flush=True)

print(flush=True)

# Test 5 : Test de callback
print("🔔 Test 5 : Test de callback", flush=True)
callback_called = False

def test_callback(timestamp):
    global callback_called
    callback_called = True
    print(f"   ✅ Callback appelé ! Timestamp: {timestamp}", flush=True)

sound.on_swing_callback = test_callback
if sound.on_swing_callback:
    sound.on_swing_callback(time.time())
    if callback_called:
        print("   ✅ Système de callback fonctionne", flush=True)

print(flush=True)

print("=" * 70, flush=True)
print("✅ TOUS LES TESTS SONT TERMINÉS", flush=True)
print("=" * 70, flush=True)
