#!/usr/bin/env python3
"""
Script de test pour vérifier la détection audio en temps réel.
Affiche le volume du microphone en continu.
"""

import sys
import time
import numpy as np
import sounddevice as sd

# Forcer l'affichage immédiat
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("=" * 70)
print("🎤 TEST DE DÉTECTION AUDIO")
print("=" * 70)
print()

# Configuration
SAMPLE_RATE = 44100
THRESHOLD = 60  # Même seuil que dans main.py

print(f"📊 Configuration:")
print(f"   Sample rate : {SAMPLE_RATE} Hz")
print(f"   Seuil       : {THRESHOLD}")
print()

print("🎙️  Périphériques audio disponibles :")
print(sd.query_devices())
print()

print("🚀 Démarrage de l'écoute audio...")
print("   → Faites du bruit pour tester la détection")
print("   → Appuyez sur Ctrl+C pour arrêter")
print()

# Variables
last_print_time = 0
max_volume_seen = 0

def audio_callback(indata, frames, time_info, status):
    """Callback audio - affiche le volume en temps réel."""
    global last_print_time, max_volume_seen
    
    if status:
        print(f"⚠️  Statut audio : {status}", flush=True)
    
    # Calculer le volume RMS
    volume_norm = np.linalg.norm(indata) * 10
    
    # Suivre le max
    max_volume_seen = max(max_volume_seen, volume_norm)
    
    # Afficher périodiquement
    current_time = time.time()
    if current_time - last_print_time >= 0.5:  # Toutes les 0.5 secondes
        bar_length = int(volume_norm / 10)
        bar = "█" * bar_length
        
        # Couleur selon dépassement du seuil
        if volume_norm > THRESHOLD:
            print(f"🔴 Volume: {volume_norm:6.2f} {bar} >>> DÉTECTION !", flush=True)
        else:
            print(f"🟢 Volume: {volume_norm:6.2f} {bar}", flush=True)
        
        last_print_time = current_time

try:
    # Démarrer le stream audio
    with sd.InputStream(
        device=None,  # Microphone par défaut
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE * 0.1),  # 100ms blocks
        callback=audio_callback
    ):
        print("✅ Écoute active\n")
        
        # Boucle infinie
        while True:
            time.sleep(0.1)
            
except KeyboardInterrupt:
    print("\n\n⚠️  Arrêt demandé")
except Exception as e:
    print(f"\n❌ Erreur : {e}")
    import traceback
    traceback.print_exc()
finally:
    print(f"\n📊 Volume maximum détecté : {max_volume_seen:.2f}")
    print("=" * 70)
