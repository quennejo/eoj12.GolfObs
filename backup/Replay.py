import time
import glob
import os
import shutil
from obsws_python import ReqClient


# Configuration
HOST = "localhost"
PORT = 4455
PASSWORD = "46Zic0AZYig9iktT"
WATCH_DIR = os.path.expanduser("~/Movies/OBS")
TARGET_FILE = os.path.join(WATCH_DIR, "latest_replay.mp4")

# Connexion au serveur OBS WebSocket
client = ReqClient(host=HOST, port=PORT, password=PASSWORD)

# Étape 1 : Sauvegarde du Replay Buffer
client.save_replay_buffer()
print("✅ Replay sauvegardé")

# Attendre un peu que le fichier soit écrit
time.sleep(2)

# Étape 2 : Trouver le dernier fichier mp4 créé
files = sorted(glob.glob(os.path.join(WATCH_DIR, "*.mp4")), key=os.path.getmtime, reverse=True)
if files:
    latest = files[0]
    shutil.copy(latest, TARGET_FILE)
    print(f"✅ Copie : {latest} → {TARGET_FILE}")
else:
    print("⚠️ Aucun fichier trouvé.")

# Étape 3 : Passer à la scène Replay
client.set_current_program_scene("Replay")
print("🎬 Scène : Replay")

# Attendre 6 secondes
time.sleep(6)

# Étape 4 : Retour à la scène Live
client.set_current_program_scene("Live")
print("🏌️ Retour à Live")
