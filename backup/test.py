import os
import glob
import shutil
import time
from obsws_python import ReqClient

# ------------------ CONFIG ------------------
HOST = "localhost"
PORT = 4455
PASSWORD = "golf123"

WATCH_DIR = os.path.expanduser("~/Movies/OBS")  # Dossier où OBS enregistre les replays
REPLAY_DURATION = 6  # secondes
LIVE_SCENE = "Live"
REPLAY_SCENE = "Replay"
MEDIA_SOURCE_NAME = "ReplayMedia"  # Nom exact de ta Media Source dans OBS
POLL_INTERVAL = 0.5
# -------------------------------------------

client = ReqClient(host=HOST, port=PORT, password=PASSWORD)
print("🔍 Surveillance du dossier des replays OBS...")

already_seen = set(glob.glob(os.path.join(WATCH_DIR, "*.mp4")))
replay_index = 1

while True:
    time.sleep(POLL_INTERVAL)
    current_files = set(glob.glob(os.path.join(WATCH_DIR, "*.mp4")))
    new_files = current_files - already_seen
    if new_files:
        latest = max(new_files, key=os.path.getmtime)

        # Générer un nom unique pour le nouveau fichier
        new_name = f"replay_{replay_index:03d}.mp4"
        target_file = os.path.join(WATCH_DIR, new_name)
        shutil.copy(latest, target_file)
        print(f"✅ Nouveau replay copié : {latest} → {target_file}")

        # Mettre à jour la Media Source
        client.set_input_settings(
            inputName=MEDIA_SOURCE_NAME,
            inputSettings={"local_file": target_file},
            overlay=True
        )
        print(f"🔄 Media Source mise à jour : {MEDIA_SOURCE_NAME} → {new_name}")

        # Passer sur la scène Replay
        client.set_current_program_scene(sceneName=REPLAY_SCENE)
        print(f"🎬 Scène : {REPLAY_SCENE}")

        # Attendre la durée du replay
        time.sleep(REPLAY_DURATION)

        # Retour à la scène Live
        client.set_current_program_scene(sceneName=LIVE_SCENE)
        print(f"🏌️ Retour à la scène : {LIVE_SCENE}")

        replay_index += 1
        already_seen.update(new_files)
