import os
import glob
import shutil
import time
from obsws_python import ReqClient

# ------------------ CONFIG ------------------
HOST = "localhost"
PORT = 4455
PASSWORD = "46Zic0AZYig9iktT"

WATCH_DIR = os.path.expanduser("~/Movies/OBS")  # Dossier des replays OBS
TARGET_FILE = os.path.join(WATCH_DIR, "latest_replay.mp4")
REPLAY_SWING_DURATION =30
REPLAY_DURATION = 10  # secondes pour le replay
SCENE ="SwingMonitor"
LIVE_SOURCE ="iphone"
REPLAY_SOURCE ="replay"
REPLAY_TEXT_SOURCE="replay_text"
POLL_INTERVAL = 0.5  # secondes entre chaque vérification

MEDIA_SOURCE_NAME = "Media Source"  # Nom exact de la Media Source
# -------------------------------------------

# Connexion OBS
client = ReqClient(host=HOST, port=PORT, password=PASSWORD)



print("🔍 Surveillance du dossier des replays OBS...")

# On garde en mémoire les fichiers déjà vus
already_seen = set(glob.glob(os.path.join(WATCH_DIR, "*.mp4")))

while True:
    time.sleep(POLL_INTERVAL)
    current_files = set(glob.glob(os.path.join(WATCH_DIR, "*.mp4")))
    new_files = current_files - already_seen
    if new_files:
        latest = max(new_files, key=os.path.getmtime)
        print(f"✅ Nouveau replay détecté : {latest} → {TARGET_FILE}")
        
        client.set_current_program_scene(SCENE)
        time.sleep(1)


        # Désactiver puis réactiver la source pour forcer le reload
        #Live
        client.set_scene_item_enabled(scene_name=SCENE,item_id=1,enabled=False)
        #Replay
        client.set_scene_item_enabled(scene_name=SCENE,item_id=2,enabled=True)
        #Replay Text
        client.set_scene_item_enabled(scene_name=SCENE,item_id=3,enabled=True)
        time.sleep(0.1)
          # Mettre à jour la Media Source pour pointer vers ce nouveau fichier

        client.set_input_settings(name=REPLAY_SOURCE,settings={"local_file": latest},overlay=True)
        client.set_input_settings(name=REPLAY_TEXT_SOURCE,settings={"text": "Replay"},overlay=True)
        #time.sleep(0.1)
   
        time.sleep(REPLAY_SWING_DURATION)
        #Replay
        print("Replay")
     
        # Retour à la scène Live
        client.set_input_settings(name=REPLAY_TEXT_SOURCE,settings={"text": ""},overlay=True)
        #Live
        client.set_scene_item_enabled(scene_name=SCENE,item_id=1,enabled=True)
        #Replay
        client.set_scene_item_enabled(scene_name=SCENE,item_id=2,enabled=False)
        #Replay text
        client.set_scene_item_enabled(scene_name=SCENE,item_id=3,enabled=False)

    

        already_seen.update(new_files)




   