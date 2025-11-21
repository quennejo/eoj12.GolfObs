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
REPLAY_DURATION = 18  # secondes pour le replay
LIVE_SCENE = "Live"
REPLAY_SCENE = "Replay"
POLL_INTERVAL = 0.5  # secondes entre chaque vérification
REPLAY_SCENE = "Replay"
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
        shutil.copy(latest, TARGET_FILE)
        print(f"✅ Nouveau replay détecté : {latest} → {TARGET_FILE}")
        #time.sleep(1)
        client.set_scene_item_enabled(scene_name=REPLAY_SCENE,item_id=1,enabled=False)
        #time.sleep(1)
        client.set_scene_item_enabled(scene_name=REPLAY_SCENE,item_id=1,enabled=True)

        # Mettre à jour la Media Source pour pointer vers ce nouveau fichier

        client.set_input_settings(name=MEDIA_SOURCE_NAME,settings={"local_file": latest},overlay=True)
        #client.set_input_settings()
         #   inputName=MEDIA_SOURCE_NAME,
          #  inputSettings={"local_file": latest},
          #  overlay=True
        #)
     


        #client.set_scene_item_transform()
        #itemLits =client.get_scene_item_list(name=REPLAY_SCENE)
       # print(itemLits.sceneItemId)

         # Désactiver et réactiver la source pour forcer le refresh
        # Désactiver puis réactiver la source pour forcer le reload
        #client.call(client.SetSceneItemProperties(item=MEDIA_SOURCE_NAME, visible=False, scene_name=REPLAY_SCENE))
        time.sleep(0.1)
        #client.call(client.SetSceneItemProperties(item=MEDIA_SOURCE_NAME, visible=True, scene_name=REPLAY_SCENE))

        # Basculer sur la scène Replay
        client.set_current_program_scene(REPLAY_SCENE)
        print(f"🎬 Changement de scène : {REPLAY_SCENE}")

        # Attendre la durée du replay
        time.sleep(REPLAY_DURATION)

        # Retour à la scène Live
        client.set_current_program_scene(LIVE_SCENE)
        print(f"🏌️ Retour à la scène : {LIVE_SCENE}")

        already_seen.update(new_files)




   