import shutil
import os

def clear_storage():
    folders_to_delete = ["/model_stt", "/model_trans", "/piper_voices", "/model_data", "/tts_models"]
    for folder in folders_to_delete:
        if os.path.exists(folder):
            print(f"🗑️ Deleting {folder}...")
            shutil.rmtree(folder)
            print("✅ Deleted.")