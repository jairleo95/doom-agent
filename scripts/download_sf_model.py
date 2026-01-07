
from huggingface_hub import hf_hub_download
import shutil
import os

repo_id = "hishamcse/doom_deathmatch_bots"
dest_dir = "external/sample_factory_model"
files = {
    "checkpoint.pth": "checkpoint_p0/checkpoint_000002443_10006528.pth",
    "config.json": "config.json"
}

os.makedirs(dest_dir, exist_ok=True)

print(f"📥 Downloading files from {repo_id}...")

for local_name, remote_name in files.items():
    try:
        print(f"Fetching {remote_name}...")
        path = hf_hub_download(repo_id=repo_id, filename=remote_name)
        # Copy to destination with local name
        shutil.copy(path, os.path.join(dest_dir, local_name))
        print(f"✅ {local_name} saved.")
    except Exception as e:
        print(f"❌ Error downloading {remote_name}: {e}")

print("Done.")
