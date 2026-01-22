from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pathlib import Path
import io
import os

from dotenv import load_dotenv

load_dotenv()

print('Choose the Dataset \n1: Learning Progress \n2: Autoscaling')
i = int(input())

if i == 1:
    FOLDER_ID = os.getenv("LEARNING_KEY")
    CREDS = os.getenv("GOOGLE_API_CREDS")
    out = Path("data/learning_progress")
elif i == 2:
    FOLDER_ID = os.getenv("SCALING_KEY")
    CREDS = os.getenv("GOOGLE_API_CREDS")
    out = Path("data/autoscaling")
else:
    raise ValueError("Invalid choice")


creds = service_account.Credentials.from_service_account_file(
    CREDS,
    scopes=["https://www.googleapis.com/auth/drive.readonly"],
)

service = build("drive", "v3", credentials=creds)

out.mkdir(parents=True, exist_ok=True)

query = f"'{FOLDER_ID}' in parents and trashed = false"
results = service.files().list(q=query, fields="files(id, name)").execute()

for f in results.get("files", []):
    file_id = f["id"]
    name = f["name"]

    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(out / name, "wb")
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Downloading {name}: {int(status.progress() * 100)}%")
