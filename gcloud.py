import os
from dotenv import load_dotenv
from google.cloud import storage
from tqdm import tqdm

STATIC_AI_PROJECT = "destino/"

# Add the entire folder
def auth_gcloud():
    try:
        load_dotenv()
    except EnvironmentError:
        print("Please add the cloud key json file")


def upload_blob(dir_path, game_name, run_id):
    # Uploads a file to the Google Cloud Storage bucket.

    auth_gcloud()
    storage_client = storage.Client()
    assert os.path.isdir(dir_path)
    files = [
        f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
    ]
    for file in files:
        localFile = os.path.join(dir_path, file)
        bucket_name = os.getenv("STORAGE_BUCKET")
        bucket = storage_client.bucket(bucket_name)
        destination = os.path.join(STATIC_AI_PROJECT, game_name, run_id, file)
        blob = bucket.blob(destination)
        blob.upload_from_filename(localFile)

    print(f"File {localFile} uploaded to {destination}.")


def download_blob(game_name, run_id):
    # Download a file from the Google Cloud Storage bucket.
    auth_gcloud()
    storage_client = storage.Client()
    bucket_name = "zdresearch"
    bucket = storage_client.bucket(bucket_name)
    source_blob_name = os.path.join(STATIC_AI_PROJECT, game_name, run_id)
    blobs = storage_client.list_blobs(bucket, prefix=source_blob_name)
    destination_folder_name = os.path.join("results", game_name, run_id)
    os.makedirs(destination_folder_name, exist_ok=True)
    for blob in blobs:
        file_name = blob.name.split("/")[-1]
        destination_file_name = os.path.join(destination_folder_name, file_name)
        with open(destination_file_name, 'wb') as f:
            with tqdm.wrapattr(f, "write", total=blob.size) as file_obj:
                storage_client.download_blob_to_file(blob, file_obj)

    print("Blob {} downloaded to {}.".format(source_blob_name, destination_folder_name))
    return destination_folder_name


def get_blobs_list(game_name):
    auth_gcloud()
    storage_client = storage.Client()
    bucket_name = "zdresearch"
    prefix = os.path.join(STATIC_AI_PROJECT, game_name)
    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    return [blob for blob in blobs if blob.name[-1] != "/"]
