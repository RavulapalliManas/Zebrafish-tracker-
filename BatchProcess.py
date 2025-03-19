import os
import cv2
import csv
import numpy as np
from tqdm import tqdm
from Main import (
    check_video_path,
    initialize_video_capture,
    log_video_info,
    process_frame,
    draw_fish_contours,
    write_center_data,
    BoxManager
)
import requests
from msal import ConfidentialClientApplication
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Constants for OneDrive API
CLIENT_ID = 'your-client-id'
CLIENT_SECRET = 'your-client-secret'
TENANT_ID = 'your-tenant-id'
AUTHORITY = f'https://login.microsoftonline.com/{TENANT_ID}'
SCOPE = ['https://graph.microsoft.com/.default']
ONEDRIVE_API_URL = 'https://graph.microsoft.com/v1.0/me/drive/root:/path/to/your/videos:/children'

# Initialize MSAL client
app = ConfidentialClientApplication(
    CLIENT_ID,
    authority=AUTHORITY,
    client_credential=CLIENT_SECRET
)

def get_access_token():
    """
    Acquire an access token for Microsoft Graph API using MSAL.

    Returns:
        str: Access token.z
    """
    result = app.acquire_token_for_client(scopes=SCOPE)
    return result['access_token']

def list_files_onedrive(access_token):
    """
    List files in the specified OneDrive directory.

    Args:
        access_token (str): Access token for authentication.

    Returns:
        list: List of files in the directory.
    """
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(ONEDRIVE_API_URL, headers=headers)
    response.raise_for_status()
    return response.json()['value']

def download_file_onedrive(access_token, file_id, file_name):
    """
    Download a file from OneDrive.

    Args:
        access_token (str): Access token for authentication.
        file_id (str): ID of the file to download.
        file_name (str): Local name to save the downloaded file.
    """
    download_url = f'https://graph.microsoft.com/v1.0/me/drive/items/{file_id}/content'
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(download_url, headers=headers, stream=True)
    response.raise_for_status()
    with open(file_name, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def delete_file_onedrive(access_token, file_id):
    """
    Delete a file from OneDrive.

    Args:
        access_token (str): Access token for authentication.
        file_id (str): ID of the file to delete.
    """
    delete_url = f'https://graph.microsoft.com/v1.0/me/drive/items/{file_id}'
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.delete(delete_url, headers=headers)
    response.raise_for_status()

def list_files_gdrive(drive, folder_id):
    """
    List files in the specified Google Drive folder.

    Args:
        drive (GoogleDrive): Authenticated GoogleDrive instance.
        folder_id (str): ID of the Google Drive folder.

    Returns:
        list: List of files in the folder.
    """
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    return file_list

def download_file_gdrive(drive, file_id, file_name):
    """
    Download a file from Google Drive.

    Args:
        drive (GoogleDrive): Authenticated GoogleDrive instance.
        file_id (str): ID of the file to download.
        file_name (str): Local name to save the downloaded file.
    """
    file = drive.CreateFile({'id': file_id})
    file.GetContentFile(file_name)

def delete_file_gdrive(drive, file_id):
    """
    Delete a file from Google Drive.

    Args:
        drive (GoogleDrive): Authenticated GoogleDrive instance.
        file_id (str): ID of the file to delete.
    """
    file = drive.CreateFile({'id': file_id})
    file.Delete()

def input_box_coordinates():
    """
    Prompt the user to input box coordinates.

    Returns:
        dict: Box data with coordinates.
    """
    num_boxes = int(input("Enter the number of boxes: "))
    box_data = {}
    for i in range(num_boxes):
        print(f"Enter coordinates for Box {i+1} (format: x1,y1 x2,y2 x3,y3 x4,y4): ")
        coords_input = input()
        coords = [tuple(map(int, point.split(','))) for point in coords_input.split()]
        box_data[f"Box {i+1}"] = {"coords": coords}
    return box_data

def process_video(file_name, box_data, output_dir):
    """
    Process a single video file to generate CSVs for data, coords, and box data.

    Args:
        file_name (str): Name of the video file.
        box_data (dict): Box data for processing.
        output_dir (str): Directory to save the output CSV files.
    """
    check_video_path(file_name)
    cap = initialize_video_capture(file_name)
    log_video_info(cap)

    video_filename = os.path.splitext(os.path.basename(file_name))[0]
    os.makedirs(output_dir, exist_ok=True)

    coord_filename = os.path.join(output_dir, f"coord_{video_filename}.csv")
    with open(coord_filename, 'w', newline='') as coord_file:
        coord_writer = csv.writer(coord_file)
        coord_writer.writerow(["box_name", "coordinates"])
        for box_name, box_info in box_data.items():
            coord_writer.writerow([box_name, box_info["coords"]])

    data_filename = os.path.join(output_dir, f"data_{video_filename}.csv")
    with open(data_filename, 'w', newline='') as data_file:
        data_writer = csv.writer(data_file)
        data_writer.writerow(["box_name", "time_spent (s)", "average_speed (px/s)"])

        center_filename = os.path.join(output_dir, f"center_{video_filename}.csv")
        with open(center_filename, 'w', newline='') as center_file:
            center_writer = csv.writer(center_file)
            center_writer.writerow(["frame", "contour_id", "center_x (px)", "center_y (px)", "instantaneous_speed (px/s)"])

            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            time_spent = [0] * len(box_data)

            clahe = cv2.createCLAHE(clipLimit=0.85, tileGridSize=(8,8))
            fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

            previous_center = None
            total_speed = 0
            speed_count = 0

            # Progress bar for individual video processing
            video_pbar = tqdm(total=total_frames, desc=f"Processing {video_filename}", unit="frame", dynamic_ncols=True)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                enhanced, contours, current_center = process_frame(frame, fgbg, clahe, 39, 1.0)

                if current_center and previous_center:
                    dx = current_center[0] - previous_center[0]
                    dy = current_center[1] - previous_center[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    instantaneous_speed = distance * original_fps
                    total_speed += instantaneous_speed
                    speed_count += 1
                else:
                    instantaneous_speed = 0

                previous_center = current_center

                contour_areas = []
                for idx, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if area < 10:
                        continue
                    contour_areas.append(area)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        write_center_data(center_writer, frame_count, idx, center_x, center_y, instantaneous_speed)

                draw_fish_contours(enhanced, contours, list(box_data.values()), time_spent, original_fps, contour_areas=contour_areas)

                video_pbar.update(1)
                frame_count += 1

            video_pbar.close()
            cap.release()
            cv2.destroyAllWindows()

            for i, (box_name, box_info) in enumerate(box_data.items()):
                box_info["time"] = time_spent[i]
                average_speed = total_speed / speed_count if speed_count > 0 else 0
                data_writer.writerow([box_name, time_spent[i], average_speed])

def process_files_in_batches():
    """
    Process video files in batches from OneDrive or Google Drive, ensuring each video file
    results in the creation of three CSV files: data, coords, and box data.
    """
    storage_choice = input("Choose storage option (1: OneDrive, 2: Google Drive, 3: Local): ").strip()

    if storage_choice == '1':
        access_token = get_access_token()
        files = list_files_onedrive(access_token)
        download_func = download_file_onedrive
        delete_func = delete_file_onedrive
    elif storage_choice == '2':
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)
        folder_id = input("Enter Google Drive folder ID: ").strip()
        files = list_files_gdrive(drive, folder_id)
        download_func = download_file_gdrive
        delete_func = delete_file_gdrive
    elif storage_choice == '3':
        directory_path = input("Enter the local directory path for input videos: ").strip()
        files = [{'name': f, 'id': f} for f in os.listdir(directory_path) if f.endswith(('.mp4', '.mov', '.avi'))]
        download_func = lambda *args: None  # No download needed for local files
        delete_func = lambda *args: None  # No delete needed for local files
    else:
        print("Invalid choice. Exiting.")
        return

    output_base_dir = input("Enter the output directory path: ").strip()

    batch_size = 5
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        for file in batch:
            file_id = file['id']
            file_name = file['name']
            if storage_choice != '3':  # Skip download for local files
                download_func(file_id, file_name)
            
            # Input box coordinates for the video
            box_data = input_box_coordinates()
            
            # Process the video file to generate CSVs
            output_dir = os.path.join(output_base_dir, os.path.splitext(file_name)[0])
            process_video(file_name, box_data, output_dir)
            
            # Delete the file after processing if not local
            if storage_choice != '3':
                delete_func(file_id)

if __name__ == "__main__":
    process_files_in_batches() 