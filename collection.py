import os
import time
import pandas
from tqdm import tqdm
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

api_key = os.getenv("YOUTUBE_DATA_API")
channel_id = os.getenv("CHANNEL_ID")

youtube = build("youtube", "v3", developerKey=api_key)

request = youtube.channels().list(
    id=channel_id,
    part="contentDetails"
)

response = request.execute()
uploads_playlist_id = response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

video_ids = []
next_page_token = None

print("Collecting video IDs...")

try:
    request = youtube.playlistItems().list(
        maxResults=1,
        part="snippet",
        playlistId=uploads_playlist_id
    )

    response = request.execute()
    total_videos = response.get("pageInfo", {}).get("totalResults", 0)
    print(f"Total videos to collect: {total_videos}")

except Exception as e:
    print(f"An error occurred: {e}")
    total_videos = None

progress_bar = tqdm(total=total_videos, desc="Collecting Video IDs", unit="videos") if total_videos else tqdm(desc="Collecting Video IDs", unit="videos")

next_page_token = None
while True:
    request = youtube.playlistItems().list(
        maxResults=50,
        part="snippet",
        pageToken=next_page_token,
        playlistId=uploads_playlist_id
    )

    response = request.execute()

    new_videos = 0
    for item in response.get("items", []):
        video_ids.append(item["snippet"]["resourceId"]["videoId"])
        new_videos += 1

    progress_bar.update(new_videos)

    next_page_token = response.get("nextPageToken")

    if not next_page_token:
        break

    time.sleep(0.1)

progress_bar.close()
print(f"Total video IDs collected: {len(video_ids)}")

print(f"Total videos found: {len(video_ids)}")

progress_bar = tqdm(total=len(video_ids), desc="Collecting Video Details", unit="videos")

all_video_details = []

for i in range(0, len(video_ids), 50):
    batch_ids = video_ids[i:i + 50]
    ids_string = ','.join(batch_ids)

    request = youtube.videos().list(
        id=ids_string,
        part="snippet,statistics,contentDetails"
    )

    response = request.execute()

    for item in response.get("items", []):
        thumbnail_data = item["snippet"]["thumbnails"]

        video_info = {
            "title": item["snippet"]["title"],
            "likes": item["statistics"].get("likeCount", 0),
            "views": item["statistics"].get("viewCount", 0),
            "upload_date": item["snippet"]["publishedAt"],
            "duration": item["contentDetails"]["duration"],
            "thumbnail_url": thumbnail_data.get("default", {}).get("url") if thumbnail_data else None
        }

        all_video_details.append(video_info)

    progress_bar.update(len(batch_ids))
    time.sleep(0.1)

progress_bar.close()

print(f"Collected details for {len(all_video_details)} videos")

data = pandas.DataFrame(all_video_details)
print(data.head())

data.to_csv("videos.csv", index = False)