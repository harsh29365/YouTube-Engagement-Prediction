import re
import torch
import pandas
import requests
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import concurrent.futures
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel

data = pandas.read_csv("videos.csv")

data = data[data["views"] != 0].copy()
data["timestamp"] = pandas.to_datetime(data["upload_date"]).astype("int64") // 10 ** 9

def parse_duration(iso_string):
    if not isinstance(iso_string, str):
        raise ValueError("Input must be a string")
    
    if not iso_string.startswith('PT'):
        raise ValueError("ISO 8601 duration must start with 'PT'")
    
    duration_part = iso_string[2:]
    
    if not duration_part:
        raise ValueError("Invalid ISO 8601 duration format")
    
    total_seconds = 0
    
    hours_match = re.search(r'(\d+)H', duration_part)
    if hours_match:
        total_seconds += int(hours_match.group(1)) * 3600
    
    minutes_match = re.search(r'(\d+)M', duration_part)
    if minutes_match:
        total_seconds += int(minutes_match.group(1)) * 60
    
    seconds_match = re.search(r'(\d+)S', duration_part)
    if seconds_match:
        total_seconds += int(seconds_match.group(1))
    
    return total_seconds

data["duration(sec)"] = data["duration"].apply(parse_duration)
data = data[data["duration(sec)"] != 0].copy()
data["likes_views_ratio"] = data["likes"] / data["views"]

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
title_embeddings = model.encode(data["title"].tolist(), show_progress_bar=True)
data["title_embeddings"] = title_embeddings.tolist()

session = requests.Session()

def download_and_open_image(image_url):
    try:
        response = session.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        tqdm.write(f"Error downloading {image_url}: {e}")
        return None
    except IOError:
        tqdm.write(f"Could not open image from {image_url}")
        return None

def load_images(urls):
    images = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(download_and_open_image, urls)

        for image in tqdm(results, desc="Downloading Images"):
            images.append(image)

    return images

images = load_images(data["thumbnail_url"])

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")
model.eval()
model.to("cuda")

def images_to_vector(images_batch):
    inputs = processor(images=images_batch, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
        image_vectors = outputs.last_hidden_state[:, 0, :]

    return image_vectors

def process_images(images, batch_size):
    thumbnail_embeddings = []

    for i in tqdm(range(0, len(images), batch_size), desc="Processing Batches"):
        batch = images[i:i + batch_size]
        batch_vectors = images_to_vector(batch)
        thumbnail_embeddings.append(batch_vectors)

    thumbnail_embeddings = torch.cat(thumbnail_embeddings, dim=0)
    return thumbnail_embeddings.cpu().tolist()

thumbnail_embeddings = process_images(images, batch_size=64)
data["thumbnail_embeddings"] = thumbnail_embeddings

data.dropna(inplace=True)
data = data.drop(columns=["title", "likes", "views", "upload_date", "duration", "thumbnail_url"])
data.to_parquet("processed_videos.parquet", engine="pyarrow")
