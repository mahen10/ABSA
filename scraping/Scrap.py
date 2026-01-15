import os
import requests
import time
import random
import re
import pandas as pd
import nltk
import langdetect

from langdetect import detect
from openpyxl import load_workbook
from nltk.corpus import wordnet as wn


# ============================
# KAMUS ASPEK UNTUK FILTER
# ============================
aspect_keywords = {
    'graphics': [
        'graphics', 'graphic', 'visual', 'visuals', 'ui', 'design', 'grafis',
        'art', 'artstyle', 'look', 'resolution', 'texture', 'animation'
    ],

    'gameplay': [
        'gameplay', 'control', 'controls', 'mechanic', 'mechanics',
        'combat', 'movement', 'interact', 'jump', 'shoot', 'run',
        'action', 'fun', 'challenging', 'responsive',
        'attack', 'defend', 'transaction', 'transactions',
        'quest', 'quests'
    ],

    'story': [
        'story', 'plot', 'narrative', 'lore', 'writing', 'dialogue',
        'ending', 'cutscene', 'quest', 'mission', 'twist',
        'character', 'development', 'script', 'storyline'
    ],

    'performance': [
        'performance', 'lag', 'bug', 'fps', 'crash', 'glitch',
        'smooth', 'loading', 'freeze', 'stutter', 'frame',
        'drop', 'optimization', 'hang', 'delay', 'disconnect',
        'rate', 'memory', 'usage', 'rendering',
        'script', 'execution', 'garbage', 'collection'
    ],

    'music': [
        'music', 'sound', 'audio', 'sfx', 'voice', 'soundtrack',
        'background', 'ost', 'noise', 'volume', 'acting',
        'ambience', 'effect', 'quality', 'melody',
        'instrumental', 'harmony', 'song'
    ]
}


def contains_aspect_keywords(text, aspect_dict):
    text_clean = re.sub(r'[^\w\s]', '', text.lower())
    for keywords in aspect_dict.values():
        if any(k in text_clean for k in keywords):
            return True
    return False

# ============================
# FUNGSI: safe_request
# ============================
def safe_request(url, headers, params, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                print("Rate limit hit. Sleeping for 5 seconds...")
                time.sleep(5)
            else:
                print(f"Unexpected status: {response.status_code}. Retrying...")
                time.sleep(3)
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(3)
    return None

# ============================
# FUNGSI: Scrape dan Simpan Excel
# ============================
def get_filtered_reviews_excel(appid, num_reviews, filename="DataSet.xlsx"):
    reviews = []
    cursor = "*"
    count_per_request = 20
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    while len(reviews) < num_reviews:
        url = f"https://store.steampowered.com/appreviews/{appid}"
        params = {
            "json": 1,
            "filter": "recent",
            "language": "english",
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": count_per_request,
            "cursor": cursor
        }

        response = safe_request(url, headers, params)
        if not response:
            print("Gagal ambil data.")
            break

        data = response.json()
        if "reviews" not in data:
            break

        for r in data["reviews"]:
            review_text = r.get("review", "").replace("\n", " ")
            if review_text.strip() == "":
                continue

            try:
                lang = detect(review_text)
            except:
                lang = "unknown"

            if lang in ["id", "en"] and contains_aspect_keywords(review_text, aspect_keywords):
                reviews.append({
                    "appid": appid,
                    "recommendation": r.get("recommendationid", ""),
                    "review": review_text,
                    "voted_up": r.get("voted_up", ""),
                    "author_steamid": r.get("author", {}).get("steamid", ""),
                    "timestamp_created": r.get("timestamp_created", ""),
                    "language": lang
                })

        cursor = data.get("cursor")
        if not cursor or len(data["reviews"]) == 0:
            break

        time.sleep(random.uniform(1, 3))

    # Buat DataFrame dari hasil review
    df_new = pd.DataFrame(reviews)

    # Simpan atau gabung ke Excel
    if os.path.exists(filepath):
        with pd.ExcelWriter(filepath, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Baca file lama
            df_existing = pd.read_excel(filepath)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_excel(writer, index=False)
        print(f"Data ditambahkan ke file {filepath}.")
    else:
        df_new.to_excel(filepath, index=False)
        print(f"File baru dibuat: {filepath}.")

# ============================
# EKSEKUSI
# ============================
appid = "854570"  # Ganti dengan appid game yang kamu inginkan
get_filtered_reviews_excel(appid, num_reviews=105, filename="DataSet.xlsx")
