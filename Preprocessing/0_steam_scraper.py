# =====================================================
# FILE: Preprocessing/0_steam_scraper.py
# Deskripsi: Scraper Spesifik dengan Filter Bahasa & Aspek
# =====================================================
import requests
import time
import random
import re
import pandas as pd
import streamlit as st
from langdetect import detect, LangDetectException

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

def contains_aspect_keywords(text):
    text_clean = re.sub(r'[^\w\s]', '', text.lower())
    for keywords in aspect_keywords.values():
        if any(k in text_clean for k in keywords):
            return True
    return False

def safe_request(url, params, max_retries=3):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                time.sleep(5)
            else:
                time.sleep(2)
        except Exception:
            time.sleep(2)
    return None

# ============================
# FUNGSI UTAMA YANG DIPANGGIL APP.PY
# ============================
def scrape_steam_reviews(appid, limit=100):
    reviews = []
    cursor = "*"
    count_per_request = 100 # Ambil banyak sekaligus biar cepat
    
    # UI Progress di Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.write("🔄 Menghubungi Steam...")

    total_fetched = 0
    
    # Loop sampai target tercapai
    while len(reviews) < limit:
        url = f"https://store.steampowered.com/appreviews/{appid}"
        params = {
            "json": 1,
            "filter": "recent",
            "language": "english", # Fokus Inggris dulu karena model kita Inggris
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": count_per_request,
            "cursor": cursor
        }

        response = safe_request(url, params)
        if not response:
            st.error("Gagal terhubung ke Steam API.")
            break

        data = response.json()
        if "reviews" not in data or not data["reviews"]:
            break

        for r in data["reviews"]:
            if len(reviews) >= limit:
                break
            
            review_text = r.get("review", "").replace("\n", " ")
            if len(review_text.split()) < 3: # Skip review terlalu pendek
                continue

            # --- FILTER BAHASA & ASPEK (LOGIKA ANDA) ---
            try:
                # Deteksi bahasa (biar aman, fallback ke unknown)
                lang = detect(review_text)
            except:
                lang = "unknown"

            # Filter: Hanya ambil jika bahasa EN/ID DAN mengandung keyword aspek
            # Note: Model sentimen kita bahasa Inggris, jadi prioritas 'en'
            if lang == 'en' and contains_aspect_keywords(review_text):
                reviews.append({
                    "review": review_text,
                    "language": lang,
                    "steam_voted_up": r.get("voted_up", False)
                })

        # Update Progress Bar
        current_progress = min(len(reviews) / limit, 1.0)
        progress_bar.progress(current_progress)
        status_text.write(f"🔍 Mendapatkan {len(reviews)} / {limit} ulasan relevan...")

        # Update Cursor untuk halaman selanjutnya
        cursor = data.get("cursor")
        if not cursor:
            break
        
        # Delay sopan agar tidak di-banned Steam
        time.sleep(1)

    progress_bar.empty()
    status_text.empty()
    
    # Return DataFrame
    return pd.DataFrame(reviews)
