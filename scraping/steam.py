import requests
import pandas as pd
from tqdm import tqdm

# --- Daftar keyword Indonesia ---
indonesia_keywords = [
    "toge productions", "mojiken", "digital happiness", "ekuator", "gamechanger",
    "jakarta", "bandung", "surabaya", "indonesia", "masshive", "semisoft",
    "studio namaapa", "tahoe games", "anjing nabrak", "khayalan arts", "rolling glory",
    "creacle", "anoman", "berangin", "enthrean", "unreal dreamers"
]

def get_app_list():
    url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
    response = requests.get(url)
    data = response.json()
    return data['applist']['apps']

def get_app_details(appid):
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&cc=us&l=en"
    response = requests.get(url)
    try:
        data = response.json()
        if data[str(appid)]['success']:
            return data[str(appid)]['data']
    except:
        return None
    return None

# Ambil data semua aplikasi di Steam
all_apps = get_app_list()

# Untuk testing cepat, batasi jumlah game dulu
sampled_apps = [app for app in all_apps if app['name'].strip() != ''][:1000]

results = []

print("Scanning game list...")
for app in tqdm(sampled_apps):
    detail = get_app_details(app['appid'])
    if detail and 'developers' in detail:
        dev = detail.get('developers', [''])[0].lower()
        pub = detail.get('publishers', [''])[0].lower()
        game_name = detail.get('name', '')
        appid = detail.get('steam_appid', '')

        # Cek apakah developer/publisher mengandung keyword Indonesia
        if any(keyword in dev or keyword in pub for keyword in indonesia_keywords):
            results.append({
                "AppID": appid,
                "Name": game_name,
                "Developer": dev,
                "Publisher": pub,
                "Steam_URL": f"https://store.steampowered.com/app/{appid}"
            })

# Simpan ke CSV
df = pd.DataFrame(results)
df.to_csv("game_developer_indonesia.csv", index=False)
print(f"Done! Ditemukan {len(df)} game buatan developer Indonesia.")
