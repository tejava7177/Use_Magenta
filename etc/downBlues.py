import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

# 저장할 폴더
save_dir = "/Volumes/Extreme SSD/NewDataSet/blues"
os.makedirs(save_dir, exist_ok=True)

# 웹 요청 헤더 추가 (봇 차단 우회)
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}

base_url = "https://midkar.com/Blues/Blues_MIDIs.html"
response = requests.get(base_url, headers=headers)

soup = BeautifulSoup(response.text, 'html.parser')
links = soup.find_all('a')

print(f"🔍 Found {len(links)} total links.")

downloaded = 0
for link in links:
    href = link.get('href')
    if href and href.lower().endswith(".mid"):
        midi_url = urljoin(base_url, href)
        filename = os.path.basename(href)
        filepath = os.path.join(save_dir, filename)

        print(f"🎶 Downloading {filename}...")
        try:
            midi_data = requests.get(midi_url, headers=headers).content
            with open(filepath, "wb") as f:
                f.write(midi_data)
            downloaded += 1
            print(f"✅ Saved to: {filepath}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")

print(f"\n🎯 Done. Total MIDI files downloaded: {downloaded}")