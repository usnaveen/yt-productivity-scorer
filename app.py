import os, re
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer, util
import gradio as gr

# YouTube API key
YT_API_KEY = os.environ.get("YOUTUBE_API_KEY")
if not YT_API_KEY:
    raise ValueError("Set YOUTUBE_API_KEY in Settings → Secrets")

youtube = build("youtube", "v3", developerKey=YT_API_KEY)

MODEL_NAMES = [
  "sentence-transformers/all-MiniLM-L6-v2",
  "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
  "sentence-transformers/paraphrase-MiniLM-L3-v2",
  "sentence-transformers/all-mpnet-base-v2",
  "sentence-transformers/distilbert-base-nli-mean-tokens",
]
models = [SentenceTransformer(name) for name in MODEL_NAMES]

def extract_video_id(url):
    patterns = [ r"v=([A-Za-z0-9_-]{11})", r"youtu\.be/([A-Za-z0-9_-]{11})" ]
    for p in patterns:
        m = re.search(p, url)
        if m: return m.group(1)
    raise ValueError("Invalid YouTube URL")

def fetch_metadata(video_url):
    vid = extract_video_id(video_url)
    resp = youtube.videos().list(part="snippet", id=vid).execute()
    items = resp.get("items", [])
    if not items: raise ValueError("Video not found")
    snip = items[0]["snippet"]
    return snip.get("title",""), snip.get("description","")

def compute_score(video_url, goal):
    title, desc = fetch_metadata(video_url)
    text = title + "\n\n" + desc
    scores = []
    for m in models:
        emb_text = m.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        emb_goal = m.encode(goal, convert_to_tensor=True, normalize_embeddings=True)
        cos_sim = util.cos_sim(emb_text, emb_goal).item()
        pct = int((cos_sim + 1) * 50)
        scores.append(max(0, min(100, pct)))
    return int(round(sum(scores)/len(scores)))

iface = gr.Interface(
  fn=compute_score,
  inputs=[ gr.Textbox(label="YouTube URL"), gr.Textbox(label="Your Goal") ],
  outputs=gr.Number(label="Score 0–100"),
  description="Average of 5 sentence-transformer models"
)

if __name__=="__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
