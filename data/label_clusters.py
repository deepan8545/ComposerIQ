import anthropic
import json


def label_cluster_with_llm(cluster_tracks: list) -> dict:
    client = anthropic.Anthropic()

    sample = cluster_tracks[:5]
    track_descriptions = [
        f"{t['name']} by {t['artist']} "
        f"(energy:{t['energy']:.2f}, valence:{t['valence']:.2f}, "
        f"tempo:{t['tempo']:.0f}, danceability:{t['danceability']:.2f}, "
        f"acousticness:{t['acousticness']:.2f})"
        for t in sample
    ]

    prompt = f"""
Here are tracks grouped together by audio similarity:
{json.dumps(track_descriptions, indent=2)}

Based on the track names, artists, and audio characteristics,
what genre or vibe best describes this group?

Reply in JSON only, no explanation, no markdown:
{{"genre": "genre name", "mood": "mood", "vibe": "one line description"}}
"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text.strip()
    return json.loads(text)


def label_all_clusters(tracks, n_clusters=12) -> tuple:
    clusters = {}
    for track in tracks:
        cid = track["cluster_id"]
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(track)

    cluster_labels = {}
    for cid, cluster_tracks in clusters.items():
        print(f"  Labeling cluster {cid} ({len(cluster_tracks)} tracks)...")
        try:
            label = label_cluster_with_llm(cluster_tracks)
            cluster_labels[cid] = label
            print(f"    → {label}")
        except Exception as e:
            print(f"    → Failed: {e}, using fallback")
            cluster_labels[cid] = {"genre": f"cluster_{cid}", "mood": "neutral", "vibe": ""}

    for track in tracks:
        label = cluster_labels[track["cluster_id"]]
        track["genre"] = label["genre"]
        track["mood"] = label["mood"]
        track["vibe"] = label["vibe"]

    return tracks, cluster_labels
