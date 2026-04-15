import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_tracks(tracks, n_clusters=12):
    vectors = []
    for t in tracks:
        vectors.append([
            t.get("danceability", 0.0),
            t.get("energy", 0.0),
            t.get("valence", 0.0),
            t.get("tempo", 120.0) / 250.0,
            t.get("acousticness", 0.0),
            t.get("instrumentalness", 0.0),
            t.get("liveness", 0.0),
            t.get("speechiness", 0.0),
        ])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(vectors)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)

    for i, track in enumerate(tracks):
        track["cluster_id"] = int(labels[i])

    return tracks, kmeans, scaler
