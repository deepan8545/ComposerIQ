import os
import cohere
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Index name constants
# ---------------------------------------------------------------------------
_MERT_INDEX_NAME = os.getenv("PINECONE_MERT_INDEX_NAME", "composeriq-mert-768")


def _parse_meta(m) -> dict:
    meta   = m.get("metadata", {}) if isinstance(m, dict) else m.metadata
    raw_id = m["id"] if isinstance(m, dict) else m.id
    raw_pop = meta.get("popularity", 0)
    try:
        popularity = int(float(raw_pop))
    except (ValueError, TypeError):
        popularity = 0
    return {
        "track_id":      raw_id,
        "name":          meta.get("name",   "Unknown"),
        "artist":        meta.get("artist", "Unknown"),
        "genre":         meta.get("genre",  "unknown"),
        "mood":          meta.get("mood",   "neutral"),
        "popularity":    popularity,
        "cluster_id":    int(meta.get("cluster_id", -1)),
        "pinecone_score": float(m.get("score", 0.0) if isinstance(m, dict) else m.score),
    }


class HybridRetriever:
    def __init__(self):
        self._mert_index    = None   # 768-dim MERT index
        self._cohere_client = None
        self.demo_mode = False

        try:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))

            existing = {idx.name for idx in pc.list_indexes()}

            if _MERT_INDEX_NAME in existing:
                self._mert_index = pc.Index(_MERT_INDEX_NAME)
                stats = self._mert_index.describe_index_stats()
                count = getattr(stats, "total_vector_count", 0)
                print(f"[Retriever] MERT index '{_MERT_INDEX_NAME}'  ({count} vectors)")
            else:
                print(f"[Retriever] MERT index '{_MERT_INDEX_NAME}' not found — "
                      "run migrate_to_mert.py to populate it")
                raise RuntimeError(f"MERT index '{_MERT_INDEX_NAME}' not found in Pinecone")

            self._cohere_client = cohere.Client(os.getenv("COHERE_API_KEY", ""))

        except Exception as e:
            print(f"Warning: Retriever init failed: {e}. Falling back to demo mode.")
            self.demo_mode = True

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def retrieve(self, query_vector: list, query_text: str,
                 top_k: int = 5) -> list:
        """
        Query composeriq-mert-768 and return re-ranked benchmark candidates.

        query_vector — 768-dim MERT embedding
        """
        if self.demo_mode:
            return []

        if self._mert_index is not None:
            try:
                candidates = self._query_index(
                    self._mert_index, query_vector, top_k, label="MERT"
                )
                if candidates:
                    return self._rerank(candidates, query_text, top_k)
                print("[Retriever] MERT index returned no candidates")
            except Exception as e:
                print(f"[Retriever] MERT index query failed: {e}")

        print("[Retriever] No candidates from any index")
        return []

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _query_index(self, index, vector: list, top_k: int,
                     label: str = "") -> list:
        safe_vector = [float(v) for v in vector]
        candidates  = []

        for min_pop in [50, 20, 0]:
            filter_dict = {"popularity": {"$gte": min_pop}} if min_pop > 0 else None
            try:
                result  = index.query(
                    vector=safe_vector,
                    top_k=top_k * 6,
                    include_metadata=True,
                    filter=filter_dict,
                )
                matches = result.get("matches", []) if isinstance(result, dict) \
                          else result.matches
                if len(matches) >= top_k:
                    print(f"[Retriever] {label}: {len(matches)} matches "
                          f"(popularity >= {min_pop})")
                    candidates = matches
                    break
                print(f"[Retriever] {label}: only {len(matches)} matches "
                      f"at popularity >= {min_pop}, retrying ...")
            except Exception as e:
                print(f"[Retriever] {label} filter query failed ({e}), trying no filter")
                result    = index.query(vector=safe_vector,
                                        top_k=top_k * 6,
                                        include_metadata=True)
                candidates = result.get("matches", []) if isinstance(result, dict) \
                             else result.matches
                break

        return [_parse_meta(m) for m in candidates]

    def _rerank(self, parsed: list, query_text: str, top_k: int) -> list:
        # BM25 pass
        tokenized = [
            (c["genre"] + " " + c["mood"] + " " + c["name"]).lower().split()
            for c in parsed
        ]
        bm25        = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(query_text.lower().split())
        ranked = sorted(
            [dict(c, bm25_score=float(s)) for c, s in zip(parsed, bm25_scores)],
            key=lambda x: x["bm25_score"], reverse=True,
        )

        # Cohere rerank
        try:
            docs = [
                f"Genre: {c['genre']}. Mood: {c['mood']}. "
                f"Track: {c['name']}. Artist: {c['artist']}. "
                f"Popularity: {c['popularity']}."
                for c in ranked
            ]
            resp = self._cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query_text,
                documents=docs,
                top_n=top_k,
            )
            return [
                dict(ranked[r.index], rerank_score=float(r.relevance_score))
                for r in resp.results
            ]
        except Exception as e:
            print(f"[Retriever] Cohere rerank failed: {e}. Using BM25 top {top_k}.")
            return ranked[:top_k]
