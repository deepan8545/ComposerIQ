"""
ComposerIQ Visibility Agent
----------------------------
LangGraph agent with 4 clean layers:
  1. node_extract_features  → audio_analysis.py (measurements only)
  2. node_retrieve_benchmarks → Pinecone MERT search
  3. node_score             → mir_analyzer.py (percentiles) + scoring_engine.py (calibration)
  4. node_generate_report   → Claude explains + adjusts +-5 points
"""
import os
import json
import numpy as np
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from pipelines.audio_analysis import AudioAnalyzer
from pipelines.scoring_engine import ScoringEngine
from pipelines.retrieval import HybridRetriever
from mert_embedder import embedder as _mert_embedder
from data.mir_analyzer import MIRAnalyzer
from observability.langfuse_config import get_langfuse_handler

load_dotenv()

_analyzer = AudioAnalyzer()
_retriever = HybridRetriever()
_mir_analyzer = MIRAnalyzer()
_scoring_engine = ScoringEngine()

# Fallback cluster stats (cross-cluster median from cluster_stats.json)
_DEFAULT_CLUSTER_STATS = {
    'skip_risk_mean': 0.13,
    'skip_risk_std': 0.10,
}


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    audio_path: str
    genre: str
    mood: str
    features: dict          # Layer 1 output
    benchmarks: list        # Pinecone results
    mir: dict               # Layer 2 output (percentiles)
    scored: dict            # Layer 3 output (calibrated scores + components)
    counterfactuals: list   # Layer 3 output (what-if fixes, sorted by impact)
    report: str             # Layer 4 output (Claude text)
    engagement_score: int   # Final score (Layer 3 base + Layer 4 adjustment)
    spotifycore_score: int
    error: Optional[str]


def _safe(obj):
    """Recursively convert numpy types to plain Python types."""
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_safe(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Node 1: Feature extraction (Layer 1 — measurements only)
# ---------------------------------------------------------------------------

def node_extract_features(state: AgentState) -> AgentState:
    try:
        features = _analyzer.analyze(state["audio_path"])
        state["features"] = _safe(dict(features))
    except Exception as e:
        state["error"] = f"Feature extraction failed: {str(e)}"
    return state


# ---------------------------------------------------------------------------
# Node 2: Benchmark retrieval (Pinecone MERT search)
# ---------------------------------------------------------------------------

def node_retrieve_benchmarks(state: AgentState) -> AgentState:
    if state.get("error") and not state.get("features"):
        state["benchmarks"] = []
        return state

    query_text = state["genre"] + " " + state["mood"]

    try:
        mert_vector = _mert_embedder.embed(state["audio_path"]).tolist()
    except Exception as e:
        print(f"[Agent] MERT embed failed: {e}")
        state["error"] = f"MERT embedding failed: {str(e)}"
        state["benchmarks"] = []
        return state

    try:
        benchmarks = _retriever.retrieve(
            query_vector=mert_vector,
            query_text=query_text,
            top_k=5,
        )
        state["benchmarks"] = _safe(benchmarks)
    except Exception as e:
        state["error"] = f"Retrieval failed: {str(e)}"
        state["benchmarks"] = []
    return state


# ---------------------------------------------------------------------------
# Node 3: Scoring (Layer 2 MIR percentiles + Layer 3 calibration)
# ---------------------------------------------------------------------------

def node_score(state: AgentState) -> AgentState:
    features = state.get("features", {})
    if not features:
        state["error"] = state.get("error") or "No features available for scoring"
        return state

    try:
        from data.ingest import CLUSTER_STATS_DB
        benchmarks = state.get("benchmarks", [])
        matched_cluster_id = benchmarks[0].get("cluster_id", -1) if benchmarks else -1
        cluster_stats = CLUSTER_STATS_DB.get(
            str(matched_cluster_id), _DEFAULT_CLUSTER_STATS
        )

        # Layer 2: MIR percentiles
        mir = _mir_analyzer.analyze(
            features=features,
            cluster_stats=cluster_stats,
            genre=state.get("genre", "pop"),
        )
        state["mir"] = _safe({
            "hook_time":             mir.hook_time,
            "hook_confidence":       mir.hook_confidence,
            "hook_percentile":       mir.hook_percentile,
            "skip_risk_percentile":  mir.skip_risk_percentile,
            "factor_breakdown":      mir.factor_breakdown,
            "raw_risk":              mir.raw_risk,
            "cluster_id":            mir.cluster_id,
            "genre_tier":            mir.genre_tier,
        })

        print(f"[MIR] hook={mir.hook_time}s, hook_pct={mir.hook_percentile}, "
              f"skip_risk_pct={mir.skip_risk_percentile}, raw_risk={mir.raw_risk}")

        # Layer 3: Calibrated scores
        scored = _scoring_engine.score(features, mir)
        state["scored"] = _safe({
            "engagement_score":     scored.engagement_score,
            "spotifycore_score":    scored.spotifycore_score,
            "components":           scored.components,
        })
        # Set as baseline (Claude can adjust +-5)
        state["engagement_score"] = scored.engagement_score
        state["spotifycore_score"] = scored.spotifycore_score

        # Layer 3b: Counterfactual what-if fixes
        try:
            cf = _scoring_engine.compute_counterfactuals(
                signals=scored.components,
                features=features,
            )
            state["counterfactuals"] = _safe(cf)
        except Exception as e:
            print(f"[Agent] Counterfactuals failed: {e}")
            state["counterfactuals"] = []

    except Exception as e:
        import traceback
        traceback.print_exc()
        state["error"] = f"Scoring failed: {str(e)}"
    return state


# ---------------------------------------------------------------------------
# Node 4: Report generation (Claude explains + adjusts +-5)
# ---------------------------------------------------------------------------

def node_generate_report(state: AgentState) -> AgentState:
    features = state.get("features", {})
    mir = state.get("mir", {})
    scored = state.get("scored", {})

    if not scored:
        state["error"] = state.get("error") or "No scores available for report"
        return state

    eng_base = scored.get("engagement_score", 0)
    spot_base = scored.get("spotifycore_score", 0)
    components = scored.get("components", {})

    print(f"[Agent] Calling Claude for report (base: eng={eng_base}, spot={spot_base})...")

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2000,
        temperature=0.2,
        timeout=60,
    )

    safe_benchmarks = state.get("benchmarks", [])
    bench_lines = "\n".join([
        "  " + str(i + 1) + ". " + str(b.get("name", "?")) + " by " + str(b.get("artist", "?")) +
        " (genre=" + str(b.get("genre", "?")) + ", mood=" + str(b.get("mood", "?")) +
        ", pop=" + str(b.get("popularity", 0)) + ")"
        for i, b in enumerate(safe_benchmarks)
    ]) or "  (no benchmarks retrieved)"

    f = features

    # Build counterfactual block for Claude prompt
    cf_list = state.get("counterfactuals", [])
    if cf_list:
        cf_lines = ["COUNTERFACTUAL FIXES (computed engine, ranked by score impact):"]
        for cf in cf_list[:4]:  # top 4 by score delta
            cf_lines.append(
                f"  - {cf['signal'].upper()}: current={cf['current']:.0f}/100 → target={cf['target']:.0f}/100 "
                f"= +{cf['score_delta']:.1f} engagement pts | {cf['fix_hint']}"
            )
    else:
        cf_lines = ["COUNTERFACTUAL FIXES: (not available)"]

    prompt_lines = [
        "You are a Spotify algorithm expert explaining a track's analysis results.",
        "",
        "IMPORTANT: The scores below were computed by a calibrated scoring engine.",
        "Your job is to EXPLAIN them, not re-compute them.",
        "You may adjust each score by at most +-5 points if you have strong musical",
        "reasoning. Explain any adjustment in your report.",
        "",
        "CALIBRATED SCORES (from scoring engine):",
        f"- Engagement Score: {eng_base}/100",
        f"- Spotifycore Score: {spot_base}/100",
        "",
        "SIGNAL BREAKDOWN (each 0-100, weighted into final scores):",
        f"- Hook timing signal: {components.get('hook', '?')}/100",
        f"- Skip risk signal: {components.get('skip_inv', '?')}/100 (higher = lower risk = better)",
        f"- Energy signal: {components.get('energy', '?')}/100",
        f"- Danceability signal: {components.get('danceability', '?')}/100",
        f"- Valence signal: {components.get('valence', '?')}/100",
        "",
        "TRACK DATA:",
        f"- Genre: {state['genre']} / Mood: {state['mood']}",
        f"- Tempo: {f.get('tempo', '?')} BPM",
        f"- Key: {f.get('key', '?')}",
        f"- Duration: {f.get('duration', '?')}s",
        f"- Hook arrives at: {mir.get('hook_time', '?')}s (confidence: {mir.get('hook_confidence', '?')})",
        f"- Hook percentile: {mir.get('hook_percentile', '?')} (vs top 10% of {state['genre']})",
        f"- Skip risk percentile: {mir.get('skip_risk_percentile', '?')}",
        f"- Energy: {f.get('energy', '?')} (0-1)",
        f"- Danceability: {f.get('danceability', '?')} (0-1)",
        f"- Valence: {f.get('valence', '?')} (0=sad, 1=happy)",
        f"- Acousticness: {f.get('acousticness', '?')}",
        f"- Instrumentalness: {f.get('instrumentalness', '?')}",
        f"- Speechiness: {f.get('speechiness', '?')}",
        f"- Loudness: {f.get('loudness', '?')} dB",
        f"- Harmonic complexity: {f.get('harmonic_complexity', '?')} (0=simple, 1=rich chords)",
        f"- Tonal stability: {f.get('tonal_stability', '?')} (0=unstable, 1=stable key)",
        f"- Chord change rate: {f.get('chord_change_rate', '?')} changes/sec",
        f"- Factor breakdown: {json.dumps(mir.get('factor_breakdown', {}))}",
        "",
        "STRUCTURE (detected segments):",
        "\n".join(
            f"  {s.get('label','?').upper()} {s.get('start', 0):.1f}s – {s.get('end', 0):.1f}s"
            for s in f.get("segments", [])
        ) or "  (no segments detected)",
        "",
        "BENCHMARK TRACKS:",
        bench_lines,
        "",
    ] + cf_lines + [
        "",
        "OUTPUT — respond ONLY with valid JSON (no markdown, no backticks):",
        '{',
        f'  "engagement_score": {eng_base},',
        f'  "spotifycore_score": {spot_base},',
        '  "adjustment_reason": "<why you adjusted, or \'no adjustment\'>",',
        f'  "report": "## Engagement Score: {eng_base}/100\\n## Spotifycore Score: {spot_base}/100\\n\\n'
        '### What these scores mean\\n<2 sentences>\\n\\n'
        '### Hook Timing\\n<cite hook time and percentile>\\n\\n'
        '### Skip Risk\\n<cite skip risk percentile and factor breakdown>\\n\\n'
        '### Structure\\n<describe detected sections: where is the chorus, verse, bridge>\\n\\n'
        '### Signal Breakdown\\n<explain each of the 5 signals plus harmonic complexity>\\n\\n'
        '### Top 3 Fixes\\n'
        '<Use COUNTERFACTUAL FIXES above. Format each as: Fix N (+X.X pts): [fix_hint]. [1 sentence on production implementation.]>\\n\\n'
        '### Benchmark Comparison\\n<compare to benchmarks>\\n\\n'
        '### Bottom Line\\n<one sentence>"',
        '}',
    ]

    prompt = "\n".join(prompt_lines)


    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content
        if isinstance(raw, list):
            raw = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw
            )
        content = raw.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        data = json.loads(content)

        # Claude's scores, clamped to +-5 of the engine's baseline
        claude_eng = int(data.get("engagement_score", eng_base))
        claude_spot = int(data.get("spotifycore_score", spot_base))
        state["engagement_score"] = max(eng_base - 5, min(eng_base + 5, claude_eng))
        state["spotifycore_score"] = max(spot_base - 5, min(spot_base + 5, claude_spot))
        state["report"] = str(data.get("report", ""))

        adj = data.get("adjustment_reason", "none")
        print(f"[Agent] Claude: eng={claude_eng}, spot={claude_spot} "
              f"(clamped to {state['engagement_score']}/{state['spotifycore_score']}) "
              f"adj='{adj}'")
    except Exception as e:
        import traceback
        traceback.print_exc()
        state["error"] = "Report generation failed: " + str(e)

    return state


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

def build_agent():
    g = StateGraph(AgentState)
    g.add_node("extract_features",    node_extract_features)
    g.add_node("retrieve_benchmarks", node_retrieve_benchmarks)
    g.add_node("score",               node_score)
    g.add_node("generate_report",     node_generate_report)
    g.set_entry_point("extract_features")
    g.add_edge("extract_features",    "retrieve_benchmarks")
    g.add_edge("retrieve_benchmarks", "score")
    g.add_edge("score",               "generate_report")
    g.add_edge("generate_report",     END)
    return g.compile()


_compiled_agent = None

def _get_agent():
    global _compiled_agent
    if _compiled_agent is None:
        _compiled_agent = build_agent()
    return _compiled_agent


def run_analysis(audio_path: str, genre: str, mood: str, user_id: str = "user") -> dict:
    agent = _get_agent()
    handler = get_langfuse_handler("composeriq-visibility", user_id)
    callbacks = [handler] if handler else []

    initial_state: AgentState = {
        "audio_path": audio_path,
        "genre": genre,
        "mood": mood,
        "features": {},
        "benchmarks": [],
        "mir": {},
        "scored": {},
        "counterfactuals": [],
        "report": "",
        "engagement_score": 0,
        "spotifycore_score": 0,
        "error": None,
    }

    try:
        final = agent.invoke(initial_state, config={"callbacks": callbacks})
        f = final.get("features", {})
        return {
            "report":               final.get("report", ""),
            "engagement_score":     final.get("engagement_score", 0),
            "spotifycore_score":    final.get("spotifycore_score", 0),
            "hook_arrival_second":  int(f.get("hook_time") or 0),
            "hook_arrival_mert":    f.get("hook_time_mert"),
            "hook_confidence_mert": f.get("hook_confidence_mert", 0.0),
            "intro_skip_risk":      final.get("mir", {}).get("skip_risk_percentile", 0),
            "tempo":                f.get("tempo", 0.0),
            "key":                  f.get("key", "UNKNOWN"),
            "duration":             f.get("duration", 0.0),
            "energy_curve":         f.get("energy_curve", []),
            "segments":             f.get("segments", []),
            "harmonic_complexity":  f.get("harmonic_complexity", 0.0),
            "tonal_stability":      f.get("tonal_stability", 0.0),
            "chord_change_rate":    f.get("chord_change_rate", 0.0),
            "benchmarks":           final.get("benchmarks", []),
            "mir_analysis":         final.get("mir", {}),
            "scored":               final.get("scored", {}),
            "counterfactuals":      final.get("counterfactuals", []),
            "error":                final.get("error"),
        }

    except Exception as e:
        return {
            "report": "", "engagement_score": 0, "spotifycore_score": 0,
            "hook_arrival_second": 0, "intro_skip_risk": 0,
            "tempo": 0.0, "key": "UNKNOWN", "duration": 0.0,
            "energy_curve": [], "benchmarks": [], "mir_analysis": {},
            "scored": {}, "error": str(e),
        }
