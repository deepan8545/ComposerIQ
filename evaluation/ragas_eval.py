"""
evaluation/ragas_eval.py
Quality gate — run in CI to block deployment if scores drop.
"""
import json
from typing import Optional


class RagasEvaluator:
    FAITHFULNESS_THRESHOLD = 0.85
    RELEVANCE_THRESHOLD = 0.80

    def evaluate(self, question: str, answer: str, contexts: list, ground_truth: Optional[str] = None) -> dict:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset

        data = {"question": [question], "answer": [answer], "contexts": [contexts]}
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        scores = result.to_pandas().iloc[0].to_dict()

        f = float(scores.get("faithfulness", 0))
        r = float(scores.get("answer_relevancy", 0))

        return {
            "faithfulness": round(f, 3),
            "answer_relevancy": round(r, 3),
            "passed": f >= self.FAITHFULNESS_THRESHOLD and r >= self.RELEVANCE_THRESHOLD,
            "thresholds": {"faithfulness": self.FAITHFULNESS_THRESHOLD, "answer_relevancy": self.RELEVANCE_THRESHOLD},
        }

    def run_ci_benchmark(self, test_cases_path: str) -> dict:
        with open(test_cases_path) as f:
            cases = json.load(f)

        results = [self.evaluate(c["question"], c["answer"], c["contexts"], c.get("ground_truth")) for c in cases]
        avg_f = sum(r["faithfulness"] for r in results) / len(results)
        avg_r = sum(r["answer_relevancy"] for r in results) / len(results)

        return {
            "total_cases": len(results),
            "avg_faithfulness": round(avg_f, 3),
            "avg_answer_relevancy": round(avg_r, 3),
            "passed": all(r["passed"] for r in results),
            "ci_verdict": "PASS" if all(r["passed"] for r in results) else "FAIL",
        }
