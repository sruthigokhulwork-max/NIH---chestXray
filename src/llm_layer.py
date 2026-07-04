# src/llm_layer.py
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import LABELS
from src.rag_pipeline import KnowledgeBaseRetriever

# CONFIDENCE LEVEL MAPPING

def get_confidence_label(probability):

    # Converts a raw probability into uncertainty-aware language. The PDF explicitly requires "uncertainty-aware phrasing" and 
    # the system must "avoid definitive diagnosis language." Directly stating probabilities as facts would violate this — instead,
    # we translate them into qualitative, appropriately hedged language.
    
    if probability >= 0.80:
        return "high confidence"
    elif probability >= 0.60:
        return "moderate confidence"
    elif probability >= 0.40:
        return "low confidence (included for completeness)"
    else:
        return None  # Too uncertain — won't be reported at all


# FILTER PREDICTIONS ABOVE REPORTING THRESHOLD
def filter_significant_predictions(predictions_dict, min_threshold=0.40):
 
    # Takes raw model output (disease -> probability for ALL 14 diseases)
    # and keeps only those worth reporting (above min_threshold).

    # WHY filter: Reporting all 14 diseases regardless of confidence
    # would create a noisy, unhelpful report. We only surface findings
    # that have at least SOME meaningful signal.

    # Args:
    #     predictions_dict : dict like {"Edema": 0.88, "Mass": 0.12, ...}
    #     min_threshold     : minimum probability to include in report

    # Returns:
    #     List of (disease, probability) tuples, sorted by probability
    #     descending (most confident finding first)

    filtered = [(disease, prob) for disease, prob in predictions_dict.items()
               if prob >= min_threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered


# GENERATE THE STRUCTURED, CITED REPORT

def generate_grounded_report(predictions_dict, retriever, image_id="N/A"):

    # The core RAG-grounded generation function. Combines model predictions
    # with retrieved knowledge base text into a structured report.

    # WHY "grounded": Every claim about a disease's characteristics comes
    # DIRECTLY from our curated knowledge base (retrieved via FAISS), never
    # from the LLM's general "memory" — this prevents hallucination and
    # satisfies the PDF's grounding requirement: "cite retrieved snippets,
    # avoid unsupported claims."

    # Args:
    #     predictions_dict : {"Edema": 0.88, "Consolidation": 0.79, ...}
    #     retriever         : a KnowledgeBaseRetriever instance (from rag_pipeline.py)
    #     image_id          : identifier for the X-ray being analyzed (for the report header)

    # Returns:
    #     A formatted string — the complete structured report

    significant = filter_significant_predictions(predictions_dict, min_threshold=0.40)

    if not significant:
        return _build_no_findings_report(image_id)

    # Retrieve knowledge base entries relevant to these specific findings
    disease_names = [d for d, p in significant]
    retrieved_chunks = retriever.retrieve(disease_names, top_k=len(disease_names))

    # Build a lookup: disease name -> its knowledge base text
    # WHY: ensures we cite the CORRECT entry for each disease, even though retrieval might return semantically similar 
    # (but different) entries — we prioritize EXACT label matches when available
    kb_lookup = {chunk["label"]: chunk["text"] for chunk in retriever.chunks}

    # ── Build the report header ──────────────────────────────────────────────
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("AI-ASSISTED CHEST X-RAY INTERPRETATION REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Image ID: {image_id}")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Model: ResNet-18 (Test Set Mean AUROC: 0.7208)")
    report_lines.append("")

    # ── Build the disclaimer (MANDATORY per project requirements) ─────────────
    report_lines.append(" IMPORTANT DISCLAIMER")
    report_lines.append("-" * 70)
    disclaimer_text = kb_lookup.get("GENERAL_DISCLAIMER",
        "This system is for research purposes only and is not a diagnostic tool.")
    report_lines.append(disclaimer_text)
    report_lines.append("")

    # ── Build the findings section ─────────────────────────────────────────────
    report_lines.append("FINDINGS SUMMARY")
    report_lines.append("-" * 70)

    for disease, probability in significant:
        confidence_label = get_confidence_label(probability)
        disease_kb_key = disease.upper().replace(" ", "_")

        report_lines.append(f"\n📍 {disease} — {confidence_label} ({probability:.1%})")

        # Retrieve the EXACT knowledge base entry for this disease
        if disease_kb_key in kb_lookup:
            kb_text = kb_lookup[disease_kb_key]
            # WHY truncate: keep report readable; full text available
            # in knowledge_base/thoracic_kb.txt for reference
            summary_sentence = kb_text.split(". ")[0] + "."
            report_lines.append(f"   {summary_sentence}")
            report_lines.append(f"   [Source: knowledge_base/thoracic_kb.txt — {disease_kb_key}]")
        else:
            report_lines.append(f"   [No matching knowledge base entry found — flagged for review]")

    # ── Build the limitations footer ───────────────────────────────────────────
    report_lines.append("\n" + "-" * 70)
    report_lines.append("MODEL LIMITATIONS & UNCERTAINTY NOTES")
    report_lines.append("-" * 70)
    report_lines.append(
        "This model was trained on a limited subset (4,999 images) of the "
        "full NIH ChestX-ray14 dataset. Performance varies significantly by "
        "disease — diseases with few training examples (e.g., Hernia, "
        "Pneumonia) show notably lower reliability. All findings require "
        "verification by a qualified radiologist before any clinical action."
    )
    report_lines.append("=" * 70)

    return "\n".join(report_lines)


def _build_no_findings_report(image_id):

    # Special case: when NO disease crosses the minimum confidence threshold.
    # WHY a separate function: avoids awkward "no significant findings" text
    # being mixed into the main report logic, keeping things readable.

    lines = []
    lines.append("=" * 70)
    lines.append("AI-ASSISTED CHEST X-RAY INTERPRETATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Image ID: {image_id}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("FINDINGS SUMMARY")
    lines.append("-" * 70)
    lines.append("No findings exceeded the minimum confidence threshold (40%)")
    lines.append("for reporting. This does NOT confirm the absence of disease —")
    lines.append("it indicates the model did not detect strong signals for any")
    lines.append("of the 14 trained disease categories in this image.")
    lines.append("")
    lines.append("  This is an assistive tool only. A qualified radiologist")
    lines.append("    should independently evaluate this image.")
    lines.append("=" * 70)
    return "\n".join(lines)

# QUICK TEST
def test_llm_layer():
    print("\n TESTING LLM LAYER (Template-Based RAG Generation)")


    retriever = KnowledgeBaseRetriever()

    # Simulate realistic predictions (matching our actual Grad-CAM test results)
    sample_predictions = {
        "Edema": 0.881,
        "Consolidation": 0.795,
        "Pneumothorax": 0.793,
        "Effusion": 0.423,
        "Atelectasis": 0.379,
        "Cardiomegaly": 0.298,
        "Mass": 0.05,
        "Hernia": 0.02
    }

    report = generate_grounded_report(sample_predictions, retriever,
                                      image_id="test_00001330_004.png")

    print("\n" + report)

    print("\n LLM LAYER TEST COMPLETE!")


if __name__ == "__main__":
    test_llm_layer()