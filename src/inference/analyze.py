import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import re

# ======================================================
# FEATURE ENGINEERING (MUST MATCH TRAINING)
# ======================================================
def engineer_features(text):
    words = re.findall(r"\b\w+\b", str(text))
    return {
        "text": text,
        "char_length": len(str(text)),
        "word_count": len(words),
        "exclaim_count": str(text).count("!"),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0.0
    }

# Ensure we can import from the same level
sys.path.append(str(Path(__file__).resolve().parents[1]))
from loggers.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
# Use the robust 'no leakage' pipeline from notebooks/model_artifacts
# This pipeline handles the full DataFrame (text + engineered features)
model_path = BASE_DIR / "notebooks/model_artifacts/logistic_regression_no_leakage_model.pkl"
try:
    model = joblib.load(model_path)
    # Extract vectorizer for the gibberish check in line 54
    vectorizer = model.named_steps['preprocessor'].named_transformers_['text']
except Exception as e:
    # Fallback to standard locations if pipeline load fails
    model = joblib.load(BASE_DIR / "model_artifacts/model_logistic_regression.pkl")
    vectorizer = joblib.load(BASE_DIR / "model_artifacts/vectorizer.pkl")

HIGH_RISK_RULES = [
    "cure permanently",
    "cures permanently",
    "without doctors",
    "no medicine",
    "guaranteed cure",
    "miracle cure",
    "secret cure",
    "doctors hide",
    "banned cure"
]

def analyze_text(text: str):
    logger.info(f"--- Analysis Started for input subset: {text[:40]}... ---")
    
    try:
        # Engineer features and create DataFrame
        features = engineer_features(text)
        df_input = pd.DataFrame([features])
        
        # We still vectorize manually for the "gibberish" check if needed, 
        # but the model predict now uses the whole DataFrame.
        X_vec = vectorizer.transform([text])

        # Handle gibberish / empty vocabulary
        if X_vec.nnz == 0:
            logger.warning("No vocabulary overlap detected (Gibson/Noise text).")
            return {
                "risk_score": 0,
                "classification": "Unknown",
                "confidence_score": 0,
                "verdict": "Input does not contain meaningful medical language.",
                "explanation": "No valid linguistic features were detected by the model.",
                "highlighted_phrases": []
            }

        prob = model.predict_proba(df_input)[0][1]
        risk_score = int(prob * 100)
        logger.info(f"ML probability: {prob:.4f}")

        # Rule-based override
        text_l = text.lower()
        rule_hit = False
        if any(rule in text_l for rule in HIGH_RISK_RULES):
            risk_score = max(risk_score, 85)
            rule_hit = True
            logger.info("Safety rule triggered -> Score forced to 85+")

        # Risk labels
        if risk_score < 20:
            classification = "Reliable"
            verdict = "The content aligns with evidence-based medical language."
        elif risk_score < 40:
            classification = "Low Risk"
            verdict = "The content contains mild or unclear medical claims."
        elif risk_score < 60:
            classification = "Moderate Risk"
            verdict = "The content shows patterns seen in misleading health claims."
        else:
            classification = "High Risk"
            verdict = "The content strongly resembles known health misinformation."

        confidence = int(max(prob, 1 - prob) * 100)
        logger.info(f"Final Class: {classification} (Confidence: {confidence}%)")

        # Explainability
        # Dynamically find the model step (it could be 'model', 'lr', 'svc', etc.)
        if hasattr(model, "named_steps"):
            # Assume the last step is the model
            step_names = list(model.named_steps.keys())
            actual_model = model.named_steps[step_names[-1]]
        else:
            actual_model = model
        
        # Handle CalibratedClassifierCV wrapper
        inner_model = actual_model.calibrated_classifiers_[0].estimator if hasattr(actual_model, "calibrated_classifiers_") else actual_model

        # Get coefficients (for LR, Linear SVM) or feature importances (for RF/DT)
        feature_names = vectorizer.get_feature_names_out()
        scores = np.zeros(len(feature_names)) # Initialize scores

        if hasattr(inner_model, "coef_"):
            coef = inner_model.coef_[0]
            # The coefficients correspond to all features (text + numeric)
            # We only use the text part to match our X_vec for phrase highlighting
            text_size = len(feature_names)
            text_coefs = coef[:text_size]
            scores = X_vec.toarray()[0] * text_coefs
        elif hasattr(inner_model, "feature_importances_"):
            # For RF/DT, token-level attribution is complex
            pass 
        
        # Prepare features for coefficient analysis
        top_idx = np.argsort(np.abs(scores))[-6:]

        highlighted = []
        for i in top_idx:
            if scores[i] != 0:
                highlighted.append({
                    "text": feature_names[i],
                    "type": "danger" if scores[i] > 0 else "info"
                })

        return {
            "risk_score": risk_score,
            "classification": classification,
            "confidence_score": confidence,
            "verdict": verdict,
            "explanation": (
                "Prediction based on calibrated TF-IDF + model "
                "probabilities combined with medical safety rules."
            ),
            "highlighted_phrases": highlighted
        }
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        raise
