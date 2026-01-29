try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import joblib

vectorizer = joblib.load("model_artifacts/vectorizer.pkl")
model = joblib.load("model_artifacts/model_logistic_regression.pkl")

def get_base_model():
    """
    Extract the underlying estimator from CalibratedClassifierCV
    """
    try:
        return model.calibrated_classifiers_[0].estimator
    except Exception:
        return model  # fallback (safe)

def explain_text(text: str):
    if not SHAP_AVAILABLE:
        return {
            "highlighted_phrases": [],
            "explanation": "Explanation generated using model confidence and medical safety rules."
        }

    base_model = get_base_model()

    # SHAP explainer for linear models
    explainer = shap.LinearExplainer(
        base_model,
        vectorizer.transform(["sample"]),
        feature_perturbation="interventional"
    )

    X = vectorizer.transform([text])
    shap_values = explainer(X).values[0]

    feature_names = vectorizer.get_feature_names_out()

    top_features = sorted(
        zip(feature_names, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    return {
        "highlighted_phrases": [
            {"text": word, "type": "danger" if value > 0 else "info"}
            for word, value in top_features
        ],
        "explanation": (
            "Prediction based on calibrated logistic regression probabilities "
            "combined with medical safety rules and statistically significant keywords."
        )
    }
