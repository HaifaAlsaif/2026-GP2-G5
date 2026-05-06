# conversation_baseline_model.py

import os
import sys
import joblib
import pandas as pd
from typing import Tuple, Optional

from sklearn.base import BaseEstimator, TransformerMixin


# ==================================================
# ✅ FIX: تعريف ItemSelector + حقنه داخل __main__
# ==================================================
class ItemSelector(BaseEstimator, TransformerMixin):
    """
    Transformer بسيط يسحب عمود معيّن من DataFrame.
    لازم اسمه يكون نفس اللي كان في التدريب: ItemSelector
    """
    def __init__(self, key: str):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # لو وصل DataFrame
        if hasattr(X, "__getitem__") and self.key in getattr(X, "columns", []):
            return X[self.key]

        # لو وصل dict-like
        if isinstance(X, dict) and self.key in X:
            return X[self.key]

        # لو وصل DataFrame لكن columns غير واضحة
        try:
            return X[self.key]
        except Exception:
            return X


# =========================
# Conversation Logistic Baseline Model
# =========================
MODEL_FILENAME = "conversation_logistic_regression.joblib"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", MODEL_FILENAME)

_conversation_model = None


def _ensure_itemselector_in_main():
    """
    ✅ مهم جداً:
    إذا المودل تدرب في كولاب/نوتبوك، غالباً تم تخزين ItemSelector كـ __main__.ItemSelector
    وقت تشغيل السيرفر، __main__ يكون app.py
    فنحقن الكلاس داخل __main__ عشان joblib يقدر يفكّه.
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "ItemSelector"):
        setattr(main_mod, "ItemSelector", ItemSelector)


def _load_model():
    """
    تحميل المودل مرة واحدة فقط (Lazy).
    """
    global _conversation_model

    if _conversation_model is not None:
        return _conversation_model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Make sure the joblib file is inside: {os.path.join(BASE_DIR, 'models')}"
        )

    # ✅ أهم خطوة قبل joblib.load
    _ensure_itemselector_in_main()

    print("⏳ Loading Conversation Logistic Regression model...")
    _conversation_model = joblib.load(MODEL_PATH)
    print("✅ Model loaded from:", MODEL_PATH)

    return _conversation_model


def predict_one_turn(text: str, prev_text: str = "") -> int:
    """
    Predict one conversation turn.

    Returns:
      0 = Human
      1 = Machine-generated
    """
    model = _load_model()

    row = {"text": text or "", "prev_text": prev_text or ""}
    df = pd.DataFrame([row])

    pred = model.predict(df)[0]
    return int(pred)


def predict_one_turn_with_confidence(text: str, prev_text: str = "") -> Tuple[int, Optional[float]]:
    """
    نفس predict_one_turn لكن يرجّع (label, confidence)
    confidence = أعلى احتمال من predict_proba إذا متوفر.

    Returns:
      (0/1, confidence or None)
    """
    model = _load_model()

    row = {"text": text or "", "prev_text": prev_text or ""}
    df = pd.DataFrame([row])

    label = int(model.predict(df)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]  # [p0, p1]
        confidence = float(max(proba))

    return label, confidence
