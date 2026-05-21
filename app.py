from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
from firebase_admin_setup import db
from firebase_admin import db as rtdb  # Realtime Database
from firebase_admin import auth as admin_auth
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from auth_rest import signup as rest_signup, signin as rest_signin, send_password_reset
from datetime import datetime
from google.cloud import storage
from flask import flash
import uuid
import json
import csv
import io
import joblib
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import re
import hashlib
import math
from html import escape
from llm_service import generate_reply
from conversation_baseline_model import predict_one_turn
from flask import abort
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# إعداد Flask
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "CHANGE_THIS_SECRET_IN_ENV_OR_CONFIG" 

# =========================
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        try:
            return data[self.key]
        except Exception:
            return data.loc[:, self.key]
# =========================
# [3] إعدادات وموديلات تحليل المحادثات
# =========================

# مسار حفظ نتائج تحليل المحادثات في Realtime DB
ANALYSIS_ROOT = "analysis_results/conversation_gen"

# مفاتيح الموديلات
CONV_LOGREG_KEY = "tfidf_logreg"
CONV_RNN_KEY = "rnn"
CONV_RNN_AI_CLASS_IS_ONE = False 
CONV_RNN_MAX_LEN = 300

CONV_MODEL_OPTIONS = {
    "logreg": {
        "result_key": CONV_LOGREG_KEY,
        "task_key": "logreg",
        "name": "Logistic Regression",
    },
    "rnn": {
        "result_key": CONV_RNN_KEY,
        "task_key": "rnn",
        "name": "RNN",
    },
}


def _normalize_conversation_model_key(value, for_results=False):
    raw = _safe_str(value).strip().lower()
    if raw in ("rnn", "baseline_rnn", "conv_rnn"):
        key = "rnn"
    elif raw in ("", "logreg", "logistic", "logistic_regression", "tfidf_logreg", "tf-idf", "tfidf"):
        key = "logreg"
    else:
        key = "logreg"
    model = CONV_MODEL_OPTIONS[key]
    return model["result_key"] if for_results else model["task_key"]


def _conversation_model_name(value):
    key = _normalize_conversation_model_key(value)
    return CONV_MODEL_OPTIONS[key]["name"]

# مسارات موديلات المحادثات
CONV_LOGREG_MODEL_PATH = "models/conversation_logistic_regression.joblib"
CONV_RNN_MODEL_PATH = "Model-Gen-Con/rnn_v2_model.keras"
CONV_RNN_TOKENIZER_PATH = "Model-Gen-Con/rnn_v2_tokenizer.pkl"

# إعدادات اختيار عينات Active Learning للمراجعة
ACTIVE_LEARNING_PERCENT = 0.20
ACTIVE_LEARNING_MAX_SAMPLES = 50

import __main__
__main__.ItemSelector = ItemSelector

# تحميل موديلات المحادثات
conv_logreg_model = joblib.load(CONV_LOGREG_MODEL_PATH)
conv_rnn_model = tf.keras.models.load_model(CONV_RNN_MODEL_PATH)
conv_rnn_tokenizer = joblib.load(CONV_RNN_TOKENIZER_PATH)

#------------- 
news_pipeline = joblib.load('news_baseline_pipeline.pkl')
# --- RNN (Baseline 2) ---
RNN_MODEL_PATH = "news_rnn_baseline.keras"
RNN_TOKENIZER_PATH = "news_rnn_tokenizer.pkl"
RNN_MAX_LEN = 600  

rnn_model = tf.keras.models.load_model(RNN_MODEL_PATH)
rnn_tokenizer = joblib.load(RNN_TOKENIZER_PATH)


def split_into_3_chunks(text):
    words = text.split()
    if len(words) <= 3:
        return [text]
    chunk_size = len(words) // 3
    chunks = [
        " ".join(words[:chunk_size]),
        " ".join(words[chunk_size:2*chunk_size]),
        " ".join(words[2*chunk_size:])
    ]
    return chunks

def rnn_predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]

    seq = rnn_tokenizer.texts_to_sequences(texts)
    x = pad_sequences(seq, maxlen=RNN_MAX_LEN, padding="post", truncating="post")

    p_ai = rnn_model.predict(x, verbose=0).ravel()  # numpy array
    p_h = 1.0 - p_ai

    # ✅ رجّعي Python float (مو numpy.float32)
    out = []
    for h, a in zip(p_h, p_ai):
        out.append([float(h), float(a)])
    return out


def _news_chunk_detail(index, chunk_text, human_prob, ai_prob, article_id=None, title="", article_prediction=None, article_prediction_int=None):
    confidence, uncertainty = _confidence_uncertainty_from_prob(ai_prob)
    prediction = "AI" if ai_prob >= 0.5 else "Human"
    chunk_index = int(index) + 1
    return {
        "chunk_id": f"chunk:{chunk_index}",
        "chunk_index": chunk_index,
        "article_id": article_id,
        "parent_article_id": article_id,
        "title": title,
        "chunk_text": chunk_text,
        "text": chunk_text,
        "label": f"F{chunk_index}",
        "human": round(float(human_prob) * 100, 2),
        "ai": round(float(ai_prob) * 100, 2),
        "human_probability": float(human_prob),
        "ai_probability": float(ai_prob),
        "prediction": prediction,
        "prediction_int": 1 if prediction == "AI" else 0,
        "confidence": _percent_or_none(confidence),
        "uncertainty": _percent_or_none(uncertainty),
        "parent_article_prediction": article_prediction,
        "parent_article_prediction_int": article_prediction_int
    }


def _news_chunks_from_scores(chunks, human_scores, ai_scores, article_id=None, title="", article_prediction=None, article_prediction_int=None):
    details = []
    for index, chunk_text in enumerate(chunks or []):
        if index >= len(human_scores or []) or index >= len(ai_scores or []):
            continue
        details.append(_news_chunk_detail(
            index,
            chunk_text,
            float(human_scores[index]),
            float(ai_scores[index]),
            article_id=article_id,
            title=title,
            article_prediction=article_prediction,
            article_prediction_int=article_prediction_int
        ))
    return details



#------------- 
def _confidence_uncertainty_from_prob(p_positive):
    """
    يحسب الثقة وعدم اليقين من احتمال الكلاس الإيجابي.
    كلما كان الاحتمال قريب من 0.5 يكون المثال أولى بالمراجعة.
    """
    try:
        p = float(p_positive)
    except Exception:
        return None, None

    p = float(np.clip(p, 0.0, 1.0))
    confidence = max(p, 1.0 - p)
    uncertainty = abs(p - 0.5)
    return confidence, uncertainty


def _percent_or_none(value):
    """
    يحول القيمة العشرية إلى نسبة مئوية عند توفرها.
    """
    if value is None:
        return None
    try:
        return round(float(value) * 100.0, 2)
    except Exception:
        return None


def _machine_probability_from_proba(model, proba_row):
    """
    يستخرج احتمال الكلاس Machine/AI من predict_proba.
    يعتمد على class 1 عند توفر classes_، وإلا يستخدم العمود الثاني مثل كود التدريب.
    """
    try:
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            return float(proba_row[classes.index(1)])
        if "1" in classes:
            return float(proba_row[classes.index("1")])
        for label in ("AI", "Machine", "Machine-generated", "machine", "machine-generated"):
            if label in classes:
                return float(proba_row[classes.index(label)])
    except Exception:
        pass

    try:
        return float(proba_row[1])
    except Exception:
        return None


def _news_probability_pair(proba_row, model=None, warn_on_fallback=True):
    """
    Returns (p_human, p_ai) for News predict_proba output using classes_ when available.
    Falls back to the existing [Human, AI] column order if model metadata is missing.
    """
    probs = np.asarray(proba_row, dtype=float).reshape(-1).tolist()
    classes_raw = getattr(model, "classes_", None) if model is not None else None
    classes = list(classes_raw) if classes_raw is not None else []

    def _class_key(value):
        return str(value).strip().lower().replace("-", "").replace("_", "").replace(" ", "")

    ai_idx = None
    human_idx = None
    for idx, cls in enumerate(classes):
        if idx >= len(probs):
            continue
        key = _class_key(cls)
        if cls == 1 or key in {"1", "ai", "machine", "machinegen", "machinegenerated"}:
            ai_idx = idx
        elif cls == 0 or key in {"0", "human", "real", "genuine", "authentic", "notai"}:
            human_idx = idx

    if ai_idx is not None:
        p_ai = float(probs[ai_idx])
        if human_idx is not None:
            p_human = float(probs[human_idx])
        elif len(probs) == 2:
            p_human = float(probs[1 - ai_idx])
        else:
            p_human = float(1.0 - p_ai)
        return p_human, p_ai

    if warn_on_fallback and not getattr(_news_probability_pair, "_warned_fallback", False):
        app.logger.warning("News model classes_ missing or unexpected (%s); using [Human, AI] probability order.", classes)
        _news_probability_pair._warned_fallback = True

    try:
        return float(probs[0]), float(probs[1])
    except Exception:
        return 0.0, 0.0


GROUND_TRUTH_COLUMN_NAMES = (
    "MachineGen", "machinegen", "label", "Label", "target", "Target",
    "ground_truth", "Ground_Truth", "groundTruth", "is_ai", "isAI",
    "class", "Class", "y", "Y"
)


def _canonical_column_name(name):
    # نوحد اسم العمود حتى لا تضيع الأعمدة بسبب BOM أو اختلاف بسيط في الكتابة.
    return _safe_str(name).replace("\ufeff", "").strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_uploaded_csv_row(row):
    # نحفظ الصف كما هو لكن ننظف أسماء الأعمدة المكررة/المخفية من ملف CSV.
    if not isinstance(row, dict):
        return {}
    cleaned = {}
    for key, value in row.items():
        raw_key = _safe_str(key).replace("\ufeff", "").strip()
        if raw_key:
            cleaned[raw_key] = value
    return cleaned


def _payload_value(payload, *names):
    # قراءة مرنة لأعمدة CSV بغض النظر عن حالة الأحرف أو وجود BOM.
    if not isinstance(payload, dict):
        return None
    wanted = {_canonical_column_name(name) for name in names if _safe_str(name)}
    for key, value in payload.items():
        if _canonical_column_name(key) in wanted:
            return value
    return None


def _normalize_binary_label(value):
    # يوحد كل صيغ التصنيف إلى 0 للبشري و 1 للنص المولد آليا.
    if value is None:
        return None

    if isinstance(value, bool):
        return 1 if value else 0

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if float(value) == 0.0:
                return 0
            if float(value) == 1.0:
                return 1
        except Exception:
            return None

    text = str(value).strip().lower()
    if not text:
        return None

    text = text.replace("_", " ").replace("-", " ")
    text = " ".join(text.split())

    human_labels = {"0", "human", "h", "real", "genuine", "authentic", "not ai"}
    ai_labels = {
        "1", "ai", "machine", "machine generated", "ai generated",
        "ai written", "machine written", "synthetic", "bot", "llm", "generated"
    }

    if text in human_labels:
        return 0
    if text in ai_labels:
        return 1

    return None


def _extract_ground_truth_from_payload(payload):
    if not isinstance(payload, dict):
        return None
    for column in GROUND_TRUTH_COLUMN_NAMES:
        value = _payload_value(payload, column)
        if value is not None:
            normalized = _normalize_binary_label(value)
            if normalized is not None:
                return normalized
    return None


def _label_text_from_int(value):
    normalized = _normalize_binary_label(value)
    if normalized == 0:
        return "Human"
    if normalized == 1:
        return "AI"
    return ""


def _compute_standard_metrics(y_true, y_pred):
    # يحسب المقاييس فقط للصفوف التي لديها ground truth صالح.
    valid_true = []
    valid_pred = []

    for true_value, pred_value in zip(y_true or [], y_pred or []):
        true_norm = _normalize_binary_label(true_value)
        pred_norm = _normalize_binary_label(pred_value)
        if true_norm is None or pred_norm is None:
            continue
        valid_true.append(true_norm)
        valid_pred.append(pred_norm)

    if not valid_true:
        return {
            "available": False,
            "reason": "Ground truth labels are missing or invalid."
        }

    try:
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

        cm = confusion_matrix(valid_true, valid_pred, labels=[0, 1]).tolist()
        tn, fp = int(cm[0][0]), int(cm[0][1])
        fn, tp = int(cm[1][0]), int(cm[1][1])

        human_support = sum(1 for value in valid_true if value == 0)
        ai_support = sum(1 for value in valid_true if value == 1)

        return {
            "available": True,
            "accuracy": float(accuracy_score(valid_true, valid_pred)),
            "precision_macro": float(precision_score(valid_true, valid_pred, labels=[0, 1], average="macro", zero_division=0)),
            "recall_macro": float(recall_score(valid_true, valid_pred, labels=[0, 1], average="macro", zero_division=0)),
            "f1_macro": float(f1_score(valid_true, valid_pred, labels=[0, 1], average="macro", zero_division=0)),
            "precision_ai": float(precision_score(valid_true, valid_pred, labels=[1], average="macro", zero_division=0)),
            "recall_ai": float(recall_score(valid_true, valid_pred, labels=[1], average="macro", zero_division=0)),
            "f1_ai": float(f1_score(valid_true, valid_pred, labels=[1], average="macro", zero_division=0)),
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
            "support": {
                "total_labeled": len(valid_true),
                "human": human_support,
                "ai": ai_support
            }
        }
    except Exception as e:
        app.logger.warning("Standard metrics failed: %s", e)
        return {
            "available": False,
            "reason": "Could not compute metrics on the server."
        }


def _active_learning_limit(total_count):
    """
    يختار 20% من العينات بحد أقصى 50 عينة.
    """
    try:
        total = int(total_count or 0)
    except Exception:
        total = 0

    if total <= 0:
        return 0

    return max(1, min(ACTIVE_LEARNING_MAX_SAMPLES, int(math.ceil(total * ACTIVE_LEARNING_PERCENT))))


def _active_learning_sort_value(row):
    """
    قيمة الترتيب لاختيار الأقل ثقة.
    الأقل يعني أقرب لاحتمال 0.5 وأهم للمراجعة.
    """
    if not isinstance(row, dict):
        return float("inf")

    raw_uncertainty = row.get("uncertainty")
    if raw_uncertainty is not None:
        try:
            value = float(raw_uncertainty)
            return value * 100.0 if value <= 1 else value
        except Exception:
            pass

    raw_confidence = row.get("confidence")
    if raw_confidence is not None:
        try:
            value = float(raw_confidence)
            value = value * 100.0 if value <= 1 else value
            return max(0.0, value - 50.0)
        except Exception:
            pass

    raw_p_machine = row.get("p_machine")
    if raw_p_machine is not None:
        try:
            return abs(float(raw_p_machine) - 0.5) * 100.0
        except Exception:
            pass

    raw_ai = row.get("ai_percentage")
    if raw_ai is not None:
        try:
            value = float(raw_ai)
            value = value / 100.0 if value > 1 else value
            return abs(value - 0.5) * 100.0
        except Exception:
            pass

    return float("inf")


def _is_logistic_model_key(model_key):
    """
    Active Learning يطبق على Logistic Regression فقط.
    """
    normalized = _safe_str(model_key).lower()
    return normalized in ("logistic", "logreg", CONV_LOGREG_KEY, "logistic regression") or "logistic" in normalized


def _uses_frozen_active_learning(model_key):
    """
    Frozen review targets are created for supported detector models.
    """
    normalized = _safe_str(model_key).strip().lower()
    return bool(normalized) and normalized not in ("sota_model", "sota", "coming_soon")


def _active_learning_retraining_supported(model_key):
    """
    Active Learning retraining is currently supported only for Logistic Regression.
    """
    normalized = _safe_str(model_key).strip().lower()
    return normalized in ("logistic", "logreg", CONV_LOGREG_KEY)


def _active_learning_model_version(project_id, selected_model_key):
    """
    Returns the current active-learning version for this project.
    """
    registry = rtdb.reference(_project_versions_path(project_id)).get() or {}
    if isinstance(registry, dict) and registry.get("current_version"):
        return _safe_str(registry.get("current_version")) or "v1"
    return "v1"


def _active_learning_selection_run_id():
    """
    Creates a unique id for one frozen target selection run.
    """
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"sel_{stamp}_{uuid.uuid4().hex[:8]}"


def _safe_target_key(sample_id):
    """
    Converts sample ids into Firebase-safe keys while keeping the original id in the record.
    """
    raw = _safe_str(sample_id) or "unknown_sample"
    cleaned = re.sub(r'[.#$\[\]/]', "_", raw) or "unknown_sample"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned}__{digest}"


def _active_learning_targets_path(project_id, model_version, selection_run_id):
    return f"active_learning_targets/{_safe_str(project_id)}/{_safe_str(model_version)}/{_safe_str(selection_run_id)}"


def _active_learning_current_path(project_id, model_version):
    return f"active_learning_current/{_safe_str(project_id)}/{_safe_str(model_version)}"


def _review_targets_path(project_id, model_version, selection_run_id):
    return f"review_targets/{_safe_str(project_id)}/{_safe_str(model_version)}/{_safe_str(selection_run_id)}"


def _review_targets_current_path(project_id, model_version):
    return f"review_targets_current/{_safe_str(project_id)}/{_safe_str(model_version)}"


def _get_current_active_learning_selection(project_id, model_version):
    pointer = rtdb.reference(_review_targets_current_path(project_id, model_version)).get()
    if not isinstance(pointer, dict):
        pointer = rtdb.reference(_active_learning_current_path(project_id, model_version)).get()
    return pointer if isinstance(pointer, dict) else None


def _load_frozen_active_learning_targets(project_id, model_version):
    pointer = _get_current_active_learning_selection(project_id, model_version)
    if not pointer:
        return None

    selection_run_id = _safe_str(pointer.get("selection_run_id"))
    if not selection_run_id:
        return None

    target_set = rtdb.reference(_review_targets_path(project_id, model_version, selection_run_id)).get()
    storage_family = "review_targets"
    if not isinstance(target_set, dict) or not isinstance(target_set.get("targets"), dict):
        target_set = rtdb.reference(_active_learning_targets_path(project_id, model_version, selection_run_id)).get()
        storage_family = "active_learning_targets"
    if not isinstance(target_set, dict) or not isinstance(target_set.get("targets"), dict):
        return None

    target_set.setdefault("selection_run_id", selection_run_id)
    target_set.setdefault("selected_model_key", pointer.get("selected_model_key"))
    target_set.setdefault("project_type", pointer.get("project_type"))
    target_set.setdefault("target_purpose", "review")
    target_set.setdefault("active_learning_enabled", _active_learning_retraining_supported(target_set.get("selected_model_key")))
    target_set.setdefault("retraining_supported", _active_learning_retraining_supported(target_set.get("selected_model_key")))
    target_set["_storage_family"] = storage_family
    return target_set


def _freeze_active_learning_targets_if_missing(project_id, project_type, selected_model_key, model_version, candidates, assigned_examiner_ids=None):
    """
    Freezes the shared review target set once, then returns the stored selection.
    """
    if not _uses_frozen_active_learning(selected_model_key):
        return None

    existing = _load_frozen_active_learning_targets(project_id, model_version)
    if _version_is_closed(project_id, model_version):
        return existing
    existing_targets = _frozen_target_values(existing)
    existing_is_news_chunk_level = (
        project_type == "news"
        and existing_targets
        and all(_safe_str(target.get("target_unit")) == "chunk" for target in existing_targets if isinstance(target, dict))
    )
    incoming_has_news_chunks = (
        project_type == "news"
        and any(isinstance(item, dict) and item.get("chunk_index") is not None for item in (candidates or []))
    )
    if existing and (project_type != "news" or existing_is_news_chunk_level or not incoming_has_news_chunks):
        return existing

    safe_candidates = [item for item in (candidates or []) if isinstance(item, dict) and _safe_str(item.get("sample_id"))]
    safe_candidates.sort(key=lambda item: (_active_learning_sort_value(item), _safe_str(item.get("sample_id"))))

    limit = _active_learning_limit(len(safe_candidates))
    selected = safe_candidates[:limit]
    now = _now_utc_iso()
    selection_run_id = _active_learning_selection_run_id()
    assigned_ids = assigned_examiner_ids if isinstance(assigned_examiner_ids, list) else []

    active_learning_enabled = True
    retraining_supported = _active_learning_retraining_supported(selected_model_key)
    targets = {}
    for rank, candidate in enumerate(selected, start=1):
        sample_id = _safe_str(candidate.get("sample_id"))
        target_key = _safe_target_key(sample_id)
        target = {
            "active_learning_selected": True,
            "target_purpose": "review",
            "active_learning_enabled": active_learning_enabled,
            "retraining_supported": retraining_supported,
            "target_status": "pending",
            "selection_run_id": selection_run_id,
            "selection_rank": rank,
            "selection_score": _owner_decimal(candidate.get("uncertainty")) or 0.0,
            "selection_strategy": "least_confidence_margin",
            "selected_at": now,
            "model_version": model_version,
            "version_id": model_version,
            "project_id": project_id,
            "selected_model_key": selected_model_key,
            "sample_id": sample_id,
            "source_id": _safe_str(candidate.get("source_id") or sample_id),
            "project_type": project_type,
            "target_unit": candidate.get("target_unit") or ("chunk" if project_type == "news" and candidate.get("chunk_index") is not None else "sample"),
            "result_path": _safe_str(candidate.get("result_path")),
            "prediction": candidate.get("prediction"),
            "prediction_int": candidate.get("prediction_int"),
            "ground_truth": candidate.get("ground_truth"),
            "confidence": _owner_decimal(candidate.get("confidence")) or 0.0,
            "uncertainty": _owner_decimal(candidate.get("uncertainty")) or 0.0,
            "reviewed_by": None,
            "reviewed_at": None,
            "feedback_count": 0,
            "assigned_examiner_ids": assigned_ids
        }

        for extra_key in (
            "article_id", "title", "text", "chunk_id", "chunk_index", "chunk_text",
            "parent_article_id", "parent_article_prediction", "parent_article_prediction_int",
            "row_id", "dialogue_id", "conversation_id", "task_id", "turn_index", "safe_key"
        ):
            if candidate.get(extra_key) is not None:
                target[extra_key] = candidate.get(extra_key)

        targets[target_key] = target

    target_set = {
        "project_id": project_id,
        "project_type": project_type,
        "target_purpose": "review",
        "active_learning_enabled": active_learning_enabled,
        "retraining_supported": retraining_supported,
        "model_version": model_version,
        "version_id": model_version,
        "selected_model_key": selected_model_key,
        "selection_run_id": selection_run_id,
        "selection_strategy": "least_confidence_margin",
        "selection_formula": "uncertainty = abs(p_ai - 0.5)",
        "selection_percent": ACTIVE_LEARNING_PERCENT,
        "max_samples": ACTIVE_LEARNING_MAX_SAMPLES,
        "source_total": len(safe_candidates),
        "selected_total": len(targets),
        "selected_at": now,
        "status": "frozen",
        "assigned_examiner_ids": assigned_ids,
        "targets": targets
    }

    pointer = {
        "selection_run_id": selection_run_id,
        "selected_model_key": selected_model_key,
        "project_type": project_type,
        "created_at": now,
        "status": "frozen"
    }

    rtdb.reference(_review_targets_path(project_id, model_version, selection_run_id)).set(_owner_json_value(target_set))
    rtdb.reference(_review_targets_current_path(project_id, model_version)).set(_owner_json_value(pointer))
    rtdb.reference(_active_learning_targets_path(project_id, model_version, selection_run_id)).set(_owner_json_value(target_set))
    rtdb.reference(_active_learning_current_path(project_id, model_version)).set(_owner_json_value(pointer))
    return target_set


def _frozen_target_values(selection):
    targets = (selection or {}).get("targets") or {}
    return list(targets.values()) if isinstance(targets, dict) else []


def _frozen_target_sample_ids(selection):
    return {_safe_str(item.get("sample_id")) for item in _frozen_target_values(selection) if _safe_str(item.get("sample_id"))}


def _frozen_active_learning_info(selection, enabled=True, unit="items"):
    targets = _frozen_target_values(selection)
    reviewed = sum(1 for item in targets if _target_feedback_finalized(item))
    selected_model_key = (selection or {}).get("selected_model_key")
    active_learning_enabled = bool(enabled)
    retraining_supported = bool((selection or {}).get("retraining_supported", _active_learning_retraining_supported(selected_model_key)))
    return {
        "enabled": active_learning_enabled,
        "review_targets_enabled": bool(enabled),
        "active_learning_enabled": active_learning_enabled,
        "retraining_supported": retraining_supported,
        "percent": ACTIVE_LEARNING_PERCENT,
        "max_samples": ACTIVE_LEARNING_MAX_SAMPLES,
        "source_total": int((selection or {}).get("source_total") or 0),
        "selected": len(targets),
        "reviewed": reviewed,
        "total": len(targets),
        "unit": unit
    }


def _feedback_lifecycle_status(record):
    status = _safe_str((record or {}).get("status") or (record or {}).get("lifecycle_status")).strip().lower()
    if status in ("locked", "submitted"):
        return status
    if status in ("draft_saved", "draft", "accepted", "reviewed"):
        return "draft_saved"
    return "pending"


def _feedback_is_final(record):
    return _feedback_lifecycle_status(record) in ("submitted", "locked") or bool((record or {}).get("locked"))


def _feedback_label_matches_prediction(record, prediction):
    return _label_text_only((record or {}).get("corrected_label") or (record or {}).get("corrected_label_text") or (record or {}).get("label")) == _label_text_only(prediction)


def _target_feedback_finalized(target):
    status = _safe_str((target or {}).get("target_status")).strip().lower()
    return status in ("submitted", "locked") or bool((target or {}).get("locked"))


def _target_feedback_started(target):
    status = _safe_str((target or {}).get("target_status")).strip().lower()
    return status in ("draft_saved", "submitted", "locked", "reviewed") or int((target or {}).get("feedback_count") or 0) > 0


def _mark_frozen_active_learning_target_reviewed(project_id, model_version, sample_id, examiner_uid, submitted_at, match=None):
    """
    Marks the frozen target as draft-saved after the existing feedback write succeeds.
    """
    if _version_is_closed(project_id, model_version):
        return False
    selection = _load_frozen_active_learning_targets(project_id, model_version)
    if not selection:
        return False

    targets = selection.get("targets") or {}
    if not isinstance(targets, dict):
        return False

    target_key = _safe_target_key(sample_id)
    if target_key not in targets and isinstance(match, dict):
        for key, target in targets.items():
            if not isinstance(target, dict):
                continue
            matched = True
            for match_key, match_value in match.items():
                if _safe_str(target.get(match_key)) != _safe_str(match_value):
                    matched = False
                    break
            if matched:
                target_key = key
                break

    if target_key not in targets:
        return False

    update_payload = {
        "target_status": "draft_saved",
        "draft_by": examiner_uid,
        "draft_saved_at": submitted_at,
        "reviewed_by": examiner_uid,
        "reviewed_at": submitted_at,
        "feedback_count": 1
    }
    selection_run_id = selection.get("selection_run_id")
    rtdb.reference(_review_targets_path(project_id, model_version, selection_run_id)).child("targets").child(target_key).update(update_payload)
    rtdb.reference(_active_learning_targets_path(project_id, model_version, selection_run_id)).child("targets").child(target_key).update(update_payload)
    try:
        _update_project_version_status(project_id, model_version, "feedback_in_progress")
        _set_labeling_tasks_status_for_version(project_id, model_version, "progress")
    except Exception:
        pass
    return True


def _labeling_task_completed(task_data):
    return _safe_str((task_data or {}).get("status")).strip().lower() == "completed"


def _feedback_owner_uid(feedback_node):
    if not isinstance(feedback_node, dict) or not feedback_node:
        return ""

    direct_uid = _safe_str(
        feedback_node.get("examiner_uid")
        or feedback_node.get("reviewed_by")
        or feedback_node.get("updated_by")
    )
    if direct_uid:
        return direct_uid

    for key, value in feedback_node.items():
        if isinstance(value, dict):
            return _safe_str(value.get("examiner_uid") or value.get("reviewed_by") or key)
    return ""


def _feedback_record_for_uid(feedback_node, uid):
    if not isinstance(feedback_node, dict) or not uid:
        return None
    if isinstance(feedback_node.get(uid), dict):
        return feedback_node.get(uid)
    owner_uid = _feedback_owner_uid(feedback_node)
    if owner_uid == uid and _safe_str(feedback_node.get("examiner_uid") or feedback_node.get("reviewed_by")):
        return feedback_node
    return None


def _feedback_history_entry(existing_feedback, edited_at):
    existing = existing_feedback if isinstance(existing_feedback, dict) else {}
    return {
        "edited_at": edited_at,
        "previous_label": existing.get("label"),
        "previous_explanation": existing.get("explanation") or existing.get("feedback_explanation"),
        "previous_agreed_with_model": bool(existing.get("agreed_with_model", False))
    }


def _prepare_feedback_payload(existing_feedback, base_payload, uid, examiner_name, now_iso):
    existing = existing_feedback if isinstance(existing_feedback, dict) else {}
    is_edit = bool(existing)
    payload = dict(base_payload)

    payload["examiner_uid"] = existing.get("examiner_uid") or uid
    payload["examiner_name"] = existing.get("examiner_name") or examiner_name
    payload["reviewed_by"] = existing.get("reviewed_by") or existing.get("examiner_uid") or uid
    payload["reviewed_by_name"] = existing.get("reviewed_by_name") or existing.get("examiner_name") or examiner_name
    payload["reviewed_at"] = existing.get("reviewed_at") or existing.get("submitted_at") or now_iso
    payload["submitted_at"] = existing.get("submitted_at") or now_iso

    history = existing.get("previous_feedback_history") if isinstance(existing.get("previous_feedback_history"), list) else []
    if is_edit:
        history = history + [_feedback_history_entry(existing, now_iso)]
        payload["updated_at"] = now_iso
        payload["updated_by"] = uid
        payload["edit_count"] = int(existing.get("edit_count") or 0) + 1
    else:
        payload["updated_at"] = None
        payload["updated_by"] = None
        payload["edit_count"] = int(existing.get("edit_count") or 0)

    payload["previous_feedback_history"] = history
    return payload, is_edit


def _looks_garbled_text(value):
    text = _safe_str(value).strip()
    if not text:
        return True
    markers = ("�", "â", "ï", "ð", "ًں", "گ", "œ")
    return any(marker in text for marker in markers)


def _short_uid(uid):
    text = _safe_str(uid)
    return f"{text[:6]}..." if len(text) > 6 else (text or "Examiner")


def _reviewer_display_name(uid, stored_name=None):
    user_id = _safe_str(uid)
    if user_id:
        try:
            user_doc = db.collection("users").document(user_id).get()
            if user_doc.exists:
                user_data = user_doc.to_dict() or {}
                profile = user_data.get("profile", {}) or {}
                profile_name = f"{profile.get('firstName','')} {profile.get('lastName','')}".strip()
                if profile_name and not _looks_garbled_text(profile_name):
                    return profile_name
                email = _safe_str(user_data.get("email"))
                if email:
                    return email
        except Exception:
            pass

    stored = _safe_str(stored_name).strip()
    if stored and not _looks_garbled_text(stored):
        return stored
    return _short_uid(user_id)


def _normalized_visible_label(value):
    return _label_text_from_int(value) or ("AI" if _is_ai_label(value) else ("Human" if _safe_str(value).strip().lower() == "human" else ""))


def _probability_percent(value):
    number = _owner_decimal(value)
    if number is None:
        return None
    return number * 100.0 if number <= 1 else number


def _uncertainty_percent_from_ai(ai_value):
    ai_percent = _probability_percent(ai_value)
    if ai_percent is None:
        return None
    return abs((ai_percent / 100.0) - 0.5) * 100.0


def _chunk_uncertainty_percent(chunk):
    if not isinstance(chunk, dict):
        return None
    direct = _owner_decimal(chunk.get("uncertainty"))
    if direct is not None:
        return direct * 100.0 if direct <= 1 else direct
    return _uncertainty_percent_from_ai(_first_present(chunk.get("ai"), chunk.get("ai_percentage"), chunk.get("p_machine")))


def _feedback_write_message(is_edit):
    return "Feedback draft updated successfully" if is_edit else "Feedback draft saved successfully"


def _update_news_labeling_status_from_frozen_targets(project_id, model_version):
    """
    Updates news labeling tasks from the shared frozen target progress.
    """
    selection = _load_frozen_active_learning_targets(project_id, model_version)
    targets = _frozen_target_values(selection)
    if not targets:
        return None

    drafts = sum(1 for target in targets if _target_feedback_started(target))
    finalized = sum(1 for target in targets if _target_feedback_finalized(target))
    total = len(targets)
    new_status = "completed" if finalized >= total else ("progress" if drafts > 0 else "pending")

    for doc in db.collection("tasks").where("project_ID", "==", project_id).where("task_type", "==", "labeling").stream():
        task_data = doc.to_dict() or {}
        conversation_type = _safe_str(task_data.get("conversation_type")).strip().lower()
        if conversation_type:
            continue
        if _safe_str(task_data.get("status")).strip().lower() != new_status:
            db.collection("tasks").document(doc.id).update({"status": new_status})

    return new_status


def _model_selection_task_guard(task_id, project_id=None, reject_completed=False):
    """
    Validates a model-selection task before detection or final selection.
    """
    if not session.get("idToken"):
        return None, None, (jsonify({"error": "Unauthorized"}), 401)

    uid = session.get("uid")
    task_id = _safe_str(task_id)
    if not task_id:
        return None, None, (jsonify({"error": "task_id is required"}), 400)

    task_ref = db.collection("tasks").document(task_id)
    task_doc = task_ref.get()
    if not task_doc.exists:
        return None, None, (jsonify({"error": "Task not found"}), 404)

    task_data = task_doc.to_dict() or {}
    if project_id and task_data.get("project_ID") != project_id:
        return None, None, (jsonify({"error": "Task does not belong to this project"}), 400)

    if task_data.get("task_type") != "model_selection":
        return None, None, (jsonify({"error": "Task is not model_selection"}), 400)

    examiner_ids = task_data.get("examiner_ids") or []
    if len(examiner_ids) != 1:
        return None, None, (jsonify({"error": "Model Selection task requires exactly 1 examiner"}), 400)

    if uid not in examiner_ids:
        return None, None, (jsonify({"error": "Forbidden"}), 403)

    if reject_completed and (task_data.get("selected_model") or _safe_str(task_data.get("status")).lower() == "completed"):
        return None, None, (jsonify({"error": "Model already selected for this task"}), 409)

    return task_ref, task_data, None


def _apply_active_learning_turn_selection(items, enabled=True):
    """
    يختار أقل turns ثقة مع إبقاء بقية Turns داخل نفس المحادثة كسياق.
    """
    if not enabled:
        return items, {
            "enabled": False,
            "percent": ACTIVE_LEARNING_PERCENT,
            "max_samples": ACTIVE_LEARNING_MAX_SAMPLES,
            "source_total": sum(len(item.get("turns") or []) for item in items),
            "selected": sum(len(item.get("turns") or []) for item in items),
            "reviewed": sum(1 for item in items if item.get("conversation_locked")),
            "total": len(items),
            "unit": "conversations"
        }

    candidates = []
    for item in items:
        conversation_id = item.get("conversation_id")
        for turn in item.get("turns") or []:
            candidates.append({
                "conversation_id": conversation_id,
                "turn_index": int(turn.get("turn_index", 0) or 0),
                "score": _active_learning_sort_value(turn)
            })

    limit = _active_learning_limit(len(candidates))
    selected = sorted(
        candidates,
        key=lambda row: (row["score"], str(row["conversation_id"]), row["turn_index"])
    )[:limit]
    selected_keys = {
        (str(row["conversation_id"]), int(row["turn_index"]))
        for row in selected
    }

    focused_items = []
    reviewed_turns = 0
    for item in items:
        conversation_id = str(item.get("conversation_id"))
        selected_count = 0
        reviewed_count = 0

        for turn in item.get("turns") or []:
            key = (conversation_id, int(turn.get("turn_index", 0) or 0))
            is_selected = key in selected_keys
            turn["active_learning_selected"] = is_selected
            if is_selected:
                selected_count += 1
                if turn.get("turn_locked"):
                    reviewed_count += 1

        if selected_count > 0:
            item["active_learning_selected_turns"] = selected_count
            item["active_learning_reviewed_turns"] = reviewed_count
            item["conversation_locked"] = reviewed_count >= selected_count
            item["has_feedback"] = reviewed_count > 0
            focused_items.append(item)
            reviewed_turns += reviewed_count

    focused_items.sort(key=lambda item: min(
        [_active_learning_sort_value(turn) for turn in (item.get("turns") or []) if turn.get("active_learning_selected")] or [float("inf")]
    ))

    return focused_items, {
        "enabled": True,
        "percent": ACTIVE_LEARNING_PERCENT,
        "max_samples": ACTIVE_LEARNING_MAX_SAMPLES,
        "source_total": len(candidates),
        "selected": len(selected_keys),
        "reviewed": reviewed_turns,
        "total": len(selected_keys),
        "unit": "turns"
    }


def _news_active_learning_candidates(project_id, selected_model_key, details):
    candidates = []
    for index, item in enumerate(details or []):
        if not isinstance(item, dict):
            continue
        article_id = _safe_str(item.get("article_id") or item.get("id"))
        if not article_id:
            continue
        chunks = item.get("chunks") if isinstance(item.get("chunks"), list) else []
        if chunks:
            for chunk_pos, chunk in enumerate(chunks):
                if not isinstance(chunk, dict):
                    continue
                chunk_index = int(chunk.get("chunk_index") or chunk_pos + 1)
                sample_id = _make_active_learning_sample_id("news", article_id=article_id, chunk_index=chunk_index)
                candidates.append({
                    "sample_id": sample_id,
                    "source_id": f"{article_id}:chunk:{chunk_index}",
                    "article_id": article_id,
                    "parent_article_id": article_id,
                    "chunk_id": chunk.get("chunk_id") or f"chunk:{chunk_index}",
                    "chunk_index": chunk_index,
                    "target_unit": "chunk",
                    "title": item.get("title", ""),
                    "text": chunk.get("chunk_text") or chunk.get("text") or item.get("content", ""),
                    "chunk_text": chunk.get("chunk_text") or chunk.get("text") or "",
                    "prediction": chunk.get("prediction"),
                    "prediction_int": _first_present(chunk.get("prediction_int"), _label_to_machinegen(chunk.get("prediction"))),
                    "ground_truth": item.get("ground_truth"),
                    "confidence": chunk.get("confidence"),
                    "uncertainty": chunk.get("uncertainty"),
                    "project_type": "news",
                    "parent_article_prediction": item.get("prediction"),
                    "parent_article_prediction_int": _first_present(item.get("prediction_int"), _label_to_machinegen(item.get("prediction"))),
                    "result_path": f"analysis_results/{project_id}/{selected_model_key}/details/{index}/chunks/{chunk_pos}"
                })
            continue

        candidates.append({
            "sample_id": _make_active_learning_sample_id("news", article_id=article_id),
            "source_id": article_id,
            "article_id": article_id,
            "target_unit": "article",
            "title": item.get("title", ""),
            "text": item.get("content", ""),
            "prediction": item.get("prediction"),
            "prediction_int": _first_present(item.get("prediction_int"), _label_to_machinegen(item.get("prediction"))),
            "ground_truth": item.get("ground_truth"),
            "confidence": item.get("confidence"),
            "uncertainty": item.get("uncertainty"),
            "project_type": "news",
            "result_path": f"analysis_results/{project_id}/{selected_model_key}/details/{index}"
        })
    return candidates


def _uploaded_conversation_active_learning_candidates(project_id, selected_model_key, run_id, run_ref):
    dialogue_turns = run_ref.child("dialogue_turns").get() or {}
    key_map = run_ref.child("dialogue_key_map").get() or {}
    reverse_key_map = {v: k for k, v in key_map.items()} if isinstance(key_map, dict) else {}
    candidates = []

    for safe_key, turns_raw in (dialogue_turns.items() if isinstance(dialogue_turns, dict) else []):
        turns = list(turns_raw.values()) if isinstance(turns_raw, dict) else (turns_raw if isinstance(turns_raw, list) else [])
        dialogue_id = reverse_key_map.get(safe_key, safe_key)
        for idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            turn_index = idx + 1
            row_id = _safe_str(turn.get("row_id") or turn.get("source_row_id"))
            sample_id = _make_active_learning_sample_id(
                "uploaded_conversation",
                row_id=row_id,
                dialogue_id=dialogue_id,
                turn_index=turn_index
            )
            candidates.append({
                "sample_id": sample_id,
                "source_id": row_id or f"{dialogue_id}:{turn_index}",
                "row_id": row_id,
                "dialogue_id": dialogue_id,
                "safe_key": safe_key,
                "turn_index": turn_index,
                "text": turn.get("text", ""),
                "prediction": turn.get("prediction"),
                "prediction_int": _first_present(turn.get("prediction_int"), _label_to_machinegen(turn.get("prediction"))),
                "ground_truth": _first_present(turn.get("ground_truth"), turn.get("gt")),
                "confidence": turn.get("confidence"),
                "uncertainty": turn.get("uncertainty"),
                "project_type": "uploaded_conversation",
                "result_path": f"analysis_results/conversations/{selected_model_key}/{project_id}/runs/{run_id}/dialogue_turns/{safe_key}/{idx}"
            })
    return candidates


def _generated_conversation_active_learning_candidates(project_id, selected_model_key, raw):
    candidates = []
    for node_key, node_val in (raw.items() if isinstance(raw, dict) else []):
        if not isinstance(node_val, dict):
            continue
        meta = node_val.get("meta", {}) or {}
        conversation_id = meta.get("task_id") or node_key
        turns_raw = node_val.get("turns", {}) or {}
        turns = list(turns_raw.values()) if isinstance(turns_raw, dict) else (turns_raw if isinstance(turns_raw, list) else [])
        for idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            turn_index = int(turn.get("turn_index", idx + 1) or idx + 1)
            sample_id = _make_active_learning_sample_id(
                "generated_conversation",
                conversation_id=conversation_id,
                turn_index=turn_index
            )
            candidates.append({
                "sample_id": sample_id,
                "source_id": f"{conversation_id}:{turn_index}",
                "task_id": node_key,
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "text": turn.get("text", ""),
                "prediction": turn.get("prediction"),
                "prediction_int": _first_present(turn.get("prediction_int"), _label_to_machinegen(turn.get("prediction"))),
                "ground_truth": _first_present(turn.get("ground_truth"), turn.get("gt")),
                "confidence": turn.get("confidence"),
                "uncertainty": turn.get("uncertainty"),
                "project_type": "generated_conversation",
                "result_path": f"{ANALYSIS_ROOT}/{selected_model_key}/{project_id}/{node_key}/turns/{idx}"
            })
    return candidates


def _generated_conversation_base_ref(project_id, model_key):
    return rtdb.reference(f"{ANALYSIS_ROOT}/{model_key}/{project_id}")


def _generated_conversation_results_payload(project_id, model_key, run_id=None):
    base_ref = _generated_conversation_base_ref(project_id, model_key)
    selected_run_id = _safe_str(run_id) or _safe_str(base_ref.child("latest_run_id").get())
    if selected_run_id:
        run_payload = base_ref.child("runs").child(selected_run_id).get()
        if isinstance(run_payload, dict):
            return run_payload

    legacy_payload = base_ref.get() or {}
    if isinstance(legacy_payload, dict) and "runs" not in legacy_payload:
        return legacy_payload
    return {}


def _generated_conversation_node_ref(project_id, model_key, conversation_id):
    base_ref = _generated_conversation_base_ref(project_id, model_key)
    run_id = _safe_str(base_ref.child("latest_run_id").get())
    if run_id:
        return base_ref.child("runs").child(run_id).child(conversation_id)
    return base_ref.child(conversation_id)


def _apply_frozen_active_learning_turn_selection(items, selection, project_type):
    targets = _frozen_target_values(selection)
    target_pairs = set()
    target_lookup = {}
    for target in targets:
        if project_type == "uploaded_conversation":
            pair = (_safe_str(target.get("dialogue_id")), int(target.get("turn_index") or 0))
        else:
            pair = (_safe_str(target.get("conversation_id")), int(target.get("turn_index") or 0))
        target_pairs.add(pair)
        target_lookup[pair] = target

    focused_items = []
    reviewed_turns = 0
    for item in items:
        conversation_id = _safe_str(item.get("conversation_id"))
        selected_count = 0
        reviewed_count = 0
        for turn in item.get("turns") or []:
            key = (conversation_id, int(turn.get("turn_index", 0) or 0))
            is_selected = key in target_pairs
            target = target_lookup.get(key) or {}
            frozen_started = _target_feedback_started(target)
            frozen_finalized = _target_feedback_finalized(target)
            turn["active_learning_selected"] = is_selected
            if is_selected:
                turn["target_status"] = target.get("target_status") or ("submitted" if frozen_finalized else ("draft_saved" if frozen_started or turn.get("turn_locked") else "pending"))
                turn["reviewed_by"] = target.get("reviewed_by")
                turn["reviewed_at"] = target.get("reviewed_at")
                turn["turn_locked"] = bool(frozen_finalized)
            if is_selected:
                selected_count += 1
                if frozen_finalized:
                    reviewed_count += 1

        if selected_count > 0:
            item["active_learning_selected_turns"] = selected_count
            item["active_learning_reviewed_turns"] = reviewed_count
            item["conversation_locked"] = reviewed_count >= selected_count
            item["has_feedback"] = reviewed_count > 0
            focused_items.append(item)
            reviewed_turns += reviewed_count

    focused_items.sort(key=lambda item: item.get("order_index", 10**9))
    info = _frozen_active_learning_info(selection, True, "turns")
    info["reviewed"] = reviewed_turns
    return focused_items, info


def get_current_user_doc():
    """
    ترجع وثيقة المستخدم الحالي من Firestore
    بناءً على uid الموجود في الـ session.
    """
    uid = session.get("uid")
    if not uid:
        return None

    snap = db.collection("users").document(uid).get()
    return snap if snap.exists else None


def get_user_full_name(user_doc):
    """
    ترجع الاسم الكامل: firstName + lastName
    لو ما فيه بيانات يرجع 'User'
    """
    if not user_doc:
        return "User"

    data = user_doc.to_dict()
    prof = data.get("profile", {})
    first = prof.get("firstName", "")
    last = prof.get("lastName", "")

    full = f"{first} {last}".strip()
    return full or "User"


# ------------------ صفحات واجهة (GET) ------------------

# 1) استبدلي دالة index() كاملة بهذا الكود
@app.route("/")
def index():
    return render_template("HomePage.html")
    

    uid      = session["uid"]
    user_doc = db.collection("users").document(uid).get()
    role     = user_doc.to_dict().get("role", "user")

@app.route("/login")
def login_page():
    return render_template("Login.html")

@app.route("/signup")
def signup_page():
    return render_template("signup.html")

@app.route("/verified")
def verified():
    # بعد التحقق، نعيد توجيهه لصفحة نجاح التفعيل
    return render_template("Verified.html")

@app.route("/profile")
def profile_page():
    if not session.get("idToken"):
        return redirect(url_for("login_page"))
    

    uid = session.get("uid")
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        return redirect(url_for("login_page"))

    user_data = user_doc.to_dict()
    first_name = user_data.get("profile", {}).get("firstName", "")
    last_name  = user_data.get("profile", {}).get("lastName", "")
    full_name  = f"{first_name} {last_name}".strip() or "User"

    return render_template("Profile.html", user_data=user_data, user_name=full_name)

@app.route("/createproject")
def create_project_page():
    project_id = request.args.get("id")  # في حال تم فتح الصفحة للتعديل
    return render_template("CreateProject.html", edit_project_id=project_id)


@app.route("/myprojectowner")
def my_project_owner_page():
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    uid = session.get("uid")
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        return redirect(url_for("login_page"))

    user_data  = user_doc.to_dict()
    first_name = user_data.get("profile", {}).get("firstName", "")
    last_name  = user_data.get("profile", {}).get("lastName", "")
    full_name  = f"{first_name} {last_name}".strip() or "User"

    return render_template("myprojectowner.html", user_name=full_name)


def _owner_page_context():
    # Builds the shared owner page context and keeps owner-only pages consistent.
    if not session.get("idToken"):
        return None, redirect(url_for("login_page"))

    uid = session.get("uid")
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        return None, redirect(url_for("login_page"))

    user_data = user_doc.to_dict() or {}
    role = _safe_str(user_data.get("role") or user_data.get("profile", {}).get("role")).strip().lower()
    if role and role not in ("owner", "project owner"):
        abort(403)

    first_name = user_data.get("profile", {}).get("firstName", "")
    last_name = user_data.get("profile", {}).get("lastName", "")
    full_name = f"{first_name} {last_name}".strip() or "User"
    return {"user_name": full_name, "uid": uid}, None


@app.route("/owner/results")
def owner_results_list_page():
    context, response = _owner_page_context()
    if response:
        return response
    return render_template("OwnerResultsList.html", user_name=context["user_name"])


@app.route("/owner/results/<project_id>")
def owner_results_detail_page(project_id):
    context, response = _owner_page_context()
    if response:
        return response

    project = get_project_basic_info(project_id)
    if not project:
        abort(404)
    if project.get("owner_id") != context["uid"]:
        abort(403)

    return render_template(
        "OwnerResults.html",
        user_name=context["user_name"],
        project_id=project_id
    )
@app.route("/api/add_examiner_to_project", methods=["POST"])
def api_add_examiner_to_project():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    owner_uid = session.get("uid")
    data = request.get_json() or {}

    project_id = data.get("project_id")
    examiner_email = data.get("examiner_email")

    if not project_id or not examiner_email:
        return jsonify({"error": "Missing project_id or examiner_email"}), 400

    # نتحقق أن المشروع للمالك الحالي
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    if proj_doc.to_dict().get("owner_id") != owner_uid:
        return jsonify({"error": "Forbidden"}), 403

    # نجيب بيانات الـ Examiner عن طريق الإيميل
    ex_docs = list(
        db.collection("users")
        .where("email", "==", examiner_email)
        .where("role", "==", "examiner")
        .limit(1)
        .stream()
    )

    if not ex_docs:
        return jsonify({"error": "Examiner not found"}), 404

    examiner_uid = ex_docs[0].id
    ex_data = ex_docs[0].to_dict()

    # استخراج اسم examiner
    prof = ex_data.get("profile", {})
    examiner_name = f"{prof.get('firstName','')} {prof.get('lastName','')}".strip()

    # جلب اسم المالك
    owner_doc = db.collection("users").document(owner_uid).get()
    owner_prof = owner_doc.to_dict().get("profile", {})
    owner_name = f"{owner_prof.get('firstName','')} {owner_prof.get('lastName','')}".strip()

    # أوّل شيء نتأكد أنه مو مضاف مسبقًا
    existing = list(
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("examiner_id", "==", examiner_uid)
        .limit(1)
        .stream()
    )
    if existing:
        return jsonify({"error": "Examiner already invited"}), 409

    # إنشاء دعوة جديدة
    inv_ref = db.collection("invitations").document()
    inv_ref.set({
        "project_id": project_id,
        "project_name": proj_doc.to_dict().get("project_name"),
        "owner_id": owner_uid,
        "owner_name": owner_name,
        "examiner_id": examiner_uid,
        "examiner_email": examiner_email,
        "status": "pending",  # مباشرة نضيفه مقبول
        "created_at": SERVER_TIMESTAMP
    })

    return jsonify({
        "message": "Examiner added successfully",
        "examiner_name": examiner_name,
        "examiner_email": examiner_email,
        "examiner_id": examiner_uid
    }), 200
    
@app.route("/api/remove_examiner", methods=["POST"])
def api_remove_examiner():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    owner_uid = session.get("uid")
    data = request.get_json() or {}

    project_id = data.get("project_id")
    examiner_id = data.get("examiner_id")

    if not project_id or not examiner_id:
        return jsonify({"error": "Missing fields"}), 400

    # تأكيد أن المشروع ملك للـ owner
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    if proj_doc.to_dict().get("owner_id") != owner_uid:
        return jsonify({"error": "Forbidden"}), 403

    # البحث عن الدعوة المقبولة
    inv_query = (
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("examiner_id", "==", examiner_id)
        .where("project_id", "==", project_id)
        .limit(1)
        .stream()
    )

    inv_list = list(inv_query)
    if not inv_list:
        return jsonify({"error": "Examiner not assigned"}), 404

    inv_id = inv_list[0].id

    # حذف الدعوة
    db.collection("invitations").document(inv_id).delete()

    # حذف الـ examiner من المهام
    tasks = db.collection("tasks").where("project_ID", "==", project_id).stream()
    batch = db.batch()

    for t in tasks:
        t_data = t.to_dict()
        examiners = t_data.get("examiner_ids", [])
        if examiner_id in examiners:
            new_list = [e for e in examiners if e != examiner_id]
            batch.update(
                db.collection("tasks").document(t.id),
                {"examiner_ids": new_list}
            )

    batch.commit()

    return jsonify({"success": True, "message": "Examiner removed"}), 200


@app.route("/myprojectexaminer")
def myprojectexaminer_page():
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    uid = session.get("uid")
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        return redirect(url_for("login_page"))

    user_data  = user_doc.to_dict()
    first_name = user_data.get("profile", {}).get("firstName", "")
    last_name  = user_data.get("profile", {}).get("lastName", "")
    full_name  = f"{first_name} {last_name}".strip() or "User"

    return render_template("myprojectexaminer.html", user_name=full_name)


@app.route("/ownerdashboard")
def owner_dashboard_page():
    # 1) نتحقق أن المستخدم مسجل دخول
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    # 2) نجيب الـ UID من الـ session
    uid = session.get("uid")

    # 3) نجيب بيانات المستخدم من Firestore
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        return redirect(url_for("login_page"))

    # 4) نستخرج الاسم
    user_data  = user_doc.to_dict()
    first_name = user_data.get("profile", {}).get("firstName", "")
    last_name  = user_data.get("profile", {}).get("lastName", "")
    full_name  = f"{first_name} {last_name}".strip() or "User"

    # 5) نرسل الاسم للصفحة
    return render_template("Ownerdashboard.html", user_name=full_name)

@app.route("/examinerdashboard")
def examiner_dashboard_page():
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    uid = session.get("uid")
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        return redirect(url_for("login_page"))

    user_data = user_doc.to_dict()
    first_name = user_data.get("profile", {}).get("firstName", "")
    last_name = user_data.get("profile", {}).get("lastName", "")
    full_name = f"{first_name} {last_name}".strip() or "User"

    return render_template("Examinerdashboard.html", user_name=full_name)

@app.route("/projectdetailsowner/<project_id>")
def project_details_owner(project_id):
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    owner_uid = session["uid"]

    # نتحقق أن المشروع فعلاً للـ owner
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        abort(404)

    proj_data = proj_doc.to_dict()
    if proj_data.get("owner_id") != owner_uid:
        abort(403)

    # نجيب بيانات المستخدم لعرض الاسم في الهيدر
    user_doc = db.collection("users").document(owner_uid).get()
    if not user_doc.exists:
        abort(404)

    user_data = user_doc.to_dict()
    first_name = user_data.get("profile", {}).get("firstName", "")
    last_name  = user_data.get("profile", {}).get("lastName", "")
    full_name  = f"{first_name} {last_name}".strip() or "User"

    return render_template(
        "ProjectDetailsOwner.html",
        user_name=full_name,
        project_id=project_id
    )
    
@app.route("/api/project_json_owner/<project_id>")
def api_project_json_owner(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    owner_uid = session["uid"]

    # نتأكد أن المشروع للـ owner
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    proj = proj_doc.to_dict()
    if proj.get("owner_id") != owner_uid:
        return jsonify({"error": "Forbidden"}), 403

    # 🔥 نجيب معلومات الـ Owner (نفس طريقة examiner)
    owner_doc = db.collection("users").document(owner_uid).get()
    if not owner_doc.exists:
        return jsonify({"error": "Owner not found"}), 404

    owner_data = owner_doc.to_dict()
    prof = owner_data.get("profile", {})

    owner_name = f"{prof.get('firstName', '')} {prof.get('lastName', '')}".strip()
    owner_email = owner_data.get("email", "")

    # 🔥 نجيب عدد المقبولين
    accepted_count = sum(
        1
        for _ in db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("status", "==", "accepted")
        .stream()
    )

    return jsonify({
        "project_name": proj.get("project_name"),
        "description": proj.get("description"),
        "domain": proj.get("domain", []),
        "category": proj.get("category"),
        "generated_from_scratch": proj.get("generated_from_scratch", False),  # ✅ جديد
        "dataset_url": proj.get("dataset_url", ""),
        "examiners_accepted": accepted_count,

        # 🔥🔥 أهم شي أضفناهم:
        "owner_name": owner_name,
        "owner_email": owner_email
    })
    
# ------------- قائمة Examiners المقبولين (للـ Owner) -------------
@app.route("/api/project_examiners_owner/<project_id>")
def api_project_examiners_owner(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    owner_uid = session["uid"]

    # تأكيد أن المشروع ملك للـ owner
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    proj_data = proj_doc.to_dict()
    if proj_data.get("owner_id") != owner_uid:
        return jsonify({"error": "Forbidden"}), 403

    # نجيب جميع المقبولين
    accepted = (
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("status", "==", "accepted")
        .stream()
    )

    examiners = []
    for inv in accepted:
        d = inv.to_dict()
        ex_id = d.get("examiner_id")
        user_doc = db.collection("users").document(ex_id).get()
        if not user_doc.exists:
            continue

        u = user_doc.to_dict()
        prof = u.get("profile", {})

        name = f"{prof.get('firstName', '')} {prof.get('lastName', '')}".strip()
        email = u.get("email", "")

        examiners.append({
            "id": ex_id,
            "name": name,
            "email": email
        })

    return jsonify({"examiners": examiners})

# --------------------------------------------------
# صفحة تفاصيل المشروع للمُقيّم (Examiner)
# --------------------------------------------------
@app.route("/projectdetailsexaminer/<project_id>")
def project_details_examiner(project_id):
   
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    examiner_uid = session["uid"]

    # نتحقق أن الم examiner قبل الدعوة
    inv = (
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("examiner_id", "==", examiner_uid)
        .where("status", "==", "accepted")
        .limit(1)
        .get()
    )
    if not inv:
        abort(404)  # أو redirect 404 page

    # نجيب بياناته لعرض الاسم بالهيدر
    user_doc = db.collection("users").document(examiner_uid).get()
    if not user_doc.exists:
        abort(404)

    user_data = user_doc.to_dict()
    first_name = user_data.get("profile", {}).get("firstName", "")
    last_name = user_data.get("profile", {}).get("lastName", "")
    full_name = f"{first_name} {last_name}".strip() or "User"

    return render_template("ProjectDetailsExaminer.html",
                         user_name=full_name,
                         project_id=project_id)
# ------------------ صفحة تفاصيل المشروع للمُقيّم (Examiner) ------------------
# --------------------------------------------------

def _get_owner_info(owner_uid):
    owner_doc = db.collection("users").document(owner_uid).get()
    if not owner_doc.exists:
        return {"name": "Unknown", "email": ""}

    data = owner_doc.to_dict()
    prof = data.get("profile", {})
    name = f"{prof.get('firstName', '')} {prof.get('lastName', '')}".strip()
    email = data.get("email", "")

    return {"name": name, "email": email}

    
@app.route("/api/project_json/<project_id>")
def api_project_json(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    examiner_uid = session["uid"]

    # نتحقق أن ال examiner قبل الدعوة
    inv = (
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("examiner_id", "==", examiner_uid)
        .where("status", "==", "accepted")
        .limit(1)
        .get()
    )
    if not inv:
        return jsonify({"error": "Project not found or you are not a member"}), 404

    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404
    proj = proj_doc.to_dict()

    # نجيب بيانات الأونر
    owner_info = _get_owner_info(proj["owner_id"])

    # نعدّ عدد الـ examiners الذين قبلوا الدعوة
    accepted_count = sum(
        1
        for _ in db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("status", "==", "accepted")
        .stream()
    )

    return jsonify(
        {
            "project_name": proj.get("project_name"),
            "description": proj.get("description"),
            "owner_name": owner_info["name"],
            "owner_email": owner_info["email"],
            "domain": proj.get("domain", []),
            "category": proj.get("category"),
            "generated_from_scratch": proj.get("generated_from_scratch", False),  # ✅ جديد
            "examiners_accepted": accepted_count,
            "dataset_url": proj.get("dataset_url", ""),
        }
    )
@app.route("/feedback")
def feedback_page():
    return render_template("feedback.html")
# ------------- قائمة Examiners المقبولين في مشروع معين -------------
@app.route("/api/project_examiners/<project_id>")
def api_project_examiners(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    # نتحقق أن السائل مقبول هو الآخر
    examiner_uid = session["uid"]
    inv = (
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("examiner_id", "==", examiner_uid)
        .where("status", "==", "accepted")
        .limit(1)
        .get()
    )
    if not inv:
        return jsonify({"error": "Forbidden"}), 403

    # نجيب كل المقبولين
    accepted = (
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("status", "==", "accepted")
        .stream()
    )

    examiners = []
    for a in accepted:
        ex_id = a.to_dict().get("examiner_id")
        ex_doc = db.collection("users").document(ex_id).get()
        if not ex_doc.exists:
            continue
        prof = ex_doc.to_dict().get("profile", {})
        name = f"{prof.get('firstName', '')} {prof.get('lastName', '')}".strip() or "Unknown"
        email = ex_doc.to_dict().get("email", "")
        examiners.append({
            "id": ex_id,
            "name": name,
            "email": email,
            "is_you": ex_id == examiner_uid
        })

    return jsonify({"examiners": examiners})
# ============= INVITATIONS APIs =============

@app.route("/invitation")
def invitation_page():
    """صفحة Invitations (GET)"""
    if not session.get("idToken"):
        return redirect(url_for("login_page"))
    uid = session["uid"]
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        return redirect(url_for("login_page"))
    first_name = user_doc.to_dict().get("profile", {}).get("firstName", "")
    last_name = user_doc.to_dict().get("profile", {}).get("lastName", "")
    full_name = f"{first_name} {last_name}".strip() or "User"
    return render_template("invitation.html", user_name=full_name)

@app.route("/api/invitations", methods=["GET"])
def api_invitations():
    """جلب الدعوات (JSON)"""
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401
    uid = session["uid"]
    docs = (
        db.collection("invitations")
        .where("examiner_id", "==", uid)
        .where("status", "==", "pending")
        .stream()
    )
    out = []
    for d in docs:
        data = d.to_dict()
        out.append({
            "id": d.id,
            "project_name": data.get("project_name"),
            "owner_name": data.get("owner_name"),
            "status": data.get("status"),
        })
    return jsonify({"invitations": out})

@app.route("/api/invitations/<invitation_id>", methods=["PATCH"])
def api_update_invitation(invitation_id):
    """قبول أو رفض دعوة"""
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session["uid"]
    data = request.get_json() or {}
    new_status = data.get("status", "").strip().lower()  # تنظيف الإدخال

    # ✅ توحيد الحالة إلى accepted / declined
    if new_status in ["accept", "accepted"]:
        new_status = "accepted"
    elif new_status in ["decline", "declined"]:
        new_status = "declined"
    else:
        return jsonify({"error": "Invalid status"}), 400

    inv_ref = db.collection("invitations").document(invitation_id)
    inv_doc = inv_ref.get()
    if not inv_doc.exists:
        return jsonify({"error": "Invitation not found"}), 404
    if inv_doc.to_dict().get("examiner_id") != uid:
        return jsonify({"error": "Forbidden"}), 403

    inv_ref.update({"status": new_status})
    return jsonify({"message": f"Invitation {new_status}ed successfully"}), 200

@app.route("/api/volunteers", methods=["GET"])
def api_volunteers():
    # نجيب المستخدمين اللي دورهم examiner واللي مفعلين volunteer.optIn
    docs = (
        db.collection("users")
        .where("role", "==", "examiner")
        .where("volunteer.optIn", "==", True)
        .stream()
    )

    volunteers = []
    for d in docs:
        data = d.to_dict()
        prof = data.get("profile", {})
        volunteers.append({
            "name": f"{prof.get('firstName','')} {prof.get('lastName','')}".strip(),
            "handle": "@" + prof.get("firstName","").lower(),
            "email": data.get("email", ""),
            "tag": prof.get("specialization", "Volunteer")
        })

    return jsonify({"volunteers": volunteers})

# ------------------ Examiner Accepted Projects ------------------
@app.route("/api/accepted_projects", methods=["GET"])
def api_accepted_projects():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    examiner_id = session["uid"]

    invitations = (
        db.collection("invitations")
        .where("examiner_id", "==", examiner_id)
        .where("status", "==", "accepted")
        .stream()
    )

    projects = []

    for inv in invitations:
        inv_data = inv.to_dict() or {}
        project_id = inv_data.get("project_id")

        project_doc = db.collection("projects").document(project_id).get()
        if not project_doc.exists:
            continue

        proj = project_doc.to_dict() or {}

        # نجمع حالات التاسكات الخاصة بهذا الـ examiner فقط داخل هذا المشروع
        personal_statuses = []
        task_docs = db.collection("tasks").where("project_ID", "==", project_id).stream()

        for t in task_docs:
            td = t.to_dict() or {}
            task_id = td.get("task_ID") or t.id
            examiner_ids = td.get("examiner_ids", []) or []

            # إذا التسك مو مسند لهذا الـ examiner نتجاهله
            if examiner_id not in examiner_ids:
                continue

            global_status = _normalize_task_status(td.get("status"))
            conversation_type = (td.get("conversation_type") or "").strip().lower()
            max_turns = int(td.get("number_of_turns", 0) or 0)

            # الحالة الشخصية الافتراضية
            personal_status = global_status

            # فقط مهام المحادثة نحسبها شخصيًا من الرسائل
            if conversation_type in ("human-ai", "human-human") and max_turns > 0:
                your_turn = 0
                try:
                    if conversation_type == "human-ai":
                        conv_ref = rtdb.reference(f"llm_conversations/{task_id}/messages")
                        raw = conv_ref.get() or {}
                        msgs = list(raw.values()) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])

                        count_for_me = 0
                        for m in msgs:
                            if not isinstance(m, dict):
                                continue
                            ex_id = m.get("examiner_id") or m.get("sender_id")
                            if ex_id != examiner_id:
                                continue
                            if m.get("sender_type") != "Ex":
                                continue
                            count_for_me += 1

                        your_turn = min(count_for_me, max_turns)

                    else:
                        conv_ref = rtdb.reference(f"hh_conversations/{task_id}/messages")
                        raw = conv_ref.get() or {}
                        msgs = list(raw.values()) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
                        msgs.sort(key=lambda m: m.get("created_at", ""))

                        ex_ids = examiner_ids or list({
                            m.get("examiner_id") or m.get("sender_id")
                            for m in msgs
                            if isinstance(m, dict) and (m.get("examiner_id") or m.get("sender_id"))
                        })

                        your_turn = _compute_hh_turns_for_examiner(msgs, examiner_id, ex_ids)
                        your_turn = min(your_turn, max_turns)

                    if your_turn >= max_turns:
                        personal_status = "completed"
                    elif your_turn > 0:
                        personal_status = "progress"
                    else:
                        personal_status = "pending"

                except Exception as e:
                    app.logger.exception("Failed to compute personal task status for task %s: %s", task_id, e)
                    personal_status = global_status

            personal_statuses.append(personal_status)

        # اشتقاق حالة المشروع للـ examiner
        if not personal_statuses:
            project_status = "pending"
        elif all(s == "completed" for s in personal_statuses):
            project_status = "completed"
        elif any(s in ("progress", "completed") for s in personal_statuses):
            project_status = "progress"
        else:
            project_status = "pending"

        projects.append({
            "project_id": project_id,
            "project_name": proj.get("project_name"),
            "owner_name": inv_data.get("owner_name"),
            "domain": proj.get("domain", []),
            "category": proj.get("category"),
            "generated_from_scratch": proj.get("generated_from_scratch", False),
            "status": project_status,
        })

    return jsonify({"projects": projects})




def _normalize_task_status(raw):
    s = str(raw or "").strip().lower()
    if s in ("completed", "done"):
        return "completed"
    if s in ("progress", "in_progress", "in-progress", "active"):
        return "progress"
    return "pending"


def _derive_project_status_from_tasks(task_statuses):
    # task_statuses: list مثل ["pending","progress","completed"]
    if not task_statuses:
        return "pending"

    if all(s == "completed" for s in task_statuses):
        return "completed"

    if any(s == "progress" for s in task_statuses):
        return "progress"

    return "pending"

# ------------------ هنا عشان تطلع المشاريع في صفحة الاونر ماي بروجكت------------------
@app.route("/api/my_projects", methods=["GET"])
def api_my_projects():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")
    projects_ref = db.collection("projects").where("owner_id", "==", uid).stream()

    projects = []
    for doc in projects_ref:
        data = doc.to_dict()
        project_id = doc.id

        # 🔹 نحسب عدد الـ examiners اللي قبلوا المشروع
        accepted_invitations = db.collection("invitations") \
            .where("project_id", "==", project_id) \
            .where("status", "==", "accepted") \
            .stream()
        accepted_count = sum(1 for _ in accepted_invitations)


         # 🔹 نحسب حالة المشروع من حالات التاسكات فقط
        task_docs = db.collection("tasks").where("project_ID", "==", project_id).stream()
        task_statuses = []
        for t in task_docs:
            td = t.to_dict() or {}
            task_statuses.append(_normalize_task_status(td.get("status")))

        project_status = _derive_project_status_from_tasks(task_statuses)


        projects.append({
            "project_id": project_id,
            "project_name": data.get("project_name"),
            "domain": data.get("domain", []),
            "category": data.get("category"),
            "generated_from_scratch": data.get("generated_from_scratch", False),
            "examiners": accepted_count,  # ✅ العدد الحقيقي
            "status": project_status,
        })

    return jsonify({"projects": projects})

def ingest_owner_dataset_to_rtdb(category, owner_id, project_id, dataset_id, raw_bytes):
    """
    تخزّن ملف CSV في Realtime Database تحت:
      datasets/uploaded_news أو datasets/uploaded_conversations

    - كل ديتاست لها dataset_id واحد ثابت
    - كل صف داخل الديتاست ينحفظ تحت auto key
    - نستخدم payload للصف كامل زي ما هو من CSV
    """
    if not raw_bytes:
        return 0

    # نحدد الفرع حسب نوع الديتاست
    cat = (category or "").strip().lower()
    if cat in ("news", "article", "articles"):
        branch = "uploaded_news"
    elif cat in ("conversation", "conversations", "chat", "chats"):
        branch = "uploaded_conversations"
    else:
        print(f"[ingest] Unknown category '{category}', skipping RTDB ingest.")
        return 0

    # نقرأ الـ CSV كنص
    text = raw_bytes.decode("utf-8", errors="ignore")
    f = io.StringIO(text)
    reader = csv.DictReader(f)

    base_ref = rtdb.reference("datasets").child(branch).child(dataset_id)
    count = 0

    for row in reader:
        row = _normalize_uploaded_csv_row(row)
        data = {
            "dataset_id": dataset_id,      # 👈 ثابت لكل الصفوف اللي من نفس الديتاست
            "project_id": project_id,
            "owner_id": owner_id,
            "payload": row,                # الصف كامل
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source_type": "owner_upload",
        }
        base_ref.push(data)  # auto key من Realtime
        count += 1

    print(f"[ingest] Inserted {count} rows into datasets/{branch} for dataset_id={dataset_id}")
    return count

# ------------------ Create Project (مع إنشاء سجلات invitations منفصلة) ------------------
@app.route("/api/create_project", methods=["POST"])
def api_create_project():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")
    if not uid:
        return jsonify({"error": "Missing user ID"}), 401

    # نقرأ البيانات سواء من form أو JSON
    data = request.form if request.form else (request.json or {})

    project_name = data.get("project_name")
    description  = data.get("description")
    category     = data.get("category")
    generated_from_scratch = str(data.get("generated_from_scratch", "false")).lower() == "true"
    dataset_id = str(uuid.uuid4())
      
 # === منع إنشاء مشروع News بدون Dataset ===
    if category and category.lower() in ["article", "news", "news article"]:
      file_check = request.files.get("dataset")
      if not file_check or not file_check.filename:
        return jsonify({
            "error": "Dataset file is required for News Article projects."
        }), 400


    if hasattr(data, "getlist"):
        domains = data.getlist("domain")
    else:
        domains = data.get("domain", [])

    examiners_raw = data.get("invited_examiners", "[]")
    try:
        examiners = json.loads(examiners_raw) if isinstance(examiners_raw, str) else examiners_raw
    except json.JSONDecodeError:
        examiners = []

    if not project_name or not description or not category:
        return jsonify({"error": "Missing required fields"}), 400

    # نقرأ ملف الديتاست دون تخزينه في Storage
    dataset_url = ""   # ما نستخدم Storage حالياً
    raw_bytes   = None

    file = request.files.get("dataset")
    if file and file.filename:
        raw_bytes = file.read()
        file.seek(0)

    # جلب بيانات الأونر من Firestore
    owner_doc = db.collection("users").document(uid).get()
    if not owner_doc.exists:
        return jsonify({"error": "Owner not found"}), 404

    owner_data = owner_doc.to_dict()
    owner_name = f"{owner_data.get('profile', {}).get('firstName', '')} {owner_data.get('profile', {}).get('lastName', '')}".strip()

    # إنشاء سجل المشروع
    project_id = str(uuid.uuid4())
    project_doc = {
    "project_ID": project_id,
    "project_name": project_name,
    "description": description,
    "domain": domains,
    "category": category,
    "generated_from_scratch": generated_from_scratch,
    "created_at": datetime.utcnow().isoformat() + "Z",
    "owner_id": uid,
    "dataset_id": dataset_id,
    "invited_examiners": [ex.get("email") for ex in examiners],
    "status": "active",
}


    db.collection("projects").document(project_id).set(project_doc)

    # إنشاء الدعوات في Collection منفصل
    batch = db.batch()
    for ex in examiners:
        email = ex.get("email")
        if not email:
            continue

        examiner_docs = list(
            db.collection("users")
              .where("email", "==", email)
              .where("role", "==", "examiner")
              .limit(1)
              .stream()
        )
        if not examiner_docs:
         return jsonify({"error": "Invalid examiner information"}), 400

        examiner_uid = examiner_docs[0].id
        invitation_ref = db.collection("invitations").document()
        invitation_data = {
            "project_id": project_id,
            "project_name": project_name,
            "owner_id": uid,
            "owner_name": owner_name,
            "examiner_id": examiner_uid,
            "status": "pending",
            "created_at": SERVER_TIMESTAMP,
            "examiner_email": email,
        }
        batch.set(invitation_ref, invitation_data)

    if examiners:
        batch.commit()

    # إدخال الديتاست إلى Realtime Database لو فيه ملف
    if raw_bytes:
        try:
            ingest_owner_dataset_to_rtdb(category, uid, project_id, dataset_id, raw_bytes)
        except Exception as e:
            app.logger.exception("Failed to ingest owner dataset into Realtime: %s", e)

    # ✅ في النهاية لازم نرجّع Response واضح دائماً
    return jsonify({
        "message": "Project created successfully",
        "project_ID": project_id,
        "dataset_id": dataset_id,
    }), 201
@app.route("/api/update_project/<project_id>", methods=["POST"])
def api_update_project(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")

    proj_ref = db.collection("projects").document(project_id)
    proj_doc = proj_ref.get()

    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    if proj_doc.to_dict().get("owner_id") != uid:
        return jsonify({"error": "Forbidden"}), 403

    # يدعم form-data أو JSON
    data = request.form if request.form else (request.get_json(silent=True) or {})

    name = (data.get("project_name") or "").strip()
    desc = (data.get("description") or "").strip()
    category = (data.get("category") or "").strip()

    if hasattr(data, "getlist"):
        domains = data.getlist("domain")
    else:
        domains = data.get("domain", [])
        if isinstance(domains, str):
            domains = [domains]
        elif not isinstance(domains, list):
            domains = []

    if not name or not desc or not category:
        return jsonify({"error": "Missing required fields"}), 400

    update_data = {
        "project_name": name,
        "description": desc,
        "category": category,
        "domain": domains,
        "updated_at": datetime.utcnow().isoformat() + "Z"
    }

    proj_ref.update(update_data)

    return jsonify({"message": "Project updated successfully"}), 200


# ------------------ حذف مشروع ------------------
@app.route("/api/delete_project/<project_id>", methods=["DELETE"])
def api_delete_project(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")
    project_ref = db.collection("projects").document(project_id)
    project_doc = project_ref.get()

    if not project_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    if project_doc.to_dict().get("owner_id") != uid:
        return jsonify({"error": "Forbidden"}), 403

    invitations = db.collection("invitations").where("project_id", "==", project_id).stream()
    batch = db.batch()
    for inv in invitations:
        batch.delete(db.collection("invitations").document(inv.id))
    batch.delete(project_ref)
    batch.commit()

    return jsonify({"message": "Project deleted successfully"}), 200

# ========== إرسال دعوة جديدة للـ Examiner ==========
@app.route("/api/send_invitation", methods=["POST"])
def api_send_invitation():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    owner_uid = session["uid"]
    data       = request.get_json() or {}
    examiner_email = data.get("examiner_email")   # أو id حسب تصميمك
    project_id     = data.get("project_id")

    if not examiner_email or not project_id:
        return jsonify({"error": "Missing examiner_email or project_id"}), 400

    # نجيب بيانات الـ Examiner من Firestore بالـ email
    examiner_docs = list(db.collection("users").where("email", "==", examiner_email).limit(1).stream())
    if not examiner_docs:
        return jsonify({"error": "Examiner email not found"}), 404
    examiner_uid = examiner_docs[0].id

    # نجيب بيانات الـ Project للتأكد أنه تابع للـ Owner
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists or proj_doc.to_dict().get("owner_id") != owner_uid:
        return jsonify({"error": "Project not found or not yours"}), 403

    # نجيب اسم الـ Owner للعرض
    owner_doc = db.collection("users").document(owner_uid).get()
    owner_name = f"{owner_doc.to_dict()['profile']['firstName']} {owner_doc.to_dict()['profile']['lastName']}".strip()

    # ننشئ الدعوة
    inv_doc = {
        "project_id":   project_id,
        "project_name": proj_doc.to_dict().get("project_name"),
        "owner_id":     owner_uid,
        "owner_name":   owner_name,
        "examiner_id":  examiner_uid,
        "status":       "pending",
        "created_at":   SERVER_TIMESTAMP
    }
    db.collection("invitations").add(inv_doc)

    return jsonify({"message": "Invitation sent successfully"}), 201

# ------------------ مصادقة (POST APIs) ------------------

# إنشاء حساب جديد
@app.route("/api/signup", methods=["POST"])
def api_signup():
    data = request.form if request.form else (request.json or {})

    email    = data.get("email")
    password = data.get("password")

    # 👈 نقرأ اليوزر نيم من الفورم
    username = data.get("username") or data.get("displayName", "")

    role       = data.get("role", "user")
    first_name = data.get("firstName", "")
    last_name  = data.get("lastName", "")
    gender     = data.get("gender", "")
    interests  = data.get("interests", "")
    github     = data.get("github", "")
    linkedin   = data.get("linkedin", "")

    volunteer_opt_in = str(data.get("volunteerOptIn", "false")).lower() == "true"
    specialization   = data.get("specialization", "")
    description      = data.get("description", "")

    # ✅ نتأكد من كل القيم الأساسية
    if not email or not password or not username:
        return jsonify({"error": "email, password and username are required"}), 400

    # ✅ نتأكد أن اليوزر نيم يونيك
    existing = list(
        db.collection("users")
          .where("username", "==", username)
          .limit(1)
          .stream()
    )
    if existing:
        return jsonify({
            "error": "USERNAME_TAKEN",
            "message": "This username is already in use."
        }), 409


    volunteer_opt_in = str(data.get("volunteerOptIn", "false")).lower() == "true"
    specialization   = data.get("specialization", "")
    description      = data.get("description", "")

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    try:
        res = rest_signup(email, password)  # Firebase
        uid = res["localId"]

        # إرسال رابط التحقق
        send_verification_email(res["idToken"])

        user_doc = {
            "uid": uid,
            "email": email,
             "username": username,  
            "role": role,
            "createdAt": SERVER_TIMESTAMP,
            "updatedAt": SERVER_TIMESTAMP,
            "profile": {
                "firstName": first_name,
                "lastName":  last_name,
                "gender":    gender,
                "interests": interests,
                "github":    github,
                "linkedin":  linkedin,
            }
        }

        if role == "examiner":
            user_doc["profile"]["specialization"] = specialization
            user_doc["profile"]["description"]    = description
            user_doc["volunteer"] = {"optIn": volunteer_opt_in}

        db.collection("users").document(uid).set(user_doc)

        # حفظ بيانات التسجيل مؤقتاً
        session["email"] = email
        session["temp_password"] = password

        # توجيه صفحة التحقق
        return render_template("CheckEmail.html")

    except Exception as e:
        try:
            return jsonify(e.response.json()), e.response.status_code
        except:
            return jsonify({"error": str(e)}), 500

# verification_email هنا كل مايخص
def send_verification_email(id_token):
    url = "https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key=AIzaSyChtQ2FaenDwe7k7bfRB8Cw5G_5C4f_xt4"
    payload = {
        "requestType": "VERIFY_EMAIL",
        "idToken": id_token,
        "continueUrl": "http://127.0.0.1:5000/verified"
    }
    headers = {"Content-Type": "application/json"}

    r = requests.post(url, json=payload, headers=headers)

    print("\n🔥 VERIFY EMAIL RESPONSE 🔥")
    print("Status:", r.status_code)
    print("Body:", r.text)
    print("🔥 ----------------------🔥\n")

    return r

@app.route("/auto-login")
def auto_login():
    email = session.get("email")
    password = session.get("temp_password")

    if not email or not password:
        return redirect(url_for("login_page"))

    try:
        res = rest_signin(email, password)

        session["idToken"] = res["idToken"]
        session["uid"] = res["localId"]

        role = db.collection("users").document(res["localId"]).get().to_dict().get("role")

        if role == "owner":
            return redirect(url_for("owner_dashboard_page"))
        elif role == "examiner":
            return redirect(url_for("examiner_dashboard_page"))
        else:
            return redirect(url_for("profile_page"))

    except:
        return redirect(url_for("login_page"))

# تسجيل الدخول
@app.route("/api/signin", methods=["POST"])
def api_signin():
    data = request.form if request.form else (request.json or {})

    # المستخدم يقدر يكتب email أو username في نفس الحقل
    identifier = (data.get("identifier") or data.get("email") or "").strip()
    password   = data.get("password")

    if not identifier or not password:
        return render_template(
            "Login.html",
            error="Email/username and password are required."
        ), 400

    # 1) نحدّد الإيميل
    email = identifier

    # لو ما فيه @ نفترض أنه Username ونبحث عنه في Firestore
    if "@" not in identifier:
        # تأكدّي أن عندك حقل اسمه "username" داخل وثيقة المستخدم في Firestore
        user_q = (
            db.collection("users")
              .where("username", "==", identifier)
              .limit(1)
              .stream()
        )
        user_docs = list(user_q)
        if not user_docs:
            # Username غير صحيح
            return render_template(
                "Login.html",
                error="Invalid username or password. Please try again."
            ), 401

        user_data = user_docs[0].to_dict()
        email = user_data.get("email")
        if not email:
            return render_template(
                "Login.html",
                error="User record is missing email."
            ), 500

    try:
        # 2) نسجّل الدخول في Firebase Auth باستخدام الإيميل اللي استخرجناه
        res = rest_signin(email, password)

        # 3) نتأكد من تفعيل الإيميل من Firebase (نفس كودك السابق)
        url = "https://identitytoolkit.googleapis.com/v1/accounts:lookup?key=AIzaSyChtQ2FaenDwe7k7bfRB8Cw5G_5C4f_xt4"
        r = requests.post(url, json={"idToken": res["idToken"]})
        user_info = r.json()

        email_verified = user_info["users"][0]["emailVerified"]
        if not email_verified:
            return render_template(
                "Login.html",
                error="Please verify your email before logging in."
            )

        uid = res["localId"]
        user_doc = db.collection("users").document(uid).get()
        if not user_doc.exists:
            session.clear()
            return render_template(
                "Login.html",
                error="User data not found."
            ), 401

        role = user_doc.to_dict().get("role", "user")

        session["idToken"] = res["idToken"]
        session["uid"] = uid

        if role == "owner":
            return redirect(url_for("owner_dashboard_page"))
        elif role == "examiner":
            return redirect(url_for("examiner_dashboard_page"))
        else:
            return redirect(url_for("profile_page"))

    except Exception:
        app.logger.exception("Signin failed")
        return render_template(
            "Login.html",
            error="Invalid email/username or password. Please try again."
        ), 401


# تسجيل الخروج
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# إرسال رابط إعادة تعيين كلمة المرور
@app.route("/api/reset", methods=["POST"])
def api_reset():
    email = request.form.get("email") or (request.json or {}).get("email")
    if not email:
        return jsonify({"error": "email is required"}), 400
    try:
        send_password_reset(email)
        return jsonify({"message": "Password reset email sent"})
    except Exception as e:
        try:
            return jsonify(e.response.json()), e.response.status_code
        except:
            return jsonify({"error": str(e)}), 500



# صحة الخادم
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/api/update-profile", methods=["POST"])
def api_update_profile():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")
    user_ref = db.collection("users").document(uid)
    snap = user_ref.get()
    if not snap.exists:
        return jsonify({"error": "User not found"}), 404

    data = request.get_json() or {}

    # ===== Username validation =====
    username = (data.get("username") or "").strip().lstrip("@")
    if not username:
        return jsonify({
            "error": "USERNAME_REQUIRED",
            "message": "Please enter a username."
        }), 400

    existing_username = list(
        db.collection("users")
          .where("username", "==", username)
          .limit(1)
          .stream()
    )
    if existing_username and existing_username[0].id != uid:
        return jsonify({
            "error": "USERNAME_TAKEN",
            "message": "This username is already taken. Please choose another one."
        }), 409

    # ===== Email validation =====
    new_email = (data.get("newEmail") or data.get("email") or "").strip().lower()
    if not new_email:
        return jsonify({
            "error": "EMAIL_REQUIRED",
            "message": "Please enter an email."
        }), 400

    if ("@" not in new_email) or ("." not in new_email.split("@")[-1]):
        return jsonify({
            "error": "INVALID_EMAIL",
            "message": "Please enter a valid email address."
        }), 400

    existing_email = list(
        db.collection("users")
          .where("email", "==", new_email)
          .limit(1)
          .stream()
    )
    if existing_email and existing_email[0].id != uid:
        return jsonify({
            "error": "EMAIL_TAKEN",
            "message": "This email is already in use."
        }), 409

    # باقي البيانات
    first_name     = (data.get("firstName") or "").strip()
    last_name      = (data.get("lastName") or "").strip()
    gender         = (data.get("gender") or "").strip()
    specialization = (data.get("specialization") or "").strip()
    github         = (data.get("github") or "").strip()
    linkedin       = (data.get("linkedin") or "").strip()
    description    = (data.get("description") or "").strip()
    interests      = (data.get("interests") or "").strip()

    updates = {
        "updatedAt": SERVER_TIMESTAMP,
        "username": username,
        "email": new_email,
        "profile.firstName": first_name,
        "profile.lastName": last_name,
        "profile.gender": gender,
        "profile.specialization": specialization,
        "profile.github": github,
        "profile.linkedin": linkedin,
        "profile.description": description,
        "profile.interests": interests,
    }

    user_ref.update(updates)

    return jsonify({"message": "Profile updated successfully"}), 200



@app.route("/forgot", methods=["GET", "POST"])
def forgot_page():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()

        if not email:
            flash("Please enter your email address.", "error")
            return render_template("ForgotPassword.html")

        try:
            # تستخدمين نفس الفنكشن اللي عندك في auth_rest
            send_password_reset(email)
            flash("If this email is registered, we’ve sent a reset link.", "success")
        except Exception as e:
            print("Reset error:", e)
            flash("Something went wrong. Please try again.", "error")

        return render_template("ForgotPassword.html", email=email)

    # GET
    return render_template("ForgotPassword.html")

#CraetaTask
@app.route("/api/create_task", methods=["POST"])
def api_create_task():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    owner_uid = session.get("uid")

    data = request.get_json() or {}
    project_id = data.get("project_id")
    task_name = data.get("task_name")
    examiner_uids = data.get("examiner_ids", [])
    task_description = (data.get("task_description") or "").strip()


    if not project_id or not task_name:
        return jsonify({"error": "Missing required fields"}), 400

    # ---- نجيب بيانات المشروع ----
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    proj_data = proj_doc.to_dict()

    # نتأكد أن هذا المشروع ملك للـ Owner الحالي
    if proj_data.get("owner_id") != owner_uid:
        return jsonify({"error": "Forbidden"}), 403

    category = proj_data.get("category", "").lower()


    if not examiner_uids:
        return jsonify({"error": "No valid examiners selected"}), 400

    # ---- تجهيز بيانات الـ Task ----
    task_id = str(uuid.uuid4())

    task_doc = {
        "task_ID": task_id,
        "project_ID": project_id,
        "task_name": task_name,
        "examiner_ids": examiner_uids,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "created_by": owner_uid,
        "status": "pending",
    }

  # ✅ حالة Generate Conversation فقط
    is_generated_conversation = (
        category in ["conversation", "conversations", "chat", "chats"]
        and bool(proj_data.get("generated_from_scratch", False))
    )

    if is_generated_conversation:
        task_type = (data.get("task_type") or "").strip().lower()
        conversation_type = (data.get("conversation_type") or "").strip().lower()
        number_of_turns = data.get("number_of_turns")

        try:
            number_of_turns = int(number_of_turns) if number_of_turns is not None else None
        except (TypeError, ValueError):
            number_of_turns = None

        # 🔒 Generate Conversation:
        # لا نسمح بـ Model Selection / Labeling إلا إذا فيه (على الأقل) مهمة محادثة مكتملة
        if task_type in ["model_selection", "labeling"]:
            conv_tasks = db.collection("tasks").where("project_ID", "==", project_id).stream()
            has_completed_conversation = False

            for t in conv_tasks:
                td = t.to_dict() or {}
                ctype = (td.get("conversation_type") or "").strip().lower()
                tstatus = (td.get("status") or "").strip().lower()

                if ctype in ["human-ai", "human-human"] and tstatus == "completed":
                    has_completed_conversation = True
                    break

            if not has_completed_conversation:
                return jsonify({
                    "error": "Please complete at least one conversation task first."
                }), 400

            task_doc["task_type"] = task_type
            task_doc["task_description"] = task_description

            if task_type == "model_selection" and len(examiner_uids) != 1:
                return jsonify({"error": "Model Selection task requires exactly 1 examiner"}), 400

        # 2) Human-AI / Human-Human
        elif conversation_type in ["human-ai", "human-human"]:
            if number_of_turns is None or not (2 <= number_of_turns <= 7):
                return jsonify({"error": "number_of_turns must be 2–7"}), 400

            task_doc["conversation_type"] = conversation_type
            task_doc["number_of_turns"] = number_of_turns

            if conversation_type == "human-ai" and len(examiner_uids) != 1:
                return jsonify({"error": "Human-AI task requires exactly 1 examiner"}), 400

            if conversation_type == "human-human" and len(examiner_uids) != 2:
                return jsonify({"error": "Human-Human task requires exactly 2 examiners"}), 400
        else:
            return jsonify({"error": "Invalid mode for generated conversation project"}), 400

    elif category == "article":
        # ✅ Article: نفس منطقك القديم بدون تغيير
        task_type = data.get("task_type")

        if task_type not in ["model_selection", "labeling"]:
            return jsonify({"error": "Invalid task_type. Must be 'model_selection' or 'FeedBack'"}), 400

        task_doc["task_type"] = task_type

        if task_type == "model_selection" and len(examiner_uids) != 1:
            return jsonify({"error": "Model Selection task requires exactly 1 examiner"}), 400

    else:
        # ✅ Uploaded Conversation: model selection / labeling مثل المقالات، بدون Human-Human أو Human-AI
        task_type = (data.get("task_type") or "").strip().lower()

        if task_type not in ["model_selection", "labeling"]:
            return jsonify({"error": "Invalid task_type. Must be 'model_selection' or 'labeling'"}), 400

        task_doc["task_type"] = task_type
        task_doc["task_description"] = task_description

        if task_type == "model_selection" and len(examiner_uids) != 1:
            return jsonify({"error": "Model Selection task requires exactly 1 examiner"}), 400


    # ---- حفظ الـ Task ----
    db.collection("tasks").document(task_id).set(task_doc)

    return jsonify({
        "message": "Task created successfully",
        "task_id": task_id
    }), 201

@app.route("/projects/<project_id>/tasks/create")
def create_task_page(project_id):
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    owner_uid = session.get("uid")
    proj_doc = db.collection("projects").document(project_id).get()

    if not proj_doc.exists:
        abort(404)

    proj_data = proj_doc.to_dict()

    if proj_data.get("owner_id") != owner_uid:
        abort(403)

    category = (proj_data.get("category", "") or "").lower().strip()

    # ✅ نمرر حالة Generate Conversation للواجهة
    is_generated_conversation = (
        category in ["conversation", "conversations", "chat", "chats"]
        and bool(proj_data.get("generated_from_scratch", False))
    )

    return render_template(
        "CreateTask.html",
        project_id=project_id,
        category=category,
        is_generated_conversation=is_generated_conversation
    )

@app.route("/api/project_examiners_for_task/<project_id>")
def get_project_examiners_for_task(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    # تأكيد أن المشروع موجود
    project_doc = db.collection("projects").document(project_id).get()
    if not project_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    # 🟦 نجيب جميع الـ examiners اللي قبلوا الدعوة
    accepted = (
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("status", "==", "accepted")
        .stream()
    )

    examiners_list = []

    for inv in accepted:
        data = inv.to_dict()
        uid = data.get("examiner_id")

        # جلب بيانات اليوزر
        user_doc = db.collection("users").document(uid).get()
        if user_doc.exists:
            info = user_doc.to_dict()
            prof = info.get("profile", {})

            examiners_list.append({
                "uid": uid,
                "email": info.get("email", ""),
                "name": f"{prof.get('firstName','')} {prof.get('lastName','')}".strip()
            })

    return jsonify({"examiners": examiners_list})
@app.route("/api/project_tasks/<project_id>")
def api_project_tasks(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    owner_uid = session.get("uid")

    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    if proj_doc.to_dict().get("owner_id") != owner_uid:
        return jsonify({"error": "Forbidden"}), 403

    tasks_ref = (
        db.collection("tasks")
        .where("project_ID", "==", project_id)
        .stream()
    )

    tasks = []
    for t in tasks_ref:
        data = t.to_dict()

        examiner_ids = data.get("examiner_ids", []) or []

        examiner_emails = []
        for ex_id in examiner_ids:
            ex_doc = db.collection("users").document(ex_id).get()
            if ex_doc.exists:
                email = ex_doc.to_dict().get("email", "")
                if email:
                    examiner_emails.append(email)

        raw_model = (data.get("selected_model") or "").lower()
        selected_model_name = data.get("selected_model_name") or (
            "RNN" if raw_model == "rnn"
            else "Logistic Regression" if raw_model in ("logreg", "logistic")
            else None
        )

        tasks.append({
            "id": data.get("task_ID"),
            "title": data.get("task_name"),
            "status": data.get("status", "pending"),
            "conversationType": data.get("conversation_type"),
            "taskType": data.get("task_type"),
            "turns": data.get("number_of_turns"),
            "examinerCount": len(examiner_emails),
            "primaryExaminerEmail": examiner_emails[0] if examiner_emails else "",
            "examinerEmails": examiner_emails,
            "selected_model_name": selected_model_name,
            "selected_model": data.get("selected_model"),
        })


    return jsonify({"tasks": tasks})

# ------------------ Examiner Tasks (Only tasks assigned to this examiner) ------------------
@app.route("/api/examiner_tasks/<project_id>")
def api_examiner_tasks(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    examiner_uid = session.get("uid")

    # تأكيد أن الـ Examiner مقبول في المشروع
    inv = (
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("examiner_id", "==", examiner_uid)
        .where("status", "==", "accepted")
        .limit(1)
        .get()
    )
    if not inv:
        return jsonify({"error": "Forbidden"}), 403

    tasks_ref = (
        db.collection("tasks")
        .where("project_ID", "==", project_id)
        .stream()
    )

    tasks = []
    for t in tasks_ref:
        data = t.to_dict()

        task_id          = data.get("task_ID")
        conversation_type = data.get("conversation_type", None)
        max_turns         = int(data.get("number_of_turns", 0) or 0)

        examiner_ids = data.get("examiner_ids", []) or []
        assigned = examiner_uid in examiner_ids

        # ✅ نجمع كل إيميلات الممتحنين في الكرت
        examiner_emails = []
        for ex_id in examiner_ids:
            ex_doc = db.collection("users").document(ex_id).get()
            if ex_doc.exists:
                em = ex_doc.to_dict().get("email", "")
                if em:
                    examiner_emails.append(em)

        # -----------------------------
        # 🔵 حساب وضعك أنت على هذا التسك
        # -----------------------------
        personal_status = "pending"
        your_turn = 0

        if assigned and conversation_type in ("human-ai", "human-human") and max_turns > 0:
            try:
                # نجيب كل رسائل هذا التاسك من RTDB
                if conversation_type == "human-ai":
                    conv_ref = rtdb.reference(f"llm_conversations/{task_id}/messages")
                else:
                    conv_ref = rtdb.reference(f"hh_conversations/{task_id}/messages")

                raw = conv_ref.get() or {}

                if isinstance(raw, dict):
                    msgs = list(raw.values())
                elif isinstance(raw, list):
                    msgs = raw
                else:
                    msgs = []

                # -------------------------
                # 👇 حساب عدد التيرنز لك
                # -------------------------
                if conversation_type == "human-ai":
                    # نفس المنطق القديم: كل رسالة من الـ Examiner = 1 turn
                    count_for_me = 0
                    for m in msgs:
                        if not isinstance(m, dict):
                            continue

                        ex_id = m.get("examiner_id") or m.get("sender_id")
                        if ex_id != examiner_uid:
                            continue

                        if m.get("sender_type") != "Ex":
                            continue

                        count_for_me += 1

                    your_turn = count_for_me

                else:
                    # 👈 Human-Human: نستخدم نفس منطق الـ runs
                    # نتأكد من قائمة الـ examiners
                    ex_ids = examiner_ids or list({
                        m.get("examiner_id") or m.get("sender_id")
                        for m in msgs
                        if isinstance(m, dict) and (m.get("examiner_id") or m.get("sender_id"))
                    })

                    # نرتب الرسائل زمنيًا
                    msgs.sort(key=lambda m: m.get("created_at", ""))

                    # نستخدم الفنكشن اللي فوق
                    your_turn = _compute_hh_turns_for_examiner(msgs, examiner_uid, ex_ids)

                # ما نتعدى الحد الأقصى
                if max_turns > 0:
                    your_turn = min(your_turn, max_turns)

            except Exception as e:
                app.logger.exception(
                    "Failed to compute turns for task %s: %s", task_id, e
                )
                your_turn = 0

            if your_turn >= max_turns:
                personal_status = "completed"
            elif your_turn > 0:
                personal_status = "progress"
            else:
                personal_status = "pending"
        else:
            # لو ما هي مهمة محادثة أو مو مسندة لك، نرجع الحالة العامة
            personal_status = data.get("status", "pending")

        tasks.append({
            "task_id": task_id,
            "task_name": data.get("task_name"),
            "task_type": data.get("task_type"),  #CONV OR ART
            "task_description": data.get("task_description", ""),

            # ✅ هذه التي تستخدمها الكروت والفلاتر
            "status": personal_status,

            # الحالة العامة لو احتجتيها
            "global_status": data.get("status", "pending"),

            "conversation_type": conversation_type,
            "number_of_turns": max_turns,
            "current_turn_for_you": your_turn,
            "is_assigned_to_you": assigned,
            "assignment_label": examiner_emails[0] if examiner_emails else "",
            "examiner_emails": examiner_emails,
            "examiner_count": len(examiner_emails),
            "selected_model": data.get("selected_model"),
"selected_model_key": data.get("selected_model_key"),
"selected_model_label": (
    "RNN" if data.get("selected_model") == "rnn"
    else "Logistic Regression" if data.get("selected_model") in ("logreg", "logistic")
    else None
),
"selectedModel": data.get("selected_model"),
"selectedModelLabel": (
    "RNN" if data.get("selected_model") == "rnn"
    else "Logistic Regression" if data.get("selected_model") in ("logreg", "logistic")
    else None
),

        })

    return jsonify({"tasks": tasks})

@app.route("/api/tasks/<task_id>", methods=["GET"])
def api_get_task(task_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    owner_uid = session.get("uid")

    # 1) نتأكد التاسك موجود
    task_doc = db.collection("tasks").document(task_id).get()
    if not task_doc.exists:
        return jsonify({"error": "Task not found"}), 404

    task_data = task_doc.to_dict() or {}
    project_id = task_data.get("project_ID")

    # 2) نتأكد المشروع موجود
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    # 3) صلاحية: لازم صاحب المشروع
    if proj_doc.to_dict().get("owner_id") != owner_uid:
        return jsonify({"error": "Forbidden"}), 403

    # 4) نرجع كل بيانات التعديل
    return jsonify({
        "task_ID": task_data.get("task_ID"),
        "task_name": task_data.get("task_name", ""),
        "examiner_ids": task_data.get("examiner_ids", []),
        "task_type": task_data.get("task_type"),
        "conversation_type": task_data.get("conversation_type"),
        "number_of_turns": task_data.get("number_of_turns"),
        "task_description": task_data.get("task_description", "")
    }), 200

@app.route("/api/tasks/<task_id>/delete", methods=["POST"])
def api_delete_task(task_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")

    # نجيب المهمة
    task_ref = db.collection("tasks").document(task_id)
    task_doc = task_ref.get()

    if not task_doc.exists:
        return jsonify({"error": "Task not found"}), 404

    task_data = task_doc.to_dict()
    project_id = task_data.get("project_ID")

    # نجيب المشروع للتأكد أن هذا الـ Owner هو صاحب المشروع
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    if proj_doc.to_dict().get("owner_id") != uid:
        return jsonify({"error": "Forbidden"}), 403

    # 🗑️ حذف المهمة
    task_ref.delete()

    return jsonify({"success": True, "message": "Task deleted successfully"}), 200




@app.route("/api/update_task/<task_id>", methods=["PATCH"])
def api_update_task(task_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    owner_uid = session.get("uid")

    # نجيب المهمة
    task_ref = db.collection("tasks").document(task_id)
    task_doc = task_ref.get()

    if not task_doc.exists:
        return jsonify({"error": "Task not found"}), 404

    task_data = task_doc.to_dict()
    project_id = task_data.get("project_ID")

    # نتأكد أن ال Owner هو صاحب المشروع
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    if proj_doc.to_dict().get("owner_id") != owner_uid:
        return jsonify({"error": "Forbidden"}), 403

    # نقرأ البيانات الجديدة
    data = request.get_json() or {}

    new_name = data.get("task_name", "").strip()
    new_examiners = data.get("examiner_ids", [])

    if not new_name:
        return jsonify({"error": "Task name is required"}), 400

    if not new_examiners:
        return jsonify({"error": "At least one examiner is required"}), 400

    # نحدّث فقط اللي تبينه
    update_data = {
        "task_name": new_name,
        "examiner_ids": new_examiners,
        "updated_at": datetime.utcnow().isoformat() + "Z"
    }

    task_ref.update(update_data)

    return jsonify({"message": "Task updated successfully"}), 200
 
# ===================================================================
# ------------- صفحة Human ↔ AI Conversation (Front) ---------------
# ===================================================================
 
@app.route("/conversation-ai")
def conversation_ai_page():
    if not session.get("idToken"):
        return redirect(url_for("login_page"))
 
    user_doc = get_current_user_doc()
    user_name = get_user_full_name(user_doc) if user_doc else "User"
 
    # نقرأ taskId من الرابط
    task_id = request.args.get("taskId")

    # 👇 نقرأ project_id من الرابط
    project_id = request.args.get("projectId")
 
    # قيم افتراضية
    max_turns = 6
    task_title = "Conversation task topic"
 
    if task_id:
        try:
            task_snapshot = db.collection("tasks").document(task_id).get()
            if task_snapshot.exists:
                task_data = task_snapshot.to_dict()
                max_turns = int(task_data.get("number_of_turns", 6))
                task_title = task_data.get("task_name", task_title)
        except Exception as e:
            app.logger.exception("Error loading task in conversation_ai_page: %s", e)
 
    return render_template(
        "ConversationH-AI.html",
        user_name=user_name,
        max_turns=max_turns,
        task_title=task_title,
        task_id=task_id,
        project_id=project_id  
)


# ===================================================================
# ------------- صفحة Human ↔ Human Conversation (Front) ------------
# ===================================================================
 
@app.route("/conversation-hh")
def conversation_hh_page():
    # لازم يكون مسجل دخول
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    # نجيب اسم المستخدم
    user_doc = get_current_user_doc()
    user_name = get_user_full_name(user_doc) if user_doc else "User"

    # 🔹 هنا كنا نقرأ بس taskId
    task_id = request.args.get("taskId")
    project_id = request.args.get("projectId")  # <-- ✅ (1) أضفنا قراءة projectId من الكويري

    # قيم افتراضية
    max_turns = 6
    task_title = "Human ↔ Human conversation task"

    if task_id:
        try:
            task_snapshot = db.collection("tasks").document(task_id).get()
            if task_snapshot.exists:
                task_data  = task_snapshot.to_dict()
                max_turns  = int(task_data.get("number_of_turns", 6))
                task_title = task_data.get("task_name", task_title)
                conv_type  = task_data.get("conversation_type")

                # لو طلع النوع مو Human-Human نحوله لصفحة AI زي ما كان
                if conv_type != "human-human":
                    return redirect(
                        url_for("conversation_ai_page", taskId=task_id, projectId=project_id)
                    )
        except Exception as e:
            app.logger.exception("Error loading task in conversation_hh_page: %s", e)

    return render_template(
        "ConversationH-H.html",
        user_name=user_name,
        max_turns=max_turns,
        task_title=task_title,
        task_id=task_id,
        project_id=project_id,  # <-- ✅ (2) نمرر project_id للتمبليت
    )
# ==========================
#  AI Conversation Reply API
# ==========================
@app.route("/api/ai_reply", methods=["POST"])
def api_ai_reply():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    user_message = data.get("message", "").strip()
    task_id = data.get("taskId")

    if not user_message or not task_id:
        return jsonify({"error": "Missing message or taskId"}), 400

    sender_id = session.get("uid")

    # اسم المستخدم
    sender_doc = db.collection("users").document(sender_id).get()
    sender_name = "User"
    if sender_doc.exists:
        prof = sender_doc.to_dict().get("profile", {})
        sender_name = f"{prof.get('firstName','')} {prof.get('lastName','')}".strip() or "User"

    ref = rtdb.reference(f"llm_conversations/{task_id}/messages")

    existing = ref.get() or {}
    count_user = sum(1 for x in existing.values() if x.get("sender_type") == "Ex") + 1

    turn_id = str(uuid.uuid4())
    now_iso = datetime.utcnow().isoformat() + "Z"

    # 🧍‍♀️ 1) نحفظ رسالة المستخدم
    ref.push({
        "turn_id": turn_id,
        "task_id": task_id,
        "turn_number": count_user,
        "sender_type": "Ex",
        "examiner_id": sender_id,
        "sender_name": sender_name,
        "message": user_message,
        "created_at": now_iso,
    })

    # 🤖 2) نجيب رد AI
    try:
        ai_response = generate_reply(user_message)
    except Exception:
        ai_response = "Sorry, I couldn’t generate a reply."

    # 🧠 3) نحفظ رسالة الـ AI بنفس turn_id
    ref.push({
        "turn_id": turn_id,
        "task_id": task_id,
        "turn_number": count_user,
        "sender_type": "LLM",
        "sender_name": "AI",
        "message": ai_response,
        "created_at": datetime.utcnow().isoformat() + "Z",
    })

    # ✅ 4) نحدّث حالة التاسك لو اكتملت
    _update_ai_task_status_if_completed(task_id)

    return jsonify({"reply": ai_response}), 200

# ==========================
#  AI Conversation message API
# ==========================
@app.route("/api/ai/messages", methods=["GET"])
def api_ai_get_messages():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    task_id = request.args.get("taskId")
    if not task_id:
        return jsonify({"error": "Missing taskId"}), 400

    uid = session.get("uid")

    try:
        ref = rtdb.reference(f"llm_conversations/{task_id}/messages")
        raw = ref.get() or {}

        if isinstance(raw, dict):
            rows = list(raw.values())
        elif isinstance(raw, list):
            rows = raw
        else:
            rows = []

        # نرتب بالوقت
        rows.sort(key=lambda m: m.get("created_at", ""))

               # نخلي كل Examiner يشوف محادثته هو فقط
        my_turn_ids = {
            m.get("turn_id")
            for m in rows
            if isinstance(m, dict)
            and (m.get("examiner_id") == uid or m.get("sender_id") == uid)
        }

        messages = []
        your_turn = 0

        for m in rows:
            if not isinstance(m, dict):
                continue
            if m.get("turn_id") not in my_turn_ids:
                continue

            sender_type = m.get("sender_type")
            text = m.get("message", "")

            if sender_type == "LLM":
                side = "ai"
            else:
                side = "you"
                your_turn = max(your_turn, int(m.get("turn_number", 0) or 0))

            messages.append({
                "text": text,
                "side": side,
            })

             # حالة التاسك من Firestore + عدد التيرنز
        task_status = "pending"
        max_turns = 0
        try:
            task_doc = db.collection("tasks").document(task_id).get()
            if task_doc.exists:
                tdata = task_doc.to_dict()
                task_status = tdata.get("status", "pending")
                max_turns = int(tdata.get("number_of_turns", 0) or 0)

                # ✅ لو مكتوبة completed لكن إنتِ ما خلصتي دوراتك
                if task_status == "completed" and max_turns > 0 and your_turn < max_turns:
                    task_status = "progress"
        except Exception as e:
            app.logger.exception("AI get: failed to load task status: %s", e)

        return jsonify({
            "messages": messages,
            "currentTurn": your_turn,
            "taskStatus": task_status
        }), 200
        
    except Exception as e:
        print("🔥 AI get error:", e)
        return jsonify({"error": "Server error"}), 500


# ==================================================
# Task Update H-H
# ==================================================
def _compute_hh_turns_for_examiner(msgs, examiner_id, examiner_ids):
    """
    يحسب عدد الـ turns لممتحِن واحد في محادثة Human-Human.

    turn واحد = (block من self) + (block من peer) أو العكس.
    البلوك = مجموعة رسائل متتالية من نفس الطرف.
    """
    speaker_seq = []

    for m in msgs:
        if not isinstance(m, dict):
            continue

        sender = m.get("examiner_id") or m.get("sender_id")
        if sender not in examiner_ids:
            continue

        if sender == examiner_id:
            speaker_seq.append("self")
        else:
            speaker_seq.append("peer")

    if not speaker_seq:
        return 0

    # ندمج البلوكات المتتالية المتشابهة
    runs = []
    last = None
    for s in speaker_seq:
        if s != last:
            runs.append(s)
            last = s

    # كل بلوكين متتاليين (self+peer أو peer+self) = 1 turn مكتمل
    turns = len(runs) // 2
    return turns

def _update_hh_task_status_if_completed(task_id):
    """
    يشيّك إذا كل الـ examiners في محادثة Human-Human
    خلصوا عدد الـ turns المطلوب بناءً على تعريفك للـ turn:

    turn واحد = (block من رسائل self) + (block من رسائل peer) أو العكس،
    بغض النظر عن عدد الرسائل داخل كل block.
    """
    try:
        task_ref = db.collection("tasks").document(task_id)
        task_doc = task_ref.get()
        if not task_doc.exists:
            return

        task_data = task_doc.to_dict()

        # نتأكد أنها مهمة Human-Human
        if task_data.get("conversation_type") != "human-human":
            return

        max_turns = int(task_data.get("number_of_turns", 0) or 0)
        if max_turns <= 0:
            return

        examiner_ids = task_data.get("examiner_ids") or []

        # لو ما فيه examiner_ids (حالات قديمة) نجمعهم من الرسائل
        conv_ref = rtdb.reference(f"hh_conversations/{task_id}/messages")
        raw = conv_ref.get() or {}
        if isinstance(raw, dict):
            msgs = list(raw.values())
        elif isinstance(raw, list):
            msgs = raw
        else:
            msgs = []

        if not examiner_ids:
            examiner_ids = list({
                m.get("examiner_id")
                for m in msgs
                if isinstance(m, dict) and m.get("examiner_id")
            })

        if not examiner_ids or not msgs:
            return

        # نرتب الرسائل زمنيًا
        msgs.sort(key=lambda m: m.get("created_at", ""))

        # نحسب عدد الـ turns لكل ممتحِن
        turns_per_examiner = {}
        for ex_id in examiner_ids:
            t = _compute_hh_turns_for_examiner(msgs, ex_id, examiner_ids)
            if max_turns > 0:
                t = min(t, max_turns)
            turns_per_examiner[ex_id] = t

        # نكمّل التاسك فقط لو كلهم وصلوا max_turns
        completed = all(turns_per_examiner.get(e, 0) >= max_turns for e in examiner_ids)

        if completed and task_data.get("status") != "completed":
            task_ref.update({"status": "completed"})

    except Exception as e:
        app.logger.exception("Failed to update HH task status: %s", e)
# ==================================================
# Task Update Ai
# ==================================================

def _update_ai_task_status_if_completed(task_id):
    """
    يشيّك إذا كل الـ examiners وصلوا لعدد الـ turns المطلوب
    (محادثة Human-AI) ولو نعم يحدّث حالة التاسك إلى completed.
    """
    try:
        task_ref = db.collection("tasks").document(task_id)
        task_doc = task_ref.get()
        if not task_doc.exists:
            return

        task_data = task_doc.to_dict()

        # نتأكد أنه Human-AI
        if task_data.get("conversation_type") != "human-ai":
            return

        max_turns = int(task_data.get("number_of_turns", 0) or 0)
        if max_turns <= 0:
            return

        examiner_ids = task_data.get("examiner_ids") or []
        if not examiner_ids:
            # لو ما فيه examiner_ids لأي سبب، نجمعهم من الرسائل
            conv_ref = rtdb.reference(f"llm_conversations/{task_id}/messages")
            raw = conv_ref.get() or {}
            if isinstance(raw, dict):
                msgs = raw.values()
            elif isinstance(raw, list):
                msgs = raw
            else:
                msgs = []

            examiner_ids = list({
                m.get("examiner_id")
                for m in msgs
                if isinstance(m, dict) and m.get("examiner_id")
            })

        if not examiner_ids:
            return

        # نقرأ كل رسائل هذه المحادثة
        conv_ref = rtdb.reference(f"llm_conversations/{task_id}/messages")
        raw = conv_ref.get() or {}
        if isinstance(raw, dict):
            msgs = raw.values()
        elif isinstance(raw, list):
            msgs = raw
        else:
            msgs = []

        # نحسب كم رسالة كتب كل examiner (sender_type == "Ex")
        counts = {ex_id: 0 for ex_id in examiner_ids}
        for m in msgs:
            if not isinstance(m, dict):
                continue
            if m.get("sender_type") != "Ex":
                continue
            ex_id = m.get("examiner_id")
            if ex_id in counts:
                counts[ex_id] += 1

        # لو كل واحد وصل على الأقل max_turns → نكمّل التاسك
        completed = all(counts.get(e, 0) >= max_turns for e in examiner_ids)

        if completed and task_data.get("status") != "completed":
            task_ref.update({"status": "completed"})

    except Exception as e:
        app.logger.exception("Failed to update AI task status: %s", e)



# ==================================================
# 🔹 Human ↔ Human Conversation APIs (Realtime DB)
# ==================================================

# 1) جلب كل رسائل التاسك من الـ Realtime DB
@app.route("/api/hh/messages", methods=["GET"])
def api_hh_get_messages():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    task_id = request.args.get("taskId")
    if not task_id:
        return jsonify({"error": "Missing taskId"}), 400

    uid = session.get("uid")

    try:
        ref = rtdb.reference(f"hh_conversations/{task_id}/messages")
        raw = ref.get() or {}

        # نحولها لقائمة
        if isinstance(raw, dict):
            rows = list(raw.values())
        elif isinstance(raw, list):
            rows = raw
        else:
            rows = []

        # نرتب الرسائل بالوقت
        rows.sort(key=lambda m: m.get("created_at", ""))

        messages = []
        speaker_sequence = []   # "you" أو "peer" بالترتيب الزمني

        for m in rows:
            if not isinstance(m, dict):
                continue

            sender_id = m.get("examiner_id") or m.get("sender_id")
            sender_name = (m.get("sender_name") or "User").strip() or "User"
            text = m.get("message", "")

            if not sender_id:
                continue

            if sender_id == uid:
                side = "you"
                speaker_sequence.append("you")
            else:
                side = "peer"
                speaker_sequence.append("peer")

            initial = (sender_name[0].upper() if sender_name else "U")

            messages.append({
                "text": text,
                "side": side,
                "authorInitial": initial,
            })

        # ======== حساب عدد الـ turns من وجهة نظرك ========
        # نحول sequence إلى blocks متتالية مختلفة
        runs = []
        last = None
        for s in speaker_sequence:
            if s != last:
                runs.append(s)
                last = s

        # كل (you + peer) أو (peer + you) = turn واحد
        your_turn = len(runs) // 2

        task_status = "pending"
        max_turns = 0

        try:
            task_doc = db.collection("tasks").document(task_id).get()
            if task_doc.exists:
                tdata = task_doc.to_dict()
                task_status = tdata.get("status", "pending")
                max_turns = int(tdata.get("number_of_turns", 0) or 0)

                if max_turns > 0:
                    your_turn = min(your_turn, max_turns)

                # لو التاسك مكتوب completed بس لسه ما خلصتي كل التيرنز → نخليها progress
                if task_status == "completed" and max_turns > 0 and your_turn < max_turns:
                    task_status = "progress"
        except Exception as e:
            app.logger.exception("HH get: failed to load task status: %s", e)

        return jsonify({
            "messages": messages,
            "currentTurn": your_turn,
            "taskStatus": task_status
        }), 200

    except Exception as e:
        print("🔥 HH get error:", e)
        return jsonify({"error": "Server error"}), 500

# 2) حفظ رسالة جديدة في الـ Realtime DB
# 2) حفظ رسالة جديدة في الـ Realtime DB
@app.route("/api/hh/send", methods=["POST"])
def api_hh_send():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}

    task_id = data.get("task_id") or data.get("taskId")
    message = (data.get("message") or data.get("text") or "").strip()

    if not task_id or not message:
        return jsonify({"error": "Missing taskId or message"}), 400

    sender_id = session.get("uid")
    if not sender_id:
        return jsonify({"error": "Missing uid in session"}), 401

    sender_doc = db.collection("users").document(sender_id).get()
    sender_name = "User"
    if sender_doc.exists:
        prof = sender_doc.to_dict().get("profile", {})
        sender_name = f"{prof.get('firstName', '')} {prof.get('lastName', '')}".strip() or "User"

    ref = rtdb.reference(f"hh_conversations/{task_id}/messages")

    # 🧠 نجيب الرسائل الموجودة
    existing = ref.get() or {}
    rows = []
    if isinstance(existing, dict):
        rows = list(existing.values())
    elif isinstance(existing, list):
        rows = existing

    # ==============================
    # 🔒 منع إرسال رسالتين ورا بعض
    # ==============================
    if rows:
        # نرتّب الرسائل بالوقت
        try:
            rows_sorted = sorted(rows, key=lambda r: r.get("created_at", ""))
        except Exception:
            rows_sorted = rows

        last_msg = rows_sorted[-1]

        # نركز على رسائل الـ examiners فقط
        if (
            isinstance(last_msg, dict)
            and last_msg.get("sender_type") == "Ex"
            and last_msg.get("examiner_id") == sender_id
        ):
            # نفس الشخص أرسل آخر رسالة → لازم ينتظر الثاني
            return jsonify({
                "error": "WAIT_FOR_PEER",
                "message": "You must wait for the other examiner to reply before sending another message."
            }), 400

    # نحسب turn_number الخاص بهذا الـ examiner فقط
    count_for_this_ex = 0
    for row in rows:
        if isinstance(row, dict) and row.get("examiner_id") == sender_id:
            count_for_this_ex += 1

    next_turn_number = count_for_this_ex + 1
    turn_id = str(uuid.uuid4())

    try:
        ref.push({
            "turn_id": turn_id,
            "task_id": task_id,
            "turn_number": next_turn_number,
            "sender_type": "Ex",
            "examiner_id": sender_id,
            "message": message,
            "sender_name": sender_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
        })

        _update_hh_task_status_if_completed(task_id)

        return jsonify({"success": True, "message": "Message saved"}), 200

    except Exception as e:
        print("🔥 HH send error:", e)
        return jsonify({"error": "Server error"}), 500



@app.route("/api/hh/messages_owner", methods=["GET"])
def api_hh_messages_owner():
    """
    عرض محادثة Human ↔ Human للـ Owner من مسار:
    hh_conversations/{taskId}/messages/{pushId}
    """
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    task_id = request.args.get("taskId")
    if not task_id:
        return jsonify({"error": "taskId is required"}), 400

    try:
        messages_ref = rtdb.reference(f"hh_conversations/{task_id}/messages")
        raw = messages_ref.get() or {}

        # نحول النودات إلى list ونرتبها حسب turn_number ثم created_at
        all_msgs = []
        for key, val in (raw or {}).items():
            if not isinstance(val, dict):
                continue
            val["_key"] = key
            all_msgs.append(val)

        def _as_int(x, default=0):
            try:
                return int(x)
            except Exception:
                return default

        all_msgs.sort(
            key=lambda m: (
                _as_int(m.get("turn_number", 0)),
                m.get("created_at") or ""
            )
        )

        # نحدد examiners عشان نوزعهم يسار/يمين
        examiner_side = {}
        side_order = ["left", "right"]

        def get_side_for_examiner(ex_id):
            if not ex_id:
                return "left"
            if ex_id not in examiner_side:
                # أول واحد يصير left، الثاني right
                examiner_side[ex_id] = side_order[len(examiner_side) % 2]
            return examiner_side[ex_id]

        msgs = []
        max_turn = 0

        for m in all_msgs:
            text = m.get("message") or ""
            if not text:
                continue

            turn_number = _as_int(m.get("turn_number", 0))
            if turn_number > max_turn:
                max_turn = turn_number

            examiner_id = m.get("examiner_id")
            sender_name = m.get("sender_name") or "Examiner"

            side = get_side_for_examiner(examiner_id)

            msgs.append({
                "text": text,
                "side": side,  # left / right
                "author": sender_name,
                "authorLabel": sender_name,
                "turnIndex": turn_number,
            })

        return jsonify({
            "messages": msgs,
            "currentTurn": max_turn,
            "isComplete": False,   # ما عندنا فلاغ واضح في السكيمة الحالية
        }), 200

    except Exception as e:
        app.logger.exception("Error in api_hh_messages_owner: %s", e)
        return jsonify({"error": "Server error while loading HH conversation"}), 500

@app.route("/api/llm/messages_owner", methods=["GET"])
def api_llm_messages_owner():
    """
    عرض محادثة Human ↔ LLM للـ Owner من مسار:
    llm_conversations/{taskId}/messages/{pushId}
    """
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    task_id = request.args.get("taskId")
    if not task_id:
        return jsonify({"error": "taskId is required"}), 400

    try:
        messages_ref = rtdb.reference(f"llm_conversations/{task_id}/messages")
        raw = messages_ref.get() or {}

        all_msgs = []
        for key, val in (raw or {}).items():
            if not isinstance(val, dict):
                continue
            val["_key"] = key
            all_msgs.append(val)

        def _as_int(x, default=0):
            try:
                return int(x)
            except Exception:
                return default

        all_msgs.sort(
            key=lambda m: (
                _as_int(m.get("turn_number", 0)),
                m.get("created_at") or ""
            )
        )

        # نفترض إن الـ human عنده examiner_id، والـ LLM غالبًا بدون examiner_id
        examiner_side = {}

        def get_side(msg):
            st = (msg.get("sender_type") or "").lower()
            ex_id = msg.get("examiner_id")

            # لو رسالة من الـ LLM
            if st in ("llm", "ai", "assistant", "model") or (not ex_id):
                return "right"

            # البشري
            if ex_id not in examiner_side:
                examiner_side[ex_id] = "left"
            return examiner_side[ex_id]

        msgs = []
        max_turn = 0

        for m in all_msgs:
            text = m.get("message") or ""
            if not text:
                continue

            turn_number = _as_int(m.get("turn_number", 0))
            if turn_number > max_turn:
                max_turn = turn_number

            sender_name = m.get("sender_name") or "Speaker"

            side = get_side(m)

            msgs.append({
                "text": text,
                "side": side,  # left = human, right = LLM
                "author": sender_name,
                "authorLabel": sender_name,
                "turnIndex": turn_number,
            })


        return jsonify({
            "messages": msgs,
            "currentTurn": max_turn,
            "isComplete": False,
        }), 200

    except Exception as e:
        app.logger.exception("Error in api_llm_messages_owner: %s", e)
        return jsonify({"error": "Server error while loading LLM conversation"}), 500
    
    
  
# ===================================================================


@app.route("/api/project/<project_id>/dataset", methods=["GET"])
def get_project_dataset(project_id):
    """
يجيب كل مقالات الـ dataset حق مشروع معين
    """
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")

    # 1️⃣ نجيب بيانات المشروع من Firestore
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    proj_data = proj_doc.to_dict()
    
    # 2️⃣ نتحقق من الصلاحية (Owner أو Examiner مقبول)
    is_owner = proj_data.get("owner_id") == uid
    
    is_examiner = False
    if not is_owner:
        # نشيك إذا هو examiner مقبول
        inv_docs = list(
            db.collection("invitations")
            .where("project_id", "==", project_id)
            .where("examiner_id", "==", uid)
            .where("status", "==", "accepted")
            .limit(1)
            .stream()
        )
        is_examiner = len(inv_docs) > 0

    if not is_owner and not is_examiner:
        return jsonify({"error": "Forbidden"}), 403

    # 3️⃣ نجيب dataset_id
    dataset_id = proj_data.get("dataset_id")
    if not dataset_id:
        return jsonify({"error": "No dataset found for this project"}), 404

    # 4️⃣ نسحب كل المقالات من Realtime Database
    try:
        ref = rtdb.reference(f"datasets/uploaded_news/{dataset_id}")
        snapshot = ref.get()
        
        if not snapshot:
            return jsonify({"articles": [], "total": 0}), 200

        articles = []
        for push_id, article_data in snapshot.items():
            if not isinstance(article_data, dict):
                continue
                
            payload = article_data.get("payload", {})
            
            # نستخرج النص (fallback)
            title, content = _text_from_news_payload(payload, {})
            
            articles.append({
                "id": push_id,
                "title": title,
                "content": content,
                "full_text": f"{title}. {content}" if title else content
            })

        return jsonify({
            "articles": articles,
            "total": len(articles),
            "dataset_id": dataset_id
        }), 200

    except Exception as e:
        app.logger.exception("Failed to fetch dataset: %s", e)
        return jsonify({"error": "Failed to fetch dataset"}), 500

@app.route("/api/project/<project_id>/analyze_all", methods=["POST"])
def analyze_all_articles(project_id):
    """
    يحلل كل مقالات المشروع بالنموذج ويحفظ النتائج
    """
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")

    # نتحقق إن المستخدم هو الـ Owner
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    if proj_doc.to_dict().get("owner_id") != uid:
        return jsonify({"error": "Only project owner can run batch analysis"}), 403
    
    data = request.get_json(silent=True) or {}
    selected_model = (data.get("model") or "logistic").lower()



    # نسحب الـ dataset
    dataset_id = proj_doc.to_dict().get("dataset_id")
    if not dataset_id:
        return jsonify({"error": "No dataset found"}), 404

    try:
        # نجيب كل المقالات
        ref = rtdb.reference(f"datasets/uploaded_news/{dataset_id}")
        snapshot = ref.get()

        if not snapshot:
            return jsonify({"error": "Dataset is empty"}), 404

        results = []
        human_count = 0
        ai_count = 0
        y_true = []
        y_pred = []

        # نحلل كل مقالة
        for push_id, article_data in snapshot.items():
            if not isinstance(article_data, dict):
                continue

            payload = article_data.get("payload", {})
            
            title, content = _text_from_news_payload(payload, {})
            
            full_text = f"{title}. {content}" if title else content

            # تحليل بالنموذج
            chunks = split_into_3_chunks(full_text)
            
            human_scores = []
            ai_scores = []
            
            for chunk in chunks:
                if selected_model == "rnn":
                    probabilities = rnn_predict_proba([chunk])[0]   # لاحظي [chunk]
                else:
                    probabilities = news_pipeline.predict_proba([chunk])[0]

           #  نحولها إلى float عادي
                h_score, a_score = _news_probability_pair(probabilities, None if selected_model == "rnn" else news_pipeline, selected_model != "rnn")
                human_scores.append(h_score)
                ai_scores.append(a_score)


            
            # المتوسط
            final_human = sum(human_scores) / len(human_scores)
            final_ai = sum(ai_scores) / len(ai_scores)
          
            #4 النتيجة النهائية
            prediction = "AI" if final_ai > final_human else "Human"
            prediction_int = 1 if prediction == "AI" else 0
            ground_truth = _extract_ground_truth_from_payload(payload)
            confidence, uncertainty = _confidence_uncertainty_from_prob(final_ai)
            chunk_details = _news_chunks_from_scores(
                chunks,
                human_scores,
                ai_scores,
                article_id=push_id,
                title=title[:100] if title else "",
                article_prediction=prediction,
                article_prediction_int=prediction_int
            )
            
            if prediction == "Human":
                human_count += 1
            else:
                ai_count += 1

            if ground_truth is not None:
                y_true.append(ground_truth)
                y_pred.append(prediction_int)

            results.append({
                "confidence": _percent_or_none(confidence),
                "uncertainty": _percent_or_none(uncertainty),
    "article_id": push_id,
    "title": title[:100] if title else "",
    "content": full_text[:500] if full_text else "",
    "prediction": prediction,
    "prediction_int": prediction_int,
    "ground_truth": ground_truth,
    "ground_truth_label": _label_text_from_int(ground_truth),
    "human_percentage": round(final_human * 100, 2),
    "ai_percentage": round(final_ai * 100, 2),
    "chunks": chunk_details
})

        # نحفظ النتائج في Firestore
        metrics = _compute_standard_metrics(y_true, y_pred)

        analysis_doc = {
            "project_id": project_id,
            "dataset_id": dataset_id,
            "model_type": selected_model,
            "total_articles": len(results),
            "human_count": human_count,
            "ai_count": ai_count,
            "human_percentage": round((human_count / len(results)) * 100, 2),
            "ai_percentage": round((ai_count / len(results)) * 100, 2),
            "metrics": metrics,
            "analyzed_at": datetime.utcnow().isoformat() + "Z",
            "analyzed_by": uid
        }


        # نرتب النتائج من الأقل ثقة للأعلى
        results.sort(key=lambda x: x["confidence"])


        # نرتب النتائج من الأقل ثقة للأعلى
        results.sort(key=lambda x: x["confidence"])

        db.collection("project_analysis").document(project_id).set(analysis_doc)

        # نحفظ النتائج التفصيلية في Realtime DB
        results_ref = rtdb.reference(f"analysis_results/{project_id}/{selected_model}")
        results_ref.set({
            "summary": analysis_doc,
            "metrics": metrics,
            "details": results
        })

        return jsonify({
            "message": "Analysis complete",
            "summary": analysis_doc,
            "total_analyzed": len(results)
        }), 200

    except Exception as e:
        app.logger.exception("Batch analysis failed: %s", e)
        return jsonify({"error": "Analysis failed"}), 500


# ═══════════════════════════════════════════════════════════════
# 🔬 Model Selection Task Routes
# ═══════════════════════════════════════════════════════════════

@app.route("/task/<task_id>/model-selection")
def model_selection_task_page(task_id):
    """صفحة Model Selection للـ Examiner"""
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    uid = session.get("uid")
    
    # نجيب معلومات Task
    task_doc = db.collection("tasks").document(task_id).get()
    if not task_doc.exists:
        abort(404)
    
    task_data = task_doc.to_dict()
    
    # نتأكد إنه Model Selection Task
    if task_data.get("task_type") != "model_selection":
        abort(403)
    
    # نتأكد إن الـ Examiner مسند له
    examiner_ids = task_data.get("examiner_ids", [])
    if len(examiner_ids) != 1 or uid not in examiner_ids:
        abort(403)
    
    # نجيب اسم المستخدم
    user_name = _reviewer_display_name(uid)
    
    project_id = task_data.get("project_ID")
    

    # ✅ لو المشروع Generated Conversation -> افتح صفحة Conversation Analysis
    proj_doc = db.collection("projects").document(project_id).get()
    proj_data = proj_doc.to_dict() if proj_doc.exists else {}
    category = (proj_data.get("category") or "").strip().lower()

    is_generated_conversation = (
        category in ["conversation", "conversations", "chat", "chats"]
        and bool(proj_data.get("generated_from_scratch", False))
    )

    if is_generated_conversation:
        return redirect(url_for("results_con", projectId=project_id, taskId=task_id))

    is_uploaded_conversation = (
        category in ["conversation", "conversations", "chat", "chats"]
        and not bool(proj_data.get("generated_from_scratch", False))
    )

    if is_uploaded_conversation:
        return redirect(url_for("conversation_analysis_page_examiner", project_id=project_id, taskId=task_id))



    return render_template(
        "ModelSelectionTask.html",
        user_name=user_name,
        user_uid=uid,
        task_id=task_id,
        project_id=project_id,
        task_name=task_data.get("task_name", "Model Selection")
    )


@app.route("/api/task/<task_id>/run_model", methods=["POST"])
def api_run_model(task_id):
    """يشغل Model على Dataset"""
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")
    task_ref, task_data, guard_error = _model_selection_task_guard(task_id, reject_completed=False)
    if guard_error:
        return guard_error
    
    # نتأكد من الصلاحية
    if uid not in task_data.get("examiner_ids", []):
        return jsonify({"error": "Forbidden"}), 403
    
    # نجيب المشروع
    project_id = task_data.get("project_ID")
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404
    
    dataset_id = proj_doc.to_dict().get("dataset_id")
    if not dataset_id:
        return jsonify({"error": "No dataset"}), 404
    
    # نقرأ Model المطلوب
    data = request.get_json() or {}
    model_type = (data.get("model") or "logistic").lower()
    
    try:
        # نسحب Dataset
        ref = rtdb.reference(f"datasets/uploaded_news/{dataset_id}")
        snapshot = ref.get()
        
        if not snapshot:
            return jsonify({"error": "Dataset is empty"}), 404
        
        results = []
        human_count = 0
        ai_count = 0
        y_true = []
        y_pred = []
        
        # نحلل كل مقالة
        for push_id, article_data in snapshot.items():
            if not isinstance(article_data, dict):
                continue
            
            payload = article_data.get("payload", {})
            
            title, content = _text_from_news_payload(payload, {})

            full_text = f"{title}. {content}" if title else content
            
            chunks = split_into_3_chunks(full_text)
            
            human_scores = []
            ai_scores = []
            
            for i, chunk in enumerate(chunks):
                if model_type == "rnn":
                    probabilities = rnn_predict_proba([chunk])[0]
                else:
                    probabilities = news_pipeline.predict_proba([chunk])[0]
                
                h_score, a_score = _news_probability_pair(probabilities, None if model_type == "rnn" else news_pipeline, model_type != "rnn")
                
                human_scores.append(h_score)
                ai_scores.append(a_score)
            
            final_human = sum(human_scores) / len(human_scores)
            final_ai = sum(ai_scores) / len(ai_scores)
            
            prediction = "AI" if final_ai > final_human else "Human"
            prediction_int = 1 if prediction == "AI" else 0
            ground_truth = _extract_ground_truth_from_payload(payload)
            confidence, uncertainty = _confidence_uncertainty_from_prob(final_ai)
            chunk_details = _news_chunks_from_scores(
                chunks,
                human_scores,
                ai_scores,
                article_id=push_id,
                title=title[:100] if title else "Untitled",
                article_prediction=prediction,
                article_prediction_int=prediction_int
            )
            
            if prediction == "Human":
                human_count += 1
            else:
                ai_count += 1

            if ground_truth is not None:
                y_true.append(ground_truth)
                y_pred.append(prediction_int)
            
            results.append({
                "confidence": _percent_or_none(confidence),
                "uncertainty": _percent_or_none(uncertainty),
                "article_id": push_id,
                "title": title[:100] if title else "Untitled",
                "content": content[:500] if content else "",  # ✅ أول 500 حرف
                "prediction": prediction,
                "prediction_int": prediction_int,
                "ground_truth": ground_truth,
                "ground_truth_label": _label_text_from_int(ground_truth),
                "human_percentage": round(final_human * 100, 2),
                "ai_percentage": round(final_ai * 100, 2),
                "chunks": chunk_details  # ✅ تفاصيل الـ Chunks
            })
        
        # ملخص النتائج
        metrics = _compute_standard_metrics(y_true, y_pred)

        summary = {
            "model_type": model_type,
            "total_articles": len(results),
            "human_count": human_count,
            "ai_count": ai_count,
            "human_percentage": round((human_count / len(results)) * 100, 2) if results else 0,
            "ai_percentage": round((ai_count / len(results)) * 100, 2) if results else 0,
            "metrics": metrics
        }
        
        return jsonify({
            "summary": summary,
            "sample_results": results[:10],  # أول 10
            "all_results": results  # ✅ كل النتائج مع التفاصيل
        }), 200
        
    except Exception as e:
        app.logger.exception("Model execution failed: %s", e)
        return jsonify({"error": "Analysis failed"}), 500


@app.route("/api/task/<task_id>/select_model", methods=["POST"])
def api_select_model(task_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")

    task_ref, task_data, guard_error = _model_selection_task_guard(task_id, reject_completed=True)
    if guard_error:
        return guard_error

    data = request.get_json() or {}
    selected_model = (data.get("model") or "logistic").lower()

    if selected_model not in ["logistic", "rnn"]:
        return jsonify({"error": "Invalid model"}), 400

    # منع الاختيار مرة ثانية
    if task_data.get("selected_model"):
        return jsonify({"error": "Model already selected for this task"}), 400

    # نحفظ الاختيار في Firestore
    task_ref.update({
        "selected_model": selected_model,
        "selected_by": uid,
        "selected_at": datetime.utcnow().isoformat() + "Z",
        "status": "completed",
        "selected_model_name": "RNN" if selected_model == "rnn" else "Logistic Regression",
    })

    # نشغل التحليل ونحفظ في RTDB
    try:
        project_id = task_data.get("project_ID")
        proj_doc = db.collection("projects").document(project_id).get()
        dataset_id = proj_doc.to_dict().get("dataset_id") if proj_doc.exists else None

        if dataset_id:
            ref = rtdb.reference(f"datasets/uploaded_news/{dataset_id}")
            snapshot = ref.get() or {}

            results = []
            human_count = 0
            ai_count = 0
            y_true = []
            y_pred = []

            for push_id, article_data in snapshot.items():
                if not isinstance(article_data, dict):
                    continue

                payload = article_data.get("payload", {})
                title, content = _text_from_news_payload(payload, {})
                full_text = f"{title}. {content}" if title else content

                chunks = split_into_3_chunks(full_text)
                human_scores = []
                ai_scores = []

                for i, chunk in enumerate(chunks):
                    if selected_model == "rnn":
                        probabilities = rnn_predict_proba([chunk])[0]
                    else:
                        probabilities = news_pipeline.predict_proba([chunk])[0]

                    h, a = _news_probability_pair(probabilities, None if selected_model == "rnn" else news_pipeline, selected_model != "rnn")
                    human_scores.append(h)
                    ai_scores.append(a)

                final_human = sum(human_scores) / len(human_scores)
                final_ai = sum(ai_scores) / len(ai_scores)
                prediction = "AI" if final_ai > final_human else "Human"
                prediction_int = 1 if prediction == "AI" else 0
                ground_truth = _extract_ground_truth_from_payload(payload)
                confidence, uncertainty = _confidence_uncertainty_from_prob(final_ai)
                chunk_details = _news_chunks_from_scores(
                    chunks,
                    human_scores,
                    ai_scores,
                    article_id=push_id,
                    title=title[:100],
                    article_prediction=prediction,
                    article_prediction_int=prediction_int
                )

                if prediction == "Human":
                    human_count += 1
                else:
                    ai_count += 1

                if ground_truth is not None:
                    y_true.append(ground_truth)
                    y_pred.append(prediction_int)

                results.append({
                    "confidence": _percent_or_none(confidence),
                    "uncertainty": _percent_or_none(uncertainty),
                    "article_id": push_id,
                    "title": title[:100],
                    "content": content[:500],
                    "prediction": prediction,
                    "prediction_int": prediction_int,
                    "ground_truth": ground_truth,
                    "ground_truth_label": _label_text_from_int(ground_truth),
                    "human_percentage": round(final_human * 100, 2),
                    "ai_percentage": round(final_ai * 100, 2),
                    "chunks": chunk_details
                })

            # نحفظ في RTDB - نظف project_id من الأحرف الممنوعة
            safe_pid = project_id.replace(".", "_").replace("#", "_").replace("$", "_").replace("[", "_").replace("]", "_")
            metrics = _compute_standard_metrics(y_true, y_pred)
            results_ref = rtdb.reference(f"analysis_results/{safe_pid}/{selected_model}")
            results_ref.set({
                "summary": {
                    "model_type": selected_model,
                    "total_articles": len(results),
                    "human_count": human_count,
                    "ai_count": ai_count,
                    "metrics": metrics,
                },
                "metrics": metrics,
                "details": results
            })
            model_version = _active_learning_model_version(project_id, selected_model)
            _freeze_active_learning_targets_if_missing(
                project_id,
                "news",
                selected_model,
                model_version,
                _news_active_learning_candidates(project_id, selected_model, results),
                task_data.get("examiner_ids") or []
            )
            _create_detection_version_snapshot(
                project_id,
                task_id,
                selected_model,
                "RNN" if selected_model == "rnn" else "Logistic Regression",
                uid
            )
            app.logger.info("=== Analysis saved to RTDB: analysis_results/%s/%s ===", safe_pid, selected_model)

    except Exception as e:
        app.logger.exception("Failed to run analysis after model selection: %s", e)

    return jsonify({"message": "Model selected successfully"}), 200

# 📝 Examiner Feedback APIs
# ═══════════════════════════════════════════════════════════════

@app.route('/api/article/<article_id>/feedback', methods=['POST'])
def submit_article_feedback(article_id):
    """Submit examiner feedback"""
    try:
        if not session.get('idToken'):
            return jsonify({"error": "Unauthorized"}), 401
        
        user_id = session.get('uid')
        user_doc = db.collection('users').document(user_id).get()
        
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404
        
        user_role = user_doc.to_dict().get('role')
        if user_role != 'examiner':
            return jsonify({"error": "Only examiners can submit feedback"}), 403
        
        data = request.get_json()
        label = data.get('label')
        explanation = data.get('explanation', '').strip()
        
        if not label or label not in ['Human', 'AI']:
            return jsonify({"error": "Invalid label"}), 400
        
        if not explanation:
            return jsonify({"error": "Explanation is required"}), 400
        
        # Get examiner name
        examiner_name = _feedback_examiner_name(user_id)
        
        # Find article
        uploaded_news_ref = rtdb.reference('datasets/uploaded_news')
        article_found = False
        article_path = None
        
        all_datasets = uploaded_news_ref.get() or {}
        for dataset_id, dataset_content in all_datasets.items():
            if isinstance(dataset_content, dict):
                for push_id, article_data in dataset_content.items():
                    if push_id == article_id:
                        article_found = True
                        article_path = f'datasets/uploaded_news/{dataset_id}/{article_id}'
                        break
            if article_found:
                break
        
        if not article_found:
            return jsonify({"error": "Article not found"}), 404
        
        feedback_data = {
            'label': label,
            'explanation': explanation,
            'examiner_name': examiner_name,
            'submitted_at': datetime.utcnow().isoformat() + "Z"
        }
        
        feedback_ref = rtdb.reference(f'{article_path}/examiner_feedbacks/{user_id}')
        feedback_ref.set(feedback_data)
        
        return jsonify({"success": True, "message": "Feedback submitted", "feedback": feedback_data}), 200
        
    except Exception as e:
        app.logger.exception("Error submitting feedback: %s", e)
        return jsonify({"error": "Failed to submit feedback"}), 500


@app.route('/api/article/<article_id>/feedbacks', methods=['GET'])
def get_article_feedbacks(article_id):
    """Get feedbacks: Owner sees ALL, Examiner sees ONLY theirs"""
    try:
        if not session.get('idToken'):
            return jsonify({"error": "Unauthorized"}), 401
        
        user_id = session.get('uid')
        user_doc = db.collection('users').document(user_id).get()
        
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404
        
        user_role = user_doc.to_dict().get('role')
        
        # Find article
        uploaded_news_ref = rtdb.reference('datasets/uploaded_news')
        article_found = False
        article_path = None
        
        all_datasets = uploaded_news_ref.get() or {}
        for dataset_id, dataset_content in all_datasets.items():
            if isinstance(dataset_content, dict):
                for push_id, article_data in dataset_content.items():
                    if push_id == article_id:
                        article_found = True
                        article_path = f'datasets/uploaded_news/{dataset_id}/{article_id}'
                        break
            if article_found:
                break
        
        if not article_found:
            return jsonify({"error": "Article not found"}), 404
        
        feedbacks_ref = rtdb.reference(f'{article_path}/examiner_feedbacks')
        all_feedbacks = feedbacks_ref.get() or {}
        
        # Filter based on role
        if user_role == 'examiner':
            if user_id in all_feedbacks:
                return jsonify({"feedbacks": {user_id: all_feedbacks[user_id]}, "has_feedback": True}), 200
            else:
                return jsonify({"feedbacks": {}, "has_feedback": False}), 200
        
        elif user_role == 'project_owner':
            return jsonify({"feedbacks": all_feedbacks, "total_count": len(all_feedbacks)}), 200
        
        else:
            return jsonify({"error": "Invalid role"}), 403
        
    except Exception as e:
        app.logger.exception("Error getting feedbacks: %s", e)
        return jsonify({"error": "Failed to get feedbacks"}), 500
    
    
# ═══════════════════════════════════════════════════════════════
# 📝 FeedBack Task Routes
# ═══════════════════════════════════════════════════════════════

@app.route("/task/<task_id>/feedback")
def feedback_task_page(task_id):
    """صفحة FeedBack للـ Examiner"""
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    uid = session.get("uid")
    
    # نجيب معلومات Task
    task_doc = db.collection("tasks").document(task_id).get()
    if not task_doc.exists:
        abort(404)
    
    task_data = task_doc.to_dict()
    
    # نتأكد إنه FeedBack Task
    if task_data.get("task_type") != "labeling":
        abort(403)
    
    # نتأكد إن الـ Examiner مسند له
    if uid not in task_data.get("examiner_ids", []):
        abort(403)
    
    # نجيب اسم المستخدم
    user_name = _reviewer_display_name(uid)
    
    project_id = task_data.get("project_ID")
    
    project_id = task_data.get("project_ID")

    # ✅ إذا كان المشروع Generated Conversation: استخدم نفس صفحة results.con لكن بوضع feedback
    proj_doc = db.collection("projects").document(project_id).get()
    proj_data = proj_doc.to_dict() if proj_doc.exists else {}

    category = (proj_data.get("category") or "").strip().lower()
    is_generated_conversation = (
        category in ["conversation", "conversations", "chat", "chats"]
        and bool(proj_data.get("generated_from_scratch", False))
    )

    if is_generated_conversation:
        return redirect(url_for("results_con", projectId=project_id, taskId=task_id, mode="feedback"))

    is_uploaded_conversation = (
        category in ["conversation", "conversations", "chat", "chats"]
        and not bool(proj_data.get("generated_from_scratch", False))
    )

    if is_uploaded_conversation:
        return redirect(url_for(
            "results_con",
            projectId=project_id,
            taskId=task_id,
            mode="feedback",
            source="uploaded"
        ))

    
    # ✅ Article flow كما هو (بدون تغيير سلوكه)
    dataset_id = proj_data.get("dataset_id", "")

    
    return render_template(
        "feedbacktask.html",
        user_name=user_name,
        task_id=task_id,
        project_id=project_id,
        dataset_id=dataset_id,  # ✅ نمرره للـ Template
        task_name=task_data.get("task_name", "Feedback Task")
    )
    

@app.route("/api/task/<task_id>/articles", methods=["GET"])
def api_get_task_articles(task_id):
    try:
        if not session.get("idToken"):
            return jsonify({"error": "Unauthorized"}), 401

        uid = session.get("uid")

        task_doc = db.collection("tasks").document(task_id).get()
        if not task_doc.exists:
            return jsonify({"error": "Task not found"}), 404

        task_data = task_doc.to_dict()
        project_id = task_data.get("project_ID")

        app.logger.info("=== DEBUG === project_id: %s", project_id)
        all_tasks = list(db.collection("tasks").where("project_ID", "==", project_id).stream())
        for t in all_tasks:
            d = t.to_dict()
            app.logger.info("=== DEBUG === task: type=%s status=%s", d.get("task_type"), d.get("status"))

        # نشيك إذا في model_selection task مكتمل
        model_selection_tasks = list(
            db.collection("tasks")
            .where("project_ID", "==", project_id)
            .where("task_type", "==", "model_selection")
            .where("status", "==", "completed")
            .limit(1)
            .stream()
        )

        if not model_selection_tasks:
            return jsonify({
                "waiting": True,
                "message": "Waiting for model selection task to be completed"
            }), 200

        ms_task = model_selection_tasks[0].to_dict()
        selected_model = ms_task.get("selected_model", "logistic")

        app.logger.info("=== DEBUG === selected_model: %s", selected_model)

        # نجيب dataset_id
        proj_doc = db.collection("projects").document(project_id).get()
        if not proj_doc.exists:
            return jsonify({"error": "Project not found"}), 404

        dataset_id = proj_doc.to_dict().get("dataset_id")
        if not dataset_id:
            return jsonify({"error": "No dataset found"}), 404

        # نجيب النتائج من RTDB
        safe_pid = project_id.replace(".", "_").replace("#", "_").replace("$", "_").replace("[", "_").replace("]", "_")
        results_ref = rtdb.reference(f"analysis_results/{safe_pid}/{selected_model}")
        results_data = results_ref.get()

        app.logger.info("=== DEBUG === results_data exists: %s", results_data is not None)

        if not results_data:
            return jsonify({
                "waiting": True,
                "message": "Analysis not ready yet, please wait"
            }), 200

        details = results_data.get("details", [])
        summary = results_data.get("summary", {})

        # نجيب الـ feedbacks الموجودة
        dataset_rows = rtdb.reference(f"datasets/uploaded_news/{dataset_id}").get() or {}
        detail_map = {
            _safe_str(item.get("article_id") or item.get("id")): item
            for item in details
            if isinstance(item, dict)
        }

        articles_with_feedback = []
        for article in details:
            article_id = article.get("article_id")
            source = dataset_rows.get(article_id, {}) if isinstance(dataset_rows, dict) else {}
            feedback = source.get("feedback") if isinstance(source, dict) else None
            articles_with_feedback.append({
                "article_id": article_id,
                "title": article.get("title", ""),
                "content": article.get("content", ""),
                "prediction": article.get("prediction", ""),
                "human_percentage": article.get("human_percentage", 0),
                "ai_percentage": article.get("ai_percentage", 0),
                "confidence": article.get("confidence"),
                "uncertainty": article.get("uncertainty"),
                "chunks": article.get("chunks", []),
                "has_feedback": feedback is not None,
                "feedback": feedback
            })

        active_learning_enabled = _uses_frozen_active_learning(selected_model)
        news_candidates = _news_active_learning_candidates(project_id, selected_model, details)
        active_learning_total = len(news_candidates) if active_learning_enabled else len(articles_with_feedback)
        active_learning_limit = _active_learning_limit(active_learning_total) if active_learning_enabled else active_learning_total

        if active_learning_enabled:
            model_version = _active_learning_model_version(project_id, selected_model)
            frozen_selection = _freeze_active_learning_targets_if_missing(
                project_id,
                "news",
                selected_model,
                model_version,
                news_candidates,
                task_data.get("examiner_ids") or []
            )
            chunk_targets = [
                target for target in _frozen_target_values(frozen_selection)
                if _safe_str(target.get("target_unit")) == "chunk"
            ]
            if chunk_targets:
                chunk_rows = []
                for target in sorted(chunk_targets, key=lambda item: int(item.get("selection_rank") or 0)):
                    article_id = _safe_str(target.get("article_id") or target.get("parent_article_id"))
                    detail = detail_map.get(article_id, {})
                    source = dataset_rows.get(article_id, {}) if isinstance(dataset_rows, dict) else {}
                    chunk_index = int(target.get("chunk_index") or 0)
                    selected_source_chunk = _news_chunk_by_index(detail, chunk_index)
                    chunk = selected_source_chunk or target
                    sample_id = _safe_str(target.get("sample_id")) or _make_active_learning_sample_id("news", article_id=article_id, chunk_index=chunk_index)
                    feedback = _news_feedback_by_sample(
                        project_id,
                        model_version,
                        sample_id,
                        source=source,
                        chunk_index=chunk_index,
                        uid=uid
                    )
                    feedback_status = _feedback_lifecycle_status(feedback)
                    target_status = feedback_status if feedback_status != "pending" else (_safe_str(target.get("target_status")) or "pending")
                    target_reviewed = target_status != "pending" or _target_feedback_started(target) or feedback is not None
                    feedback_locked = _feedback_is_final(feedback) or _target_feedback_finalized(target) or _labeling_task_completed(task_data)
                    reviewed_by = _safe_str(
                        (feedback or {}).get("reviewed_by")
                        or (feedback or {}).get("examiner_uid")
                        or target.get("reviewed_by")
                    )
                    chunks = []
                    for pos, source_chunk_item in enumerate(detail.get("chunks") or [], start=1):
                        if not isinstance(source_chunk_item, dict):
                            continue
                        current_index = int(source_chunk_item.get("chunk_index") or pos)
                        current_sample_id = _make_active_learning_sample_id("news", article_id=article_id, chunk_index=current_index)
                        current_feedback = _news_feedback_by_sample(
                            project_id,
                            model_version,
                            current_sample_id,
                            source=source,
                            chunk_index=current_index,
                            uid=uid
                        )
                        chunks.append({
                            **source_chunk_item,
                            "prediction": _normalized_visible_label(source_chunk_item.get("prediction")),
                            "uncertainty": _chunk_uncertainty_percent(source_chunk_item),
                            "active_learning_selected": current_index == chunk_index,
                            "has_feedback": bool(current_feedback)
                        })
                    selected_uncertainty = _chunk_uncertainty_percent(chunk)
                    reviewed_label = _normalized_visible_label(
                        (feedback or {}).get("corrected_label_text")
                        or
                        (feedback or {}).get("corrected_label")
                        or (feedback or {}).get("label")
                        or target.get("corrected_label")
                    )
                    reviewer_uid = reviewed_by or _safe_str((feedback or {}).get("reviewed_by") or (feedback or {}).get("examiner_uid"))
                    reviewer_name = _reviewer_display_name(
                        reviewer_uid,
                        (feedback or {}).get("reviewed_by_name")
                        or (feedback or {}).get("examiner_name")
                        or target.get("reviewed_by_name")
                    )
                    chunk_rows.append({
                        "article_id": article_id,
                        "title": detail.get("title", target.get("title", "")),
                        "content": detail.get("content", ""),
                        "prediction": _normalized_visible_label(detail.get("prediction", target.get("parent_article_prediction"))),
                        "prediction_int": _first_present(detail.get("prediction_int"), target.get("parent_article_prediction_int")),
                        "human_percentage": detail.get("human_percentage", 0),
                        "ai_percentage": detail.get("ai_percentage", 0),
                        "confidence": chunk.get("confidence", target.get("confidence")),
                        "uncertainty": selected_uncertainty,
                        "chunks": chunks,
                        "selected_chunk": {
                            **chunk,
                            "prediction": _normalized_visible_label(chunk.get("prediction")),
                            "uncertainty": selected_uncertainty,
                            "chunk_index": chunk_index,
                            "chunk_text": chunk.get("chunk_text") or chunk.get("text") or target.get("chunk_text") or target.get("text"),
                            "selection_rank": target.get("selection_rank"),
                            "sample_id": target.get("sample_id")
                        },
                        "selected_chunk_uncertainty": selected_uncertainty,
                        "target_unit": "chunk",
                        "chunk_index": chunk_index,
                        "sample_id": sample_id,
                        "active_learning_selected": True,
                        "target_status": target_status if target_reviewed else "pending",
                        "review_state": target_status if target_reviewed else "pending",
                        "reviewed_by": reviewer_uid or None,
                        "reviewed_by_name": reviewer_name,
                        "reviewed_label": reviewed_label,
                        "corrected_label": reviewed_label,
                        "feedback_explanation": (feedback or {}).get("feedback_explanation") or (feedback or {}).get("explanation"),
                        "submitted_at": (feedback or {}).get("submitted_at") or (feedback or {}).get("reviewed_at") or target.get("reviewed_at"),
                        "is_my_feedback": bool(reviewer_uid and reviewer_uid == uid),
                        "has_feedback": target_reviewed,
                        "feedback": feedback,
                        "my_feedback": feedback if reviewer_uid == uid else None,
                        "can_edit_feedback": bool(feedback and reviewer_uid == uid and not feedback_locked),
                        "feedback_locked": feedback_locked
                    })
                articles_with_feedback = chunk_rows
            else:
                selected_ids = {
                    _safe_str(target.get("article_id") or target.get("source_id"))
                    for target in _frozen_target_values(frozen_selection)
                }
                for item in articles_with_feedback:
                    item["active_learning_selected"] = _safe_str(item.get("article_id")) in selected_ids
                articles_with_feedback = [item for item in articles_with_feedback if item.get("active_learning_selected")]
            articles_with_feedback.sort(key=lambda item: (_active_learning_sort_value(item), item.get("article_id", ""), int(item.get("chunk_index") or 0)))
            active_learning_limit = len(articles_with_feedback)
        return jsonify({
            "articles": articles_with_feedback,
            "summary": summary,
            "selected_model": selected_model,
            "active_learning": {
                "enabled": active_learning_enabled,
                "review_targets_enabled": active_learning_enabled,
                "active_learning_enabled": active_learning_enabled,
                "retraining_supported": _active_learning_retraining_supported(selected_model),
                "percent": ACTIVE_LEARNING_PERCENT,
                "max_samples": ACTIVE_LEARNING_MAX_SAMPLES,
                "selected": len(articles_with_feedback),
                "source_total": active_learning_total,
                "model_version": _active_learning_model_version(project_id, selected_model)
            },
            "total": len(articles_with_feedback)
        }), 200

    except Exception as e:
        app.logger.exception("Unexpected error in api_get_task_articles: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/article/<article_id>/submit_feedback", methods=["POST"])
def api_submit_article_feedback(article_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401
    uid = session.get("uid")

    data = request.get_json() or {}
    agreed_with_model = bool(data.get("agreed_with_model", False))
    dataset_id = data.get("dataset_id")
    chunk_index = data.get("chunk_index")
    target_unit = "chunk" if chunk_index is not None and str(chunk_index).strip() != "" else "article"
    if target_unit == "chunk":
        try:
            chunk_index = int(chunk_index)
        except Exception:
            return jsonify({"error": "Invalid chunk_index"}), 400

    if not dataset_id:
        return jsonify({"error": "Dataset ID is required"}), 400

    if not agreed_with_model:
        label = data.get("label")
        explanation = data.get("explanation", "").strip()
        if not label or label not in ["Human", "AI"]:
            return jsonify({"error": "Invalid label"}), 400

    try:
        if target_unit == "chunk":
            feedback_ref = rtdb.reference(
                f"datasets/uploaded_news/{dataset_id}/{article_id}/chunk_feedback/{chunk_index}"
            )
        else:
            feedback_ref = rtdb.reference(
                f"datasets/uploaded_news/{dataset_id}/{article_id}/feedback"
            )

        existing_feedbacks = feedback_ref.get() or {}
        if not isinstance(existing_feedbacks, dict):
            existing_feedbacks = {}

        try:
            project_id, project = _project_by_dataset_id(dataset_id)
        except Exception:
            project_id, project = None, None

        task_completed = False
        if project_id:
            for doc in db.collection("tasks").where("project_ID", "==", project_id).where("task_type", "==", "labeling").stream():
                tdata = doc.to_dict() or {}
                conversation_type = _safe_str(tdata.get("conversation_type")).strip().lower()
                if not conversation_type and _labeling_task_completed(tdata):
                    task_completed = True
                    break
        if task_completed:
            return jsonify({"error": "Feedback is locked because this labeling task is completed"}), 423

        existing_owner_uid = _feedback_owner_uid(existing_feedbacks)
        existing_user_feedback = _feedback_record_for_uid(existing_feedbacks, uid)
        if existing_owner_uid and existing_owner_uid != uid:
            return jsonify({"error": "Feedback already exists for this target and can only be edited by the original reviewer"}), 409

        examiner_name = _feedback_examiner_name(uid)

        try:
            if project_id:
                detection_snapshot = _find_detection_snapshot_for_feedback(
                    "news",
                    project_id,
                    project=project,
                    dataset_id=dataset_id,
                    article_id=article_id,
                    chunk_index=chunk_index if target_unit == "chunk" else None
                )
            else:
                detection_snapshot = {}
        except Exception:
            project_id, project, detection_snapshot = None, None, {}

        if project_id and _version_is_closed(project_id, detection_snapshot.get("version_id") or detection_snapshot.get("model_version") or "v1"):
            return jsonify({"error": "Feedback is locked because this evaluation version is closed"}), 423

        if project_id:
            sample_id = _make_active_learning_sample_id(
                "news",
                article_id=article_id,
                chunk_index=chunk_index if target_unit == "chunk" else None
            )
            try:
                _ensure_feedback_can_be_saved(project_id, detection_snapshot.get("version_id") or detection_snapshot.get("model_version") or "v1", sample_id, uid)
            except ValueError as e:
                return jsonify({"error": str(e)}), 423
        predicted_label = _label_text_only(detection_snapshot.get("model_prediction_label") or detection_snapshot.get("model_prediction"))
        if not agreed_with_model and data.get("label") != predicted_label and not data.get("explanation", "").strip():
            return jsonify({"error": "Explanation is required when correcting the model"}), 400

        now_iso = datetime.utcnow().isoformat() + "Z"
        feedback_data = {
            "examiner_uid": uid,
            "examiner_name": examiner_name,
            "agreed_with_model": agreed_with_model,
            "label": None if agreed_with_model else data.get("label"),
            "explanation": "" if agreed_with_model else data.get("explanation", "").strip(),
            "submitted_at": now_iso,
            "article_id": article_id,
            "target_unit": target_unit,
            "chunk_index": chunk_index if target_unit == "chunk" else None,
            "chunk_id": detection_snapshot.get("chunk_id"),
            "chunk_text": detection_snapshot.get("chunk_text"),
            "model_prediction": detection_snapshot.get("model_prediction_label") or detection_snapshot.get("model_prediction"),
            "corrected_label": detection_snapshot.get("model_prediction_label") if agreed_with_model else data.get("label"),
            "feedback_explanation": "" if agreed_with_model else data.get("explanation", "").strip(),
            "confidence": detection_snapshot.get("confidence"),
            "uncertainty": detection_snapshot.get("uncertainty"),
            "status": "draft_saved",
            "lifecycle_status": "draft_saved",
            "locked": False
        }
        feedback_data, is_edit = _prepare_feedback_payload(existing_user_feedback, feedback_data, uid, examiner_name, now_iso)

        if target_unit == "chunk":
            feedback_ref.child(uid).set(feedback_data)
        else:
            feedback_ref.set(feedback_data)

        try:
            if project_id:
                sample_id = _make_active_learning_sample_id("news", article_id=article_id, chunk_index=chunk_index if target_unit == "chunk" else None)
                _write_active_learning_feedback(project_id, sample_id, detection_snapshot, feedback_data)
                selected_model_key = detection_snapshot.get("selected_model_key") or "logistic"
                if _uses_frozen_active_learning(selected_model_key):
                    model_version = _active_learning_model_version(project_id, selected_model_key)
                    _mark_frozen_active_learning_target_reviewed(
                        project_id,
                        model_version,
                        sample_id,
                        uid,
                        feedback_data.get("submitted_at"),
                        {"article_id": article_id, "chunk_index": chunk_index} if target_unit == "chunk" else {"article_id": article_id}
                    )
                    _update_news_labeling_status_from_frozen_targets(project_id, model_version)
        except Exception as normalized_error:
            app.logger.exception("Failed to write normalized news feedback: %s", normalized_error)

        return jsonify({"message": _feedback_write_message(is_edit), "updated": is_edit}), 200

    except Exception as e:
        app.logger.exception("Failed to submit feedback: %s", e)
        return jsonify({"error": "Failed to submit feedback"}), 500
    
# =========================
# [4] صفحة نتائج المحادثات
# =========================
@app.route("/results_con", endpoint="results_con")
def show_results():
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    project_id = request.args.get("projectId")
    task_id = request.args.get("taskId")
    source = (request.args.get("source") or "").strip().lower()
    mode = (request.args.get("mode") or "model_selection").strip().lower()  # ✅ mode: model_selection | feedback

    user_doc = get_current_user_doc()
    user_name = get_user_full_name(user_doc) if user_doc else "Examiner"
    if task_id and mode == "model_selection":
        task_doc = db.collection("tasks").document(task_id).get()
        if not task_doc.exists:
            abort(404)
        task_data = task_doc.to_dict() or {}
        examiner_ids = task_data.get("examiner_ids") or []
        if task_data.get("task_type") != "model_selection" or len(examiner_ids) != 1 or session.get("uid") not in examiner_ids:
            abort(403)

    return render_template(
        "results.con.html",
        user_name=user_name,
        user_uid=session.get("uid"),
        user_role="Examiner",
        project_id=project_id,
        task_id=task_id,
        mode=mode,
        source=source
    )



# =========================
# [5] Helper functions لتحليل المحادثات
# =========================
def _get_conversation_messages(task_id, conversation_type):
    ref = rtdb.reference(f"llm_conversations/{task_id}/messages") if conversation_type == "human-ai" \
        else rtdb.reference(f"hh_conversations/{task_id}/messages")

    raw = ref.get() or {}
    rows = list(raw.values()) if isinstance(raw, dict) else (raw or [])
    rows.sort(key=lambda m: m.get("created_at", ""))

    speaker_side = {}
    sides = ["left", "right"]
    messages = []

    for m in rows:
        if not isinstance(m, dict):
            continue
        text = (m.get("message") or "").strip()
        if not text:
            continue

        sender_type = (m.get("sender_type") or "").lower()
        ex_id = m.get("examiner_id") or m.get("sender_id")

        if conversation_type == "human-ai":
            side = "right" if sender_type in ("llm", "ai", "assistant", "model") else "left"
        else:
            if not ex_id:
                continue
            if ex_id not in speaker_side:
                speaker_side[ex_id] = sides[len(speaker_side) % 2]
            side = speaker_side[ex_id]

        messages.append({"text": text, "side": side, "sender_type": sender_type})

    return messages


def _compute_turns_count(messages, conversation_type):
    if conversation_type == "human-ai":
        return sum(1 for m in messages if m.get("sender_type") not in ("llm", "ai", "assistant", "model"))
    seq = [m.get("side") for m in messages if m.get("side")]
    runs = []
    for s in seq:
        if not runs or s != runs[-1]:
            runs.append(s)
    return len(runs) // 2


def _gt_label_from_sender(sender_type, conversation_type):
    st = (sender_type or "").lower()
    if conversation_type == "human-ai":
        return "AI" if st in ("llm", "ai", "assistant", "model") else "Human"
    return "Human"


def _gt_int_from_sender(sender_type, conversation_type):
    return _normalize_binary_label(_gt_label_from_sender(sender_type, conversation_type))


def _sender_label(sender_type, conversation_type):
    st = (sender_type or "").lower()
    if conversation_type == "human-ai":
        return "Machine" if st in ("llm", "ai", "assistant", "model") else "Human"
    return "Human"

# =========================
# [6] API تشغيل تحليل المحادثات (أهم جزء)
# =========================
@app.route("/api/run_analysis_project/<project_id>", methods=["POST"])
def api_run_analysis_project(project_id):
    try:
        selected_model = _normalize_conversation_model_key(request.args.get("model") or "logreg")
        model_key = _normalize_conversation_model_key(selected_model, for_results=True)
        selected_model_name = _conversation_model_name(selected_model)
        task_id_from_query = (request.args.get("task_id") or request.args.get("taskId") or "").strip()

        if not session.get("idToken"):
            return jsonify({"error": "Unauthorized"}), 401
        uid = session.get("uid")

        proj_doc = db.collection("projects").document(project_id).get()
        if not proj_doc.exists:
            return jsonify({"error": "Project not found"}), 404

        ms_task_ref, ms_task_data, guard_error = _model_selection_task_guard(
            task_id_from_query,
            project_id=project_id,
            reject_completed=False
        )
        if guard_error:
            return guard_error
        task_finalized = bool(ms_task_data.get("selected_model") or _safe_str(ms_task_data.get("status")).lower() == "completed")

        tasks = db.collection("tasks").where("project_ID", "==", project_id).stream()
        run_id = uuid.uuid4().hex[:12]
        base_ref = _generated_conversation_base_ref(project_id, model_key)
        out_ref = base_ref.child("runs").child(run_id)

        analyzed_conversations = 0
        analyzed_turns = 0

        for t in tasks:
            d = t.to_dict() or {}
            task_id = d.get("task_ID") or t.id
            if not task_id:
                continue

            conv_type = d.get("conversation_type", "human-human")
            msgs = _get_conversation_messages(task_id, conv_type)
            if not msgs:
                continue

            analyzed_conversations += 1
            analyzed_turns += len(msgs)

            texts = [m["text"] for m in msgs]
            prev_texts = [""] + texts[:-1]

            if selected_model == CONV_RNN_KEY:
                seq = conv_rnn_tokenizer.texts_to_sequences(texts)
                x = pad_sequences(seq, maxlen=300, padding="post", truncating="post")
                raw_pred = conv_rnn_model.predict(x, verbose=0)
                p_pos = raw_pred[:, 1].astype(float) if (raw_pred.ndim == 2 and raw_pred.shape[1] == 2) else raw_pred.reshape(-1).astype(float)
                p_ai = p_pos if CONV_RNN_AI_CLASS_IS_ONE else (1.0 - p_pos)
                p_ai = np.clip(p_ai, 0.0, 1.0)
                labels = ["AI" if p >= 0.5 else "Human" for p in p_ai]
                probs = [[float(1.0 - p), float(p)] for p in p_ai]
            else:
                df_in = pd.DataFrame({"text": texts, "prev_text": prev_texts})
                preds = conv_logreg_model.predict(df_in)
                labels = ["Human" if p == 0 else "AI" for p in preds]
                try:
                    probs = conv_logreg_model.predict_proba(df_in)
                except Exception:
                    probs = None

            task_ref = out_ref.child(task_id)
            task_ref.child("meta").set({
                "task_id": task_id,
                "task_name": d.get("task_name", "Conversation"),
                "conversation_type": conv_type,
                "selected_model": selected_model,
                "selected_model_name": selected_model_name
            })


            turns_ref = task_ref.child("turns")
            for i, (m, label) in enumerate(zip(msgs, labels), start=1):
                p_machine = None
                conf = None
                uncertainty = None
                prediction_int = _normalize_binary_label(label)
                ground_truth = _gt_int_from_sender(m.get("sender_type"), conv_type)
                if probs is not None:
                    if selected_model == CONV_RNN_KEY:
                        p_machine = float(probs[i - 1][1])
                    else:
                        p_machine = _machine_probability_from_proba(conv_logreg_model, probs[i - 1])
                    conf, uncertainty = _confidence_uncertainty_from_prob(p_machine)
                turns_ref.push({
                    "turn_index": i,
                    "text": m["text"],
                    "prev_text": prev_texts[i - 1],
                    "prediction": label,
                    "prediction_int": prediction_int,
                    "gt": _gt_label_from_sender(m.get("sender_type"), conv_type),
                    "ground_truth": ground_truth,
                    "ground_truth_label": _label_text_from_int(ground_truth),
                    "sender": _sender_label(m.get("sender_type"), conv_type),
                    "p_machine": p_machine,
                    "confidence": conf,
                    "uncertainty": uncertainty,
                })

        if analyzed_conversations == 0:
            return jsonify({
                "error": "No conversation messages found for this project. Complete a conversation task first."
            }), 400

        selected_at = datetime.utcnow().isoformat() + "Z"
        if not task_finalized:
            base_ref.child("latest_run_id").set(run_id)
            base_ref.child("latest_model_key").set(model_key)
            base_ref.child("latest_analyzed_at").set(selected_at)

        return jsonify({
            "success": True,
            "model": selected_model,
            "run_id": run_id,
            "preview_only": task_finalized,
            "ran_at": datetime.utcnow().isoformat() + "Z",
            "analyzed_conversations": analyzed_conversations,
            "analyzed_turns": analyzed_turns
        }), 200

    except Exception as e:
        app.logger.exception("api_run_analysis_project failed")
        return jsonify({"error": str(e)}), 500



# =========================
# [7] API قراءة نتائج تحليل المحادثات
# =========================
@app.route("/api/analysis_project/<project_id>", methods=["GET"])
def api_analysis_project(project_id):
    selected_model = _normalize_conversation_model_key(request.args.get("model") or "logreg")
    model_key = _normalize_conversation_model_key(selected_model, for_results=True)
    run_id = _safe_str(request.args.get("run_id"))

    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    raw = _generated_conversation_results_payload(project_id, model_key, run_id=run_id) or {}
    results = []

    y_true = []
    y_pred = []

    for task_id, node in raw.items():
        meta = node.get("meta", {})
        turns_raw = node.get("turns", {}) or {}
        turns = list(turns_raw.values()) if isinstance(turns_raw, dict) else turns_raw
        turns.sort(key=lambda x: x.get("turn_index", 0))

        for t in turns:
            gt = _first_present(t.get("ground_truth"), t.get("gt"))
            pr = _first_present(t.get("prediction_int"), t.get("prediction"))
            if _normalize_binary_label(gt) is not None and _normalize_binary_label(pr) is not None:
                y_true.append(gt)
                y_pred.append(pr)

        results.append({
            "task_id": task_id,
            "task_name": meta.get("task_name", "Conversation"),
            "selected_model_name": meta.get("selected_model_name"),
            "turns": turns
        })

    metrics = _compute_standard_metrics(y_true, y_pred)
    values = _owner_confusion_values(metrics.get("confusion_matrix")) if metrics.get("available") else None
    confusion_matrix = {
        "true_negative": values["tn"],
        "false_positive": values["fp"],
        "false_negative": values["fn"],
        "true_positive": values["tp"],
    } if values else None

    return jsonify({
        "count": len(results),
        "results": results,
        "confusion_matrix": confusion_matrix,
        "metrics": metrics
    }), 200


@app.route("/api/conversation/select_model_task", methods=["POST"])
def api_conversation_select_model_task():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")
    data = request.get_json() or {}

    project_id = (data.get("project_id") or "").strip()
    task_id = (data.get("task_id") or "").strip()
    selected_model = _normalize_conversation_model_key(data.get("model"))

    if selected_model not in ("logreg", "rnn"):
        return jsonify({"error": "Invalid model"}), 400
    if not project_id or not task_id:
        return jsonify({"error": "project_id and task_id are required"}), 400

    task_ref, task_data, guard_error = _model_selection_task_guard(task_id, project_id=project_id, reject_completed=True)
    if guard_error:
        return guard_error

    now_iso = datetime.utcnow().isoformat() + "Z"
    model_name = _conversation_model_name(selected_model)

    project = get_project_basic_info(project_id) or {}
    result_type = detect_project_result_type(project)
    selected_key_for_results = _normalize_conversation_model_key(selected_model, for_results=True)
    model_version = _active_learning_model_version(project_id, selected_key_for_results)
    analysis_run_id = None
    freeze_candidates = []

    if result_type == "uploaded_conversation":
        run_id, run_ref, _ = _uploaded_conversation_run(project_id, selected_key_for_results)
        if not run_ref:
            return jsonify({"error": "Analysis results are not ready for the selected model"}), 400
        analysis_run_id = run_id
        freeze_candidates = _uploaded_conversation_active_learning_candidates(project_id, selected_key_for_results, run_id, run_ref)
    elif result_type == "generated_conversation":
        analysis_run_id = _safe_str(_generated_conversation_base_ref(project_id, selected_key_for_results).child("latest_run_id").get())
        raw = _generated_conversation_results_payload(project_id, selected_key_for_results)
        if not isinstance(raw, dict) or not raw:
            return jsonify({"error": "Analysis results are not ready for the selected model"}), 400
        freeze_candidates = _generated_conversation_active_learning_candidates(project_id, selected_key_for_results, raw)

    task_ref.update({
        "selected_model": selected_model,
        "selected_model_name": model_name,
        "selected_by": uid,
        "selected_at": now_iso,
        "analysis_run_id": analysis_run_id,
        "status": "completed"
    })

    _freeze_active_learning_targets_if_missing(
        project_id,
        result_type,
        selected_key_for_results,
        model_version,
        freeze_candidates,
        task_data.get("examiner_ids") or []
    )
    _create_detection_version_snapshot(
        project_id,
        task_id,
        selected_key_for_results,
        model_name,
        uid,
        analysis_run_id=analysis_run_id
    )

    return jsonify({
        "message": "Conversation model selected and saved",
        "project_id": project_id,
        "task_id": task_id,
        "selected_model_name": model_name
    }), 200


@app.route("/api/conversation/selected_model_task", methods=["GET"])
def api_conversation_selected_model_task():
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    project_id = (request.args.get("project_id") or "").strip()
    task_id = (request.args.get("task_id") or "").strip()

    if not project_id or not task_id:
        return jsonify({"error": "project_id and task_id are required"}), 400

    task_doc = db.collection("tasks").document(task_id).get()
    if not task_doc.exists:
        return jsonify({"selected": False}), 200

    task_data = task_doc.to_dict() or {}
    if task_data.get("project_ID") != project_id:
        return jsonify({"selected": False}), 200

    model_key = _normalize_conversation_model_key(task_data.get("selected_model")) if task_data.get("selected_model") else ""
    model_name = (task_data.get("selected_model_name") or "").strip()

    if model_key in ("rnn", "logreg") or model_name:
        if not model_name:
            model_name = _conversation_model_name(model_key)

        return jsonify({
            "selected": True,
            "selected_model": model_key,
            "selected_model_name": model_name,
            "selected_at": task_data.get("selected_at"),
            "task_id": task_id,
            "project_id": project_id
        }), 200

    return jsonify({"selected": False}), 200




# =========================
# [8] Conversation Feedback APIs (Generated Conversation)
# =========================
def _pick_conversation_model_for_project(project_id):
    """
    يحدد الموديل المختار للمشروع (logreg أو rnn) من model_selection task المكتمل.
    """
    ms_tasks = list(
        db.collection("tasks")
        .where("project_ID", "==", project_id)
        .where("task_type", "==", "model_selection")
        .where("status", "==", "completed")
        .stream()
    )

    if not ms_tasks:
        return None, None, None

    # نأخذ آخر واحدة حسب selected_at إن وجدت
    ms_tasks.sort(key=lambda d: (d.to_dict() or {}).get("selected_at", ""), reverse=True)

    picked = None
    for doc in ms_tasks:
        td = doc.to_dict() or {}
        raw = (td.get("selected_model") or "").strip().lower()
        name = (td.get("selected_model_name") or "").strip().lower()

        if raw in ("rnn",):
            picked = "rnn"
            break
        if raw in ("logreg", "logistic", "tfidf_logreg"):
            picked = "logreg"
            break
        if "rnn" in name:
            picked = "rnn"
            break
        if "logistic" in name:
            picked = "logreg"
            break

    if not picked:
        return None, None, None

    model_key = CONV_RNN_KEY if picked == "rnn" else CONV_LOGREG_KEY
    model_label = "RNN" if picked == "rnn" else "Logistic Regression"
    return picked, model_key, model_label


@app.route("/api/task/<task_id>/conversation_feedback_list", methods=["GET"])
def api_conversation_feedback_list(task_id):
    """
    قائمة المحادثات للفيدباك (لـ labeling task) + progress + sorting metadata
    """
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")

    task_doc = db.collection("tasks").document(task_id).get()
    if not task_doc.exists:
        return jsonify({"error": "Task not found"}), 404

    task_data = task_doc.to_dict() or {}
    if task_data.get("task_type") != "labeling":
        return jsonify({"error": "Task is not labeling"}), 400
    if uid not in (task_data.get("examiner_ids") or []):
        return jsonify({"error": "Forbidden"}), 403

    project_id = task_data.get("project_ID")
    if not project_id:
        return jsonify({"error": "Missing project_ID"}), 400

    selected_model, model_key, model_label = _pick_conversation_model_for_project(project_id)
    if not selected_model:
        return jsonify({
            "waiting": True,
            "message": "Waiting for model selection task to be completed"
        }), 200

    # نجيب نتائج التحليل من نفس موديل الاختيار
    raw = _generated_conversation_results_payload(project_id, model_key) or {}
    if not isinstance(raw, dict) or not raw:
        return jsonify({
            "waiting": True,
            "message": "Analysis not ready yet, please wait"
        }), 200

    # ترتيب افتراضي حسب ترتيب tasks بالمشروع (created_at)
    conv_tasks = []
    for t in db.collection("tasks").where("project_ID", "==", project_id).stream():
        td = t.to_dict() or {}
        ctype = (td.get("conversation_type") or "").strip().lower()
        if ctype in ("human-ai", "human-human"):
            conv_tasks.append({
                "task_id": td.get("task_ID") or t.id,
                "created_at": td.get("created_at", "")
            })
    conv_tasks.sort(key=lambda x: x["created_at"])
    order_map = {row["task_id"]: idx for idx, row in enumerate(conv_tasks)}

    # cache للأسماء
    name_cache = {}

    def _name_for(user_id):
        if not user_id:
            return "Examiner"
        if user_id in name_cache:
            return name_cache[user_id]
        udoc = db.collection("users").document(user_id).get()
        if not udoc.exists:
            name_cache[user_id] = "Examiner"
            return "Examiner"
        u = udoc.to_dict() or {}
        p = u.get("profile", {})
        full = f"{p.get('firstName','')} {p.get('lastName','')}".strip() or "Examiner"
        name_cache[user_id] = full
        return full

    items = []

    for node_key, node_val in raw.items():
        node = node_val or {}
        meta = node.get("meta", {}) or {}
        conversation_id = meta.get("task_id") or node_key

        turns_raw = node.get("turns", {}) or {}
        turns = list(turns_raw.values()) if isinstance(turns_raw, dict) else (turns_raw if isinstance(turns_raw, list) else [])
        turns.sort(key=lambda x: int(x.get("turn_index", 0) or 0))

        total = len(turns)
        ai_count = 0
        human_count = 0
        confs = []
        uncertainties = []
        clean_turns = []
        conv_feedback_users_map = {}

        turn_feedbacks_root = node.get("turn_feedbacks") or {}



        reviewed_in_conv = 0

        for t in turns:
            pred = str(t.get("prediction", "")).strip()
            if pred == "AI":
                ai_count += 1
            else:
                human_count += 1

            c_raw = t.get("confidence")
            c_pct = None
            if c_raw is not None:
                try:
                    c_pct = float(c_raw)
                    if c_pct <= 1:
                        c_pct *= 100.0
                    confs.append(c_pct)
                except Exception:
                    c_pct = None

            u_raw = t.get("uncertainty")
            u_pct = None
            if u_raw is not None:
                try:
                    u_pct = float(u_raw)
                    if u_pct <= 1:
                        u_pct *= 100.0
                    uncertainties.append(u_pct)
                except Exception:
                    u_pct = None

            turn_idx = int(t.get("turn_index", 0) or 0)

            tf = {}
            if isinstance(turn_feedbacks_root, dict):
                tf = turn_feedbacks_root.get(str(turn_idx), {}) or {}
            elif isinstance(turn_feedbacks_root, list):
                candidate = None
                if 0 <= turn_idx < len(turn_feedbacks_root):
                  candidate = turn_feedbacks_root[turn_idx]
                if isinstance(candidate, dict):
                    tf = candidate

            if not isinstance(tf, dict):
                tf = {}

            my_tf = tf.get(uid)

            turn_locked = len(tf) > 0
            if turn_locked:
                reviewed_in_conv += 1

            turn_feedback_users = []
            for f_uid, f_data in tf.items():
                f = f_data or {}
                user_item = {
                    "uid": f_uid,
                    "examiner_uid": f_uid,
                    "name": f.get("examiner_name") or _name_for(f_uid),
                    "examiner_name": f.get("examiner_name") or _name_for(f_uid),
                    "label": f.get("label"),
                    "explanation": f.get("explanation", ""),
                    "agreed_with_model": bool(f.get("agreed_with_model", False)),  # ✅ جديد
                    "submitted_at": f.get("submitted_at")
                }
                turn_feedback_users.append(user_item)
                conv_feedback_users_map[f_uid] = {
                    "uid": user_item["uid"],
                    "name": user_item["name"]
                }

            shared_feedback = turn_feedback_users[0] if turn_feedback_users else None

            clean_turns.append({
                "turn_index": turn_idx,
                "sender": t.get("sender", ""),
                "text": t.get("text", ""),
                "prediction": pred,
                "gt": t.get("gt", ""),
                "confidence": round(c_pct, 2) if isinstance(c_pct, (int, float)) else None,
                "uncertainty": round(u_pct, 2) if isinstance(u_pct, (int, float)) else None,
                "turn_locked": turn_locked,
                "turn_feedback": shared_feedback,
                "my_feedback": my_tf,
                "feedback_users": turn_feedback_users
            })


        ai_pct = round((ai_count / total) * 100, 2) if total else 0.0
        human_pct = round((human_count / total) * 100, 2) if total else 0.0
        conv_conf = round(sum(confs) / len(confs), 2) if confs else 0.0
        conv_locked = (total > 0 and reviewed_in_conv >= total)

        items.append({
            "conversation_id": conversation_id,
            "task_name": meta.get("task_name", "Conversation"),
            "order_index": order_map.get(conversation_id, 10**9),
            "turns_count": total,
            "ai_percentage": ai_pct,
            "human_percentage": human_pct,
            "confidence": conv_conf,
            "uncertainty": round(sum(uncertainties) / len(uncertainties), 2) if uncertainties else 0.0,
            "has_feedback": reviewed_in_conv > 0,
            "conversation_locked": conv_locked,
            "feedback_users": list(conv_feedback_users_map.values()),
            "turns": clean_turns
        })

    items.sort(key=lambda x: x["order_index"])

    active_learning_enabled = _uses_frozen_active_learning(model_key)
    if active_learning_enabled:
        model_version = _active_learning_model_version(project_id, model_key)
        frozen_selection = _freeze_active_learning_targets_if_missing(
            project_id,
            "generated_conversation",
            model_key,
            model_version,
            _generated_conversation_active_learning_candidates(project_id, model_key, raw),
            task_data.get("examiner_ids") or []
        )
        items, active_learning_info = _apply_frozen_active_learning_turn_selection(
            items,
            frozen_selection,
            "generated_conversation"
        )
    else:
        items, active_learning_info = _apply_active_learning_turn_selection(items, False)

    total_conversations = len(items)
    reviewed_conversations = sum(1 for x in items if x.get("conversation_locked"))

    if active_learning_enabled:
        status_total = active_learning_info["total"]
        status_reviewed = active_learning_info["reviewed"]
    else:
        status_total = total_conversations
        status_reviewed = reviewed_conversations

    new_status = "completed" if status_total > 0 and status_reviewed >= status_total else ("progress" if status_reviewed > 0 else "pending")
    if (task_data.get("status") or "").strip().lower() != new_status:
        db.collection("tasks").document(task_id).update({"status": new_status})

    return jsonify({
        "waiting": False,
        "project_id": project_id,
        "task_id": task_id,
        "task_status": new_status,
        "feedback_locked": new_status == "completed",
        "selected_model": selected_model,
        "selected_model_name": model_label,
        "active_learning": active_learning_info,
        "progress": {
            "reviewed": status_reviewed,
            "total": status_total,
            "unit": active_learning_info.get("unit", "conversations")
        },

        "items": items
    }), 200



@app.route("/api/task/<task_id>/conversation_feedback/<conversation_id>/turn/<int:turn_index>/submit", methods=["POST"])
def api_submit_conversation_turn_feedback(task_id, conversation_id, turn_index):
    try:
        if not session.get("idToken"):
            return jsonify({"error": "Unauthorized"}), 401

        uid = session.get("uid")

        raw_payload = request.get_json(silent=True) or {}
        if isinstance(raw_payload, list):
            data = raw_payload[0] if raw_payload and isinstance(raw_payload[0], dict) else {}
        elif isinstance(raw_payload, dict):
            data = raw_payload
        else:
            data = {}

        agree_with_model = bool(data.get("agree_with_model", False))
        label = (data.get("label") or "").strip()
        explanation = (data.get("explanation") or "").strip()

        if turn_index <= 0:
            return jsonify({"error": "Invalid turn_index"}), 400

        task_doc = db.collection("tasks").document(task_id).get()
        if not task_doc.exists:
            return jsonify({"error": "Task not found"}), 404

        task_data = task_doc.to_dict() or {}
        if task_data.get("task_type") != "labeling":
            return jsonify({"error": "Task is not labeling"}), 400
        if uid not in (task_data.get("examiner_ids") or []):
            return jsonify({"error": "Forbidden"}), 403
        if _labeling_task_completed(task_data):
            return jsonify({"error": "Feedback is locked because this labeling task is completed"}), 423

        project_id = task_data.get("project_ID")
        selected_model, model_key, _ = _pick_conversation_model_for_project(project_id)
        if not selected_model:
            return jsonify({"error": "Model selection is not completed yet"}), 400
        if _version_is_closed(project_id, _active_learning_model_version(project_id, model_key)):
            return jsonify({"error": "Feedback is locked because this evaluation version is closed"}), 423

        conv_ref = _generated_conversation_node_ref(project_id, model_key, conversation_id)
        conv_node = conv_ref.get() or {}
        if isinstance(conv_node, list):
            conv_node = conv_node[0] if conv_node and isinstance(conv_node[0], dict) else {}
        if not isinstance(conv_node, dict) or not conv_node:
            return jsonify({"error": "Conversation not found in analysis results"}), 404

        turns_raw = conv_node.get("turns", {}) or {}
        turns = list(turns_raw.values()) if isinstance(turns_raw, dict) else (turns_raw if isinstance(turns_raw, list) else [])

        target_turn = None
        for t in turns:
            if int((t or {}).get("turn_index", 0) or 0) == turn_index:
                target_turn = t
                break

        if not target_turn:
            return jsonify({"error": "Turn not found"}), 404

        # إذا Agree with Model = ON -> نأخذ نفس prediction تلقائيًا
        if agree_with_model:
            model_prediction = str(target_turn.get("prediction", "")).strip()
            if model_prediction not in ["Human", "AI"]:
                return jsonify({"error": "Model prediction is missing for this turn"}), 400
            label = model_prediction
            if not explanation:
                explanation = "Agreed with model prediction."
        else:
            if label not in ["Human", "AI"]:
                return jsonify({"error": "Invalid label"}), 400
            if label != model_prediction and not explanation:
                return jsonify({"error": "Explanation is required when correcting the model"}), 400

        turn_feedbacks_root = conv_node.get("turn_feedbacks") or {}
        existing_turn_feedbacks = {}

        if isinstance(turn_feedbacks_root, dict):
            existing_turn_feedbacks = turn_feedbacks_root.get(str(turn_index), {}) or {}
        elif isinstance(turn_feedbacks_root, list):
            candidate = None
            if 0 <= turn_index < len(turn_feedbacks_root):
                candidate = turn_feedbacks_root[turn_index]
            if isinstance(candidate, dict):
                existing_turn_feedbacks = candidate

        # قفل نهائي: أول فيدباك فقط لكل turn
        if not isinstance(existing_turn_feedbacks, dict):
            existing_turn_feedbacks = {}
        existing_owner_uid = _feedback_owner_uid(existing_turn_feedbacks)
        existing_user_feedback = _feedback_record_for_uid(existing_turn_feedbacks, uid)
        if existing_owner_uid and existing_owner_uid != uid:
            return jsonify({"error": "Feedback already submitted for this turn and can only be edited by the original reviewer"}), 409

        detection_snapshot = _find_detection_snapshot_for_feedback(
            "generated_conversation",
            project_id,
            task_id=task_id,
            conversation_id=conversation_id,
            turn_index=turn_index,
            target_turn=target_turn
        )
        sample_id = _make_active_learning_sample_id(
            "generated_conversation",
            conversation_id=conversation_id,
            turn_index=turn_index
        )
        try:
            _ensure_feedback_can_be_saved(project_id, detection_snapshot.get("version_id") or _active_learning_model_version(project_id, model_key), sample_id, uid)
        except ValueError as e:
            return jsonify({"error": str(e)}), 423

        examiner_name = _reviewer_display_name(uid)

        now_iso = datetime.utcnow().isoformat() + "Z"
        payload = {
            "examiner_uid": uid,
            "examiner_name": examiner_name,
            "agreed_with_model": agree_with_model,  # ✅ جديد
            "label": label,
            "explanation": explanation,
            "submitted_at": now_iso,
            "status": "draft_saved",
            "lifecycle_status": "draft_saved",
            "locked": False
        }
        payload, is_edit = _prepare_feedback_payload(existing_user_feedback, payload, uid, examiner_name, now_iso)

        conv_ref.child("turn_feedbacks").child(str(turn_index)).child(uid).set(payload)

        try:
            _write_active_learning_feedback(project_id, sample_id, detection_snapshot, payload)
            if _uses_frozen_active_learning(model_key):
                model_version = _active_learning_model_version(project_id, model_key)
                _mark_frozen_active_learning_target_reviewed(
                    project_id,
                    model_version,
                    sample_id,
                    uid,
                    payload.get("submitted_at"),
                    {"conversation_id": conversation_id, "turn_index": turn_index}
                )
        except Exception as normalized_error:
            app.logger.exception("Failed to write normalized generated conversation feedback: %s", normalized_error)

        return jsonify({
            "message": _feedback_write_message(is_edit),
            "updated": is_edit,
            "conversation_id": conversation_id,
            "turn_index": turn_index,
            "selected_model": selected_model
        }), 200

    except Exception as e:
        app.logger.exception("api_submit_conversation_turn_feedback failed: %s", e)
        return jsonify({"error": "Server error while saving turn feedback"}), 500


def _feedback_examiner_name(uid):
    user_doc = db.collection("users").document(uid).get()
    if not user_doc.exists:
        return _short_uid(uid)

    user_data = user_doc.to_dict() or {}
    profile = user_data.get("profile", {}) or {}
    name = f"{profile.get('firstName','')} {profile.get('lastName','')}".strip()
    if name and not _looks_garbled_text(name):
        return name
    return _safe_str(user_data.get("email")) or _short_uid(uid)


def _uploaded_conversation_run(project_id, model_key):
    base_ref = rtdb.reference(f"analysis_results/conversations/{model_key}/{project_id}")
    run_id = _safe_str(base_ref.child("latest_run_id").get())
    if not run_id:
        return None, None, None

    run_ref = base_ref.child("runs").child(run_id)
    summary = run_ref.child("summary").get()
    if not summary:
        return None, None, None

    return run_id, run_ref, summary


def _uploaded_feedback_model_key(model_key):
    return CONV_RNN_KEY if model_key == CONV_RNN_KEY else CONV_LOGREG_KEY


def _label_to_int(label):
    return _normalize_binary_label(label)


def _is_ai_label(label):
    value = _safe_str(label).strip().lower()
    return _normalize_binary_label(label) == 1 or value in ("ai-generated", "ai generated", "true")


def _make_active_learning_sample_id(project_type, article_id=None, row_id=None, task_id=None, conversation_id=None, dialogue_id=None, turn_index=None, chunk_index=None):
    if project_type == "news":
        parts = ["news"]
        if article_id:
            parts.extend(["article", article_id])
        if chunk_index is not None and chunk_index != "":
            parts.extend(["chunk", chunk_index])
        return re.sub(r'[.#$\[\]/]', "_", ":".join(_safe_str(part) for part in parts if _safe_str(part)))

    parts = [project_type]
    if article_id:
        parts.append(f"article:{article_id}")
    if chunk_index is not None and chunk_index != "":
        parts.append(f"chunk:{chunk_index}")
    if row_id:
        parts.append(f"row:{row_id}")
    if task_id:
        parts.append(f"task:{task_id}")
    if conversation_id:
        parts.append(f"conversation:{conversation_id}")
    if dialogue_id:
        parts.append(f"dialogue:{dialogue_id}")
    if turn_index is not None and turn_index != "":
        parts.append(f"turn:{turn_index}")
    return _rtdb_safe_key("|".join(_safe_str(part) for part in parts if _safe_str(part)))


def _project_by_dataset_id(dataset_id):
    docs = list(db.collection("projects").where("dataset_id", "==", dataset_id).limit(1).stream())
    if not docs:
        return None, None
    project = docs[0].to_dict() or {}
    project_id = project.get("project_id") or project.get("project_ID") or docs[0].id
    project["project_id"] = project_id
    return project_id, project


def _find_detection_snapshot_for_feedback(project_type, project_id, **kwargs):
    project = kwargs.get("project") or get_project_basic_info(project_id) or {}
    tasks = get_project_tasks(project_id)
    selected = _get_project_selected_model(project_id, project, tasks)
    model_key = selected.get("key") or ("logistic" if project_type == "news" else CONV_LOGREG_KEY)
    model_name = selected.get("name") or ("RNN" if model_key == "rnn" else "Logistic Regression")
    version_id = _active_learning_model_version(project_id, model_key)
    snapshot = {
        "project_id": project_id,
        "project_type": project_type,
        "dataset_id": project.get("dataset_id") or kwargs.get("dataset_id") or "",
        "selected_model_key": model_key,
        "selected_model_name": model_name,
        "model_version": version_id,
        "version_id": version_id
    }

    if project_type == "news":
        article_id = _safe_str(kwargs.get("article_id"))
        chunk_index = kwargs.get("chunk_index")
        dataset_id = snapshot["dataset_id"]
        source = rtdb.reference(f"datasets/uploaded_news/{dataset_id}/{article_id}").get() or {}
        payload = source.get("payload") if isinstance(source, dict) and isinstance(source.get("payload"), dict) else source
        if not isinstance(payload, dict):
            payload = {}

        results_data = _news_results_payload(project_id, model_key) or {}
        details = results_data.get("details") or results_data.get("results") or []
        details = list(details.values()) if isinstance(details, dict) else (details if isinstance(details, list) else [])
        detection = next((item for item in details if _safe_str((item or {}).get("article_id") or (item or {}).get("id")) == article_id), {}) or {}
        title, text = _text_from_news_payload(payload, detection)
        chunk = _news_chunk_by_index(detection, chunk_index) if chunk_index is not None else None
        if chunk:
            text = chunk.get("chunk_text") or chunk.get("text") or text
        snapshot.update({
            "source_id": article_id,
            "article_id": article_id,
            "chunk_index": int(chunk.get("chunk_index") or chunk_index) if chunk else None,
            "chunk_id": chunk.get("chunk_id") if chunk else None,
            "chunk_text": chunk.get("chunk_text") if chunk else None,
            "target_unit": "chunk" if chunk else "article",
            "title": title,
            "parent_article_title": title,
            "text": text,
            "original_label": _first_present(payload.get("MachineGen"), payload.get("label"), payload.get("target"), payload.get("ground_truth")),
            "model_prediction": chunk.get("prediction") if chunk else detection.get("prediction"),
            "model_prediction_label": chunk.get("prediction") if chunk else detection.get("prediction"),
            "parent_article_prediction": detection.get("prediction"),
            "confidence": _owner_decimal(chunk.get("confidence") if chunk else detection.get("confidence")),
            "uncertainty": _owner_decimal(chunk.get("uncertainty") if chunk else detection.get("uncertainty"))
        })
        if chunk:
            snapshot["source_id"] = f"{article_id}:chunk:{snapshot.get('chunk_index')}"
        return snapshot

    if project_type == "uploaded_conversation":
        target_turn = kwargs.get("target_turn") or {}
        dialogue_id = kwargs.get("dialogue_id")
        turn_index = kwargs.get("turn_index")
        snapshot.update({
            "source_id": _first_present(target_turn.get("row_id"), target_turn.get("source_row_id"), f"{dialogue_id}:{turn_index}"),
            "row_id": _safe_str(target_turn.get("row_id") or target_turn.get("source_row_id")),
            "dialogue_id": dialogue_id,
            "turn_index": turn_index,
            "target_unit": "turn",
            "text": target_turn.get("text"),
            "prev_text": _first_present(target_turn.get("previous_text"), target_turn.get("prev_text")),
            "original_label": _first_present(target_turn.get("ground_truth"), target_turn.get("gt")),
            "model_prediction": target_turn.get("prediction"),
            "model_prediction_label": "AI" if _is_ai_label(target_turn.get("prediction")) else "Human",
            "confidence": _owner_decimal(target_turn.get("confidence")),
            "uncertainty": _owner_decimal(target_turn.get("uncertainty"))
        })
        return snapshot

    target_turn = kwargs.get("target_turn") or {}
    task_id = kwargs.get("task_id")
    conversation_id = kwargs.get("conversation_id")
    turn_index = kwargs.get("turn_index")
    snapshot.update({
        "source_id": f"{conversation_id}:{turn_index}",
        "task_id": task_id,
        "conversation_id": conversation_id,
        "turn_index": turn_index,
        "target_unit": "turn",
        "text": target_turn.get("text"),
        "prev_text": _first_present(target_turn.get("previous_text"), target_turn.get("prev_text")),
        "original_label": _first_present(target_turn.get("gt"), target_turn.get("ground_truth")),
        "model_prediction": target_turn.get("prediction"),
        "model_prediction_label": "AI" if _is_ai_label(target_turn.get("prediction")) else "Human",
        "confidence": _owner_decimal(target_turn.get("confidence")),
        "uncertainty": _owner_decimal(target_turn.get("uncertainty"))
    })
    return snapshot


def _write_active_learning_feedback(project_id, sample_id, detection_snapshot, feedback_data):
    snapshot = detection_snapshot or {}
    feedback = feedback_data or {}
    agreed_with_model = bool(feedback.get("agreed_with_model", False))
    model_label = snapshot.get("model_prediction_label") or snapshot.get("model_prediction")
    submitted_label = feedback.get("label")
    corrected_label_text = model_label if agreed_with_model else submitted_label
    corrected_label = _label_to_machinegen(corrected_label_text)

    record = {
        "project_id": project_id,
        "sample_id": sample_id,
        "project_type": snapshot.get("project_type"),
        "dataset_id": snapshot.get("dataset_id"),
        "source_id": snapshot.get("source_id"),
        "target_unit": snapshot.get("target_unit"),
        "article_id": snapshot.get("article_id"),
        "chunk_index": snapshot.get("chunk_index"),
        "chunk_id": snapshot.get("chunk_id"),
        "chunk_text": snapshot.get("chunk_text"),
        "parent_article_title": snapshot.get("parent_article_title") or snapshot.get("title"),
        "parent_article_prediction": snapshot.get("parent_article_prediction"),
        "row_id": snapshot.get("row_id"),
        "task_id": snapshot.get("task_id"),
        "conversation_id": snapshot.get("conversation_id"),
        "dialogue_id": snapshot.get("dialogue_id"),
        "turn_index": snapshot.get("turn_index"),
        "title": snapshot.get("title"),
        "text": snapshot.get("text"),
        "prev_text": snapshot.get("prev_text"),
        "original_label": snapshot.get("original_label"),
        "model_prediction": _label_to_machinegen(model_label),
        "model_prediction_label": model_label,
        "corrected_label": corrected_label,
        "corrected_label_text": corrected_label_text,
        "MachineGen": corrected_label,
        "confidence": snapshot.get("confidence"),
        "uncertainty": snapshot.get("uncertainty"),
        "selected_model_key": snapshot.get("selected_model_key"),
        "selected_model_name": snapshot.get("selected_model_name"),
        "model_version": snapshot.get("model_version") or "v1",
        "version_id": snapshot.get("version_id") or snapshot.get("model_version") or "v1",
        "examiner_uid": feedback.get("examiner_uid"),
        "examiner_name": feedback.get("examiner_name"),
        "reviewed_by": feedback.get("reviewed_by") or feedback.get("examiner_uid"),
        "reviewed_by_name": feedback.get("reviewed_by_name") or feedback.get("examiner_name"),
        "reviewed_at": feedback.get("reviewed_at") or feedback.get("submitted_at"),
        "agreed_with_model": agreed_with_model,
        "feedback_explanation": _safe_str(feedback.get("explanation") or feedback.get("feedback_explanation")),
        "submitted_at": feedback.get("submitted_at") or _now_utc_iso(),
        "updated_at": feedback.get("updated_at"),
        "updated_by": feedback.get("updated_by"),
        "edit_count": int(feedback.get("edit_count") or 0),
        "previous_feedback_history": feedback.get("previous_feedback_history") if isinstance(feedback.get("previous_feedback_history"), list) else [],
        "used_for_retraining": False,
        "training_export_id": None
    }
    if record["version_id"] == "v1":
        rtdb.reference(f"active_learning_feedback/{project_id}/{sample_id}").set(_owner_json_value(record))
    else:
        rtdb.reference(f"active_learning_feedback_versions/{project_id}/{record['version_id']}/{sample_id}").set(_owner_json_value(record))
    _write_version_feedback_record(project_id, record["version_id"], sample_id, record)
    return record


def _version_path(project_id, version_id):
    return f"detection_versions/{_safe_str(project_id)}/{_safe_str(version_id or 'v1')}"


def _project_versions_path(project_id):
    return f"project_versions/{_safe_str(project_id)}"


def _feedback_versions_path(project_id, version_id):
    return f"feedback_versions/{_safe_str(project_id)}/{_safe_str(version_id or 'v1')}"


def _detection_version_id(project_id, model_key=None):
    return _active_learning_model_version(project_id, model_key or "logistic")


def _official_project_type(project):
    result_type = detect_project_result_type(project or {})
    return result_type if result_type in ("news", "uploaded_conversation", "generated_conversation") else "news"


def _official_model_name(model_key, fallback=None):
    key = _safe_str(model_key).strip().lower()
    if key in ("rnn", CONV_RNN_KEY):
        return "RNN"
    if key in ("logreg", CONV_LOGREG_KEY, "logistic", "logistic regression") or "logistic" in key:
        return "Logistic Regression"
    return fallback or _safe_str(model_key) or "Unknown Model"


def _label_text_only(value):
    normalized = _normalize_binary_label(value)
    if normalized == 0:
        return "Human"
    if normalized == 1:
        return "AI"
    text = _safe_str(value).strip()
    if text.lower() in ("machine-generated", "machine generated", "machine"):
        return "AI"
    return text if text in ("Human", "AI") else None


def _prediction_int_from_label(label):
    normalized = _normalize_binary_label(label)
    if normalized is not None:
        return normalized
    return 1 if _is_ai_label(label) else 0


def _model_version_label(model_key, dataset_version):
    key = _safe_str(model_key).strip().lower()
    version = _safe_str(dataset_version or "v1").strip() or "v1"
    if key in ("rnn", CONV_RNN_KEY, "baseline_rnn", "conv_rnn"):
        prefix = "rnn"
    else:
        prefix = "logistic"
    return f"{prefix}_{version}"


def _clean_review_state(target, feedback):
    if not isinstance(target, dict) or not target:
        return "not_selected_for_review"

    if _feedback_is_final(feedback) or _target_feedback_finalized(target):
        return "locked" if bool((feedback or {}).get("locked") or target.get("locked")) else "submitted"

    if isinstance(feedback, dict) and feedback:
        return "draft_saved"

    status = _safe_str(target.get("target_status")).strip().lower()
    if status in ("pending", "draft_saved", "submitted", "locked"):
        return status
    if status in ("reviewed", "accepted", "selected_for_review"):
        return "draft_saved"
    return "pending"


def _feedback_label_from_record(feedback, prediction):
    if not isinstance(feedback, dict) or not feedback:
        return ""
    label = _label_text_only(
        feedback.get("corrected_label_text")
        or feedback.get("corrected_label")
        or feedback.get("label")
    )
    if label:
        return label
    if feedback.get("agreed_with_model") is True:
        return _label_text_only(prediction) or ""
    return ""


def _reviewed_by_from_feedback(feedback, target):
    if isinstance(feedback, dict) and feedback:
        return _safe_str(feedback.get("reviewed_by") or feedback.get("examiner_uid") or feedback.get("uid") or feedback.get("examiner_id")) or None
    if isinstance(target, dict) and target:
        return _safe_str(target.get("reviewed_by") or target.get("submitted_by") or target.get("draft_by")) or None
    return None


def _reviewed_at_from_feedback(feedback, target):
    if isinstance(feedback, dict) and feedback:
        return _safe_str(feedback.get("reviewed_at") or feedback.get("submitted_at") or feedback.get("updated_at") or feedback.get("draft_updated_at")) or None
    if isinstance(target, dict) and target:
        return _safe_str(target.get("reviewed_at") or target.get("submitted_at") or target.get("draft_saved_at")) or None
    return None


def _snapshot_row_key(sample_id):
    return _safe_target_key(sample_id)


def _row_review_targets(project_id, version_id):
    selection = _load_frozen_active_learning_targets(project_id, version_id)
    targets = {}
    for target in _frozen_target_values(selection):
        sample_id = _safe_str((target or {}).get("sample_id"))
        if sample_id:
            targets[sample_id] = target
    return targets


def _news_detection_rows(project_id, project, model_key, version_id, task_id, created_at):
    dataset_id = project.get("dataset_id") or ""
    dataset_rows = rtdb.reference(f"datasets/uploaded_news/{dataset_id}").get() or {}
    results_data = _news_results_payload(project_id, model_key)
    details = results_data.get("details") or results_data.get("results") or []
    details = list(details.values()) if isinstance(details, dict) else (details if isinstance(details, list) else [])
    targets = _row_review_targets(project_id, version_id)

    rows = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        article_id = _safe_str(detail.get("article_id") or detail.get("id"))
        source = dataset_rows.get(article_id, {}) if isinstance(dataset_rows, dict) else {}
        payload = _news_source_payload(source)
        title, article_text = _text_from_news_payload(payload, detail)
        chunks = detail.get("chunks") or []
        if not isinstance(chunks, list):
            chunks = []

        for pos, chunk in enumerate(chunks, start=1):
            if not isinstance(chunk, dict):
                continue
            chunk_index = int(chunk.get("chunk_index") or pos)
            sample_id = _make_active_learning_sample_id("news", article_id=article_id, chunk_index=chunk_index)
            target = targets.get(sample_id) or {}
            prediction = _label_text_only(chunk.get("prediction") or detail.get("prediction"))
            row = {
                "sample_id": sample_id,
                "article_id": article_id,
                "chunk_id": chunk.get("chunk_id") or f"chunk:{chunk_index}",
                "chunk_index": chunk_index,
                "title": title,
                "text": chunk.get("chunk_text") or chunk.get("text") or article_text,
                "ground_truth": _label_text_only(_first_present(chunk.get("ground_truth"), detail.get("ground_truth"), _extract_ground_truth_from_payload(payload))),
                "prediction": prediction,
                "prediction_int": _first_present(chunk.get("prediction_int"), _prediction_int_from_label(prediction)),
                "confidence": _owner_decimal(chunk.get("confidence")),
                "uncertainty": _owner_decimal(chunk.get("uncertainty")),
                "review_target": bool(target),
                "feedback_label": None,
                "final_label": prediction,
                "model_used": _official_model_name(model_key),
                "active_learning_state": "pending" if target else "not_selected_for_review",
                "timestamp": created_at
            }
            rows.append(row)
    return rows


def _uploaded_conversation_detection_rows(project_id, project, model_key, version_id, task_id, created_at):
    run_id, run_ref, _ = _uploaded_conversation_run(project_id, model_key)
    if not run_ref:
        return [], None
    dialogue_turns = run_ref.child("dialogue_turns").get() or {}
    key_map = run_ref.child("dialogue_key_map").get() or {}
    reverse_key_map = {v: k for k, v in key_map.items()} if isinstance(key_map, dict) else {}
    targets = _row_review_targets(project_id, version_id)
    rows = []

    for safe_key, turns_raw in (dialogue_turns.items() if isinstance(dialogue_turns, dict) else []):
        turns = _turns_to_list(turns_raw)
        for idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            turn_index = int(turn.get("turn_index", idx + 1) or idx + 1)
            dialogue_id = turn.get("dialogue_id") or reverse_key_map.get(safe_key, safe_key)
            sample_id = _make_active_learning_sample_id(
                "uploaded_conversation",
                row_id=turn.get("row_id") or turn.get("source_row_id"),
                dialogue_id=dialogue_id,
                turn_index=turn_index
            )
            source_id = _safe_str(turn.get("row_id") or turn.get("source_row_id") or dialogue_id)
            target = targets.get(sample_id) or {}
            prediction = _label_text_only(turn.get("prediction"))
            rows.append({
                "sample_id": sample_id,
                "source_id": source_id,
                "dialogue_id": dialogue_id,
                "turn_index": turn_index,
                "sender": turn.get("sender"),
                "text": turn.get("text"),
                "previous_turn": _first_present(turn.get("previous_text"), turn.get("prev_text")),
                "ground_truth": _label_text_only(_first_present(turn.get("ground_truth"), turn.get("gt"))),
                "prediction": prediction,
                "prediction_int": _first_present(turn.get("prediction_int"), _prediction_int_from_label(prediction)),
                "confidence": _owner_decimal(turn.get("confidence")),
                "uncertainty": _owner_decimal(turn.get("uncertainty")),
                "review_target": bool(target),
                "feedback_label": None,
                "final_label": prediction,
                "model_used": _official_model_name(model_key),
                "active_learning_state": "pending" if target else "not_selected_for_review",
                "timestamp": created_at
            })
    return rows, run_id


def _generated_conversation_detection_rows(project_id, project, model_key, version_id, task_id, created_at):
    raw = _generated_conversation_results_payload(project_id, model_key) or {}
    targets = _row_review_targets(project_id, version_id)
    rows = []
    for node_key, node in (raw.items() if isinstance(raw, dict) else []):
        if not isinstance(node, dict):
            continue
        meta = node.get("meta") or {}
        conversation_id = meta.get("task_id") or node_key
        turns = _turns_to_list(node.get("turns") or {})
        for idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            turn_index = int(turn.get("turn_index", idx + 1) or idx + 1)
            sample_id = _make_active_learning_sample_id(
                "generated_conversation",
                conversation_id=conversation_id,
                turn_index=turn_index
            )
            target = targets.get(sample_id) or {}
            prediction = _label_text_only(turn.get("prediction"))
            rows.append({
                "sample_id": sample_id,
                "source_id": conversation_id,
                "dialogue_id": conversation_id,
                "turn_index": turn_index,
                "sender": turn.get("sender"),
                "text": turn.get("text"),
                "previous_turn": _first_present(turn.get("previous_text"), turn.get("prev_text")),
                "ground_truth": _label_text_only(_first_present(turn.get("ground_truth"), turn.get("gt"))),
                "prediction": prediction,
                "prediction_int": _first_present(turn.get("prediction_int"), _prediction_int_from_label(prediction)),
                "confidence": _owner_decimal(turn.get("confidence")),
                "uncertainty": _owner_decimal(turn.get("uncertainty")),
                "review_target": bool(target),
                "feedback_label": None,
                "final_label": prediction,
                "model_used": _official_model_name(model_key),
                "active_learning_state": "pending" if target else "not_selected_for_review",
                "timestamp": created_at
            })
    return rows, _safe_str(_generated_conversation_base_ref(project_id, model_key).child("latest_run_id").get())


def _version_rows_to_map(rows):
    result = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        sample_id = _safe_str(row.get("sample_id"))
        if sample_id:
            result[_snapshot_row_key(sample_id)] = row
    return result


def _version_number(version_id):
    match = re.match(r"^v(\d+)$", _safe_str(version_id).strip().lower())
    return int(match.group(1)) if match else 0


def _normalize_version_registry(project_id, registry=None):
    registry = registry if isinstance(registry, dict) else (rtdb.reference(_project_versions_path(project_id)).get() or {})
    if not isinstance(registry, dict):
        registry = {}

    raw_versions = registry.get("versions") or {}
    versions = {}
    if isinstance(raw_versions, list):
        for item in raw_versions:
            if isinstance(item, dict):
                vid = _safe_str(item.get("version_id") or f"v{len(versions) + 1}") or f"v{len(versions) + 1}"
                item = dict(item)
                item["version_id"] = vid
                versions[vid] = item
    elif isinstance(raw_versions, dict):
        for vid, item in raw_versions.items():
            if isinstance(item, dict):
                clean_vid = _safe_str(item.get("version_id") or vid) or _safe_str(vid)
                item = dict(item)
                item["version_id"] = clean_vid
                versions[clean_vid] = item

    current_version = _safe_str(registry.get("current_version"))
    if not current_version and versions:
        current_version = sorted(versions.keys(), key=_version_number)[-1]

    closed_versions = [
        vid for vid, item in versions.items()
        if isinstance(item, dict) and _safe_str(item.get("status")) == "evaluation_closed"
    ]
    latest_closed = _safe_str(registry.get("latest_closed_version"))
    if not latest_closed and closed_versions:
        latest_closed = sorted(closed_versions, key=_version_number)[-1]

    normalized = {
        "project_id": project_id,
        "current_version": current_version or "v1",
        "previous_version": _safe_str(registry.get("previous_version")),
        "latest_closed_version": latest_closed,
        "total_versions": len(versions),
        "versions": versions
    }
    return normalized


def _save_version_registry(project_id, registry):
    registry = _normalize_version_registry(project_id, registry)
    registry["total_versions"] = len(registry.get("versions") or {})
    rtdb.reference(_project_versions_path(project_id)).set(_owner_json_value(registry))
    return registry


def _version_record(project_id, version_id):
    registry = _normalize_version_registry(project_id)
    return (registry.get("versions") or {}).get(_safe_str(version_id)) or {}


def _version_is_closed(project_id, version_id):
    return _safe_str(_version_record(project_id, version_id).get("status")) == "evaluation_closed"


def _next_version_id(project_id):
    registry = _normalize_version_registry(project_id)
    versions = registry.get("versions") or {}
    highest = max([_version_number(vid) for vid in versions.keys()] or [0])
    return f"v{highest + 1}"


def _write_project_version_registry(project_id, version_record):
    registry = _normalize_version_registry(project_id)
    version_id = version_record.get("version_id") or "v1"
    previous_current = _safe_str(registry.get("current_version"))
    versions = registry.get("versions") or {}
    existing = versions.get(version_id) if isinstance(versions.get(version_id), dict) else {}
    merged = dict(existing)
    merged.update(version_record)
    merged["version_id"] = version_id
    versions[version_id] = merged
    registry["versions"] = versions
    if previous_current and previous_current != version_id:
        registry["previous_version"] = previous_current
    registry["current_version"] = version_id
    _save_version_registry(project_id, registry)


def _create_detection_version_snapshot(project_id, task_id, model_key, model_name, created_by, analysis_run_id=None, version_id=None, rows_override=None, status=None):
    project = get_project_basic_info(project_id) or {}
    project_type = _official_project_type(project)
    version_id = version_id or _detection_version_id(project_id, model_key)
    if _version_is_closed(project_id, version_id):
        raise ValueError("Evaluation version is closed and cannot be overwritten.")
    created_at = _now_utc_iso()
    source_dataset_id = project.get("dataset_id") or ""

    if rows_override is not None:
        rows = rows_override
        target_unit = "chunk" if project_type == "news" else "turn"
    elif project_type == "news":
        rows = _news_detection_rows(project_id, project, model_key, version_id, task_id, created_at)
        target_unit = "chunk"
    elif project_type == "uploaded_conversation":
        rows, found_run_id = _uploaded_conversation_detection_rows(project_id, project, model_key, version_id, task_id, created_at)
        analysis_run_id = analysis_run_id or found_run_id
        target_unit = "turn"
    else:
        rows, found_run_id = _generated_conversation_detection_rows(project_id, project, model_key, version_id, task_id, created_at)
        analysis_run_id = analysis_run_id or found_run_id
        target_unit = "turn"

    metadata = {
        "project_id": project_id,
        "project_type": project_type,
        "model_key": model_key,
        "model_name": model_name or _official_model_name(model_key),
        "version_id": version_id,
        "created_at": created_at,
        "created_by": created_by,
        "task_id": task_id,
        "source_dataset_id": source_dataset_id,
        "analysis_run_id": analysis_run_id,
        "active_learning_supported": _active_learning_retraining_supported(model_key),
        "retraining_supported": _active_learning_retraining_supported(model_key),
        "target_unit": target_unit,
        "row_count": len(rows),
        "final_label_policy": "Rows without submitted or locked feedback keep the model prediction as final_label."
    }
    snapshot = {
        "metadata": metadata,
        "rows": _version_rows_to_map(rows)
    }
    rtdb.reference(_version_path(project_id, version_id)).set(_owner_json_value(snapshot))
    frozen_selection = _load_frozen_active_learning_targets(project_id, version_id)
    review_target_count = len(_frozen_target_values(frozen_selection))
    _write_project_version_registry(project_id, {
        "version_id": version_id,
        "model_key": model_key,
        "model_name": metadata["model_name"],
        "task_id": task_id,
        "created_at": created_at,
        "status": status or ("feedback_in_progress" if review_target_count else "detection_finalized"),
        "metrics_available": False,
        "active_learning_supported": metadata["active_learning_supported"],
        "review_target_count": review_target_count
    })
    return snapshot


def _write_version_feedback_record(project_id, version_id, sample_id, record):
    if not project_id or not sample_id:
        return None
    version_id = version_id or "v1"
    if _version_is_closed(project_id, version_id):
        raise ValueError("Evaluation version is closed and cannot accept feedback.")
    now_iso = record.get("updated_at") or record.get("submitted_at") or _now_utc_iso()
    feedback_ref = rtdb.reference(_feedback_versions_path(project_id, version_id)).child(_snapshot_row_key(sample_id))
    existing = feedback_ref.get() or {}
    if isinstance(existing, dict) and existing:
        owner_uid = _safe_str(existing.get("examiner_uid"))
        incoming_uid = _safe_str(record.get("examiner_uid"))
        if owner_uid and incoming_uid and owner_uid != incoming_uid:
            raise ValueError("Feedback already exists for this sample and can only be edited by the original reviewer.")
        if _feedback_is_final(existing):
            raise ValueError("Feedback is locked after final submission.")
    corrected_label = _label_text_only(record.get("corrected_label_text") or record.get("corrected_label"))
    prediction_label = _label_text_only(record.get("model_prediction_label") or record.get("model_prediction"))
    feedback_record = {
        "project_id": project_id,
        "version_id": version_id,
        "sample_id": sample_id,
        "corrected_label": corrected_label,
        "feedback_explanation": _safe_str(record.get("feedback_explanation")),
        "examiner_uid": record.get("examiner_uid"),
        "examiner_name": record.get("examiner_name"),
        "submitted_at": record.get("submitted_at") or now_iso,
        "draft_updated_at": now_iso,
        "updated_at": now_iso,
        "status": "draft_saved",
        "lifecycle_status": "draft_saved",
        "locked": False,
        "agreed_with_model": bool(record.get("agreed_with_model", False)) or bool(corrected_label and prediction_label and corrected_label == prediction_label),
        "history": record.get("previous_feedback_history") if isinstance(record.get("previous_feedback_history"), list) else []
    }
    feedback_ref.set(_owner_json_value(feedback_record))
    _update_project_version_status(project_id, version_id, "feedback_in_progress")
    return feedback_record


def _ensure_feedback_can_be_saved(project_id, version_id, sample_id, examiner_uid):
    existing = rtdb.reference(_feedback_versions_path(project_id, version_id)).child(_snapshot_row_key(sample_id)).get() or {}
    if not isinstance(existing, dict) or not existing:
        return
    owner_uid = _safe_str(existing.get("examiner_uid"))
    if owner_uid and owner_uid != _safe_str(examiner_uid):
        raise ValueError("Feedback already exists for this sample and can only be edited by the original reviewer.")
    if _feedback_is_final(existing):
        raise ValueError("Feedback is locked after final submission.")


def _version_feedback_status(project_id, version_id):
    selection = _load_frozen_active_learning_targets(project_id, version_id)
    targets = _frozen_target_values(selection)
    if not targets:
        return "feedback_in_progress"
    feedback_map = _load_version_feedback(project_id, version_id)
    finalized = 0
    for target in targets:
        sample_id = _safe_str(target.get("sample_id"))
        feedback = (feedback_map or {}).get(_snapshot_row_key(sample_id)) if sample_id else None
        if _target_feedback_finalized(target) or _feedback_is_final(feedback):
            finalized += 1
    return "feedback_completed" if finalized >= len(targets) else "feedback_in_progress"


def _invalidate_evaluation_snapshot(project_id, version_id, reason):
    now_iso = _now_utc_iso()
    payload = {
        "stale": True,
        "rebuild_required": True,
        "stale_reason": reason,
        "stale_at": now_iso
    }
    rtdb.reference(f"evaluation_snapshots/{project_id}/{version_id}").update(payload)
    rtdb.reference(_version_path(project_id, version_id)).child("evaluation_snapshot").update(payload)
    registry = _normalize_version_registry(project_id)
    versions = registry.get("versions") or {}
    record = dict(versions.get(version_id) or {"version_id": version_id})
    record["metrics_available"] = False
    record["evaluation_rebuild_required"] = True
    versions[version_id] = record
    registry["versions"] = versions
    _save_version_registry(project_id, registry)


def _set_labeling_tasks_status_for_version(project_id, version_id, status):
    updated = []
    for doc in db.collection("tasks").where("project_ID", "==", project_id).where("task_type", "==", "labeling").stream():
        data = doc.to_dict() or {}
        active_version = _safe_str(data.get("active_version") or version_id)
        if active_version and active_version != _safe_str(version_id):
            continue
        db.collection("tasks").document(doc.id).update({
            "status": status,
            "active_version": version_id,
            "feedback_status_updated_at": _now_utc_iso()
        })
        updated.append(doc.id)
    return updated


def _validate_feedback_final_submit(project_id, version_id, examiner_uid):
    registry = _normalize_version_registry(project_id)
    version_record = (registry.get("versions") or {}).get(version_id)
    if not isinstance(version_record, dict):
        raise ValueError("Version not found.")
    if _version_is_closed(project_id, version_id):
        raise ValueError("Evaluation version is closed.")
    if _safe_str(version_record.get("status")) in ("feedback_completed", "evaluation_ready", "evaluation_closed"):
        raise ValueError("Feedback task has already been submitted for this version.")

    project_tasks = get_project_tasks(project_id)
    assigned = False
    for task in project_tasks:
        if _safe_str(task.get("task_type")) == "labeling" and examiner_uid in (task.get("examiner_ids") or []):
            active_version = _safe_str(task.get("active_version") or version_id)
            if not active_version or active_version == _safe_str(version_id):
                assigned = True
                break
    project = get_project_basic_info(project_id) or {}
    if project.get("owner_id") == examiner_uid:
        assigned = True
    if not assigned:
        raise PermissionError("You are not assigned to submit feedback for this version.")

    selection = _load_frozen_active_learning_targets(project_id, version_id)
    targets = _frozen_target_values(selection)
    if not targets:
        raise ValueError("No review targets are available for this version.")

    feedback_map = _load_version_feedback(project_id, version_id)
    missing = []
    wrong_owner = []
    submitted = []
    draft_keys = []
    for target in targets:
        sample_id = _safe_str(target.get("sample_id"))
        key = _snapshot_row_key(sample_id)
        feedback = (feedback_map or {}).get(key)
        if not isinstance(feedback, dict) or not feedback:
            missing.append(sample_id)
            continue
        owner_uid = _safe_str(feedback.get("examiner_uid"))
        if owner_uid != _safe_str(examiner_uid):
            wrong_owner.append(sample_id)
            continue
        if _feedback_is_final(feedback):
            submitted.append(sample_id)
            continue
        label = _label_text_only(feedback.get("corrected_label") or feedback.get("corrected_label_text") or feedback.get("label"))
        if not label or _feedback_lifecycle_status(feedback) != "draft_saved":
            missing.append(sample_id)
            continue
        draft_keys.append((key, sample_id))

    if submitted and len(submitted) == len(targets):
        raise ValueError("Feedback task has already been submitted for this version.")
    if missing or wrong_owner:
        remaining = len(missing) + len(wrong_owner)
        raise ValueError(f"Complete {remaining} remaining review targets before final submission.")
    if len(draft_keys) + len(submitted) < len(targets):
        remaining = len(targets) - len(draft_keys) - len(submitted)
        raise ValueError(f"Complete {remaining} remaining review targets before final submission.")
    return draft_keys


def _mark_frozen_targets_submitted(project_id, version_id, sample_ids, examiner_uid, submitted_at):
    selection = _load_frozen_active_learning_targets(project_id, version_id)
    if not selection:
        return 0
    selection_run_id = selection.get("selection_run_id")
    submitted = 0
    for sample_id in sample_ids:
        target_key = _safe_target_key(sample_id)
        payload = {
            "target_status": "submitted",
            "submitted_by": examiner_uid,
            "submitted_at": submitted_at,
            "reviewed_by": examiner_uid,
            "reviewed_at": submitted_at,
            "locked": True,
            "feedback_count": 1
        }
        rtdb.reference(_review_targets_path(project_id, version_id, selection_run_id)).child("targets").child(target_key).update(payload)
        rtdb.reference(_active_learning_targets_path(project_id, version_id, selection_run_id)).child("targets").child(target_key).update(payload)
        submitted += 1
    return submitted


def _finalize_feedback_for_examiner(project_id, version_id, examiner_uid):
    draft_keys = _validate_feedback_final_submit(project_id, version_id, examiner_uid)
    now_iso = _now_utc_iso()
    submitted_sample_ids = []
    for key, sample_id in draft_keys:
        update_payload = {
            "status": "submitted",
            "lifecycle_status": "submitted",
            "locked": True,
            "submitted_at": now_iso,
            "finalized_at": now_iso,
            "finalized_by": examiner_uid,
            "updated_at": now_iso
        }
        rtdb.reference(_feedback_versions_path(project_id, version_id)).child(key).update(update_payload)
        if _safe_str(version_id) == "v1":
            rtdb.reference(f"active_learning_feedback/{project_id}/{sample_id}").update(update_payload)
        else:
            rtdb.reference(f"active_learning_feedback_versions/{project_id}/{version_id}/{sample_id}").update(update_payload)
        submitted_sample_ids.append(sample_id)
    _mark_frozen_targets_submitted(project_id, version_id, submitted_sample_ids, examiner_uid, now_iso)
    status = _version_feedback_status(project_id, version_id)
    _update_project_version_status(project_id, version_id, status)
    task_ids = _set_labeling_tasks_status_for_version(project_id, version_id, "completed" if status == "feedback_completed" else "progress")
    _invalidate_evaluation_snapshot(project_id, version_id, "new submitted feedback")
    return {
        "submitted_count": len(submitted_sample_ids),
        "already_locked_count": 0,
        "status": status,
        "submitted_at": now_iso,
        "task_ids": task_ids
    }


def _update_project_version_status(project_id, version_id, status, metrics_available=None):
    registry = _normalize_version_registry(project_id)
    versions = registry.get("versions") or {}
    item = dict(versions.get(version_id) or {"version_id": version_id})
    item["status"] = status
    if metrics_available is not None:
        item["metrics_available"] = bool(metrics_available)
    now_iso = _now_utc_iso()
    if status == "feedback_completed" and not item.get("feedback_completed_at"):
        item["feedback_completed_at"] = now_iso
    if status == "evaluation_ready" and not item.get("evaluation_ready_at"):
        item["evaluation_ready_at"] = now_iso
    if status == "evaluation_ready":
        item["evaluation_rebuild_required"] = False
    versions[version_id] = item
    registry["versions"] = versions
    registry["current_version"] = registry.get("current_version") or version_id
    if status == "evaluation_closed":
        registry["latest_closed_version"] = version_id
    _save_version_registry(project_id, registry)


def _version_review_counts(project_id, version_id):
    snapshot = rtdb.reference(f"evaluation_snapshots/{project_id}/{version_id}").get()
    review = snapshot.get("review_statistics") if isinstance(snapshot, dict) else {}
    if isinstance(review, dict) and review and not (snapshot.get("stale") or snapshot.get("rebuild_required")):
        return {
            "review_target_count": int(review.get("total_review_targets") or 0),
            "reviewed_targets": int(review.get("reviewed_targets") or 0),
            "corrected_targets": int(review.get("corrected_targets") or 0),
            "correction_rate": float(review.get("correction_rate") or 0.0),
            "metrics_available": _evaluation_snapshot_complete(snapshot)
        }

    metadata, rows = _load_detection_version(project_id, version_id)
    try:
        enhanced_rows = _build_enhanced_dataset(project_id, version_id).get("rows") or []
    except Exception:
        enhanced_rows = rows
    stats = _review_statistics(enhanced_rows)
    feedback_map = _load_version_feedback(project_id, version_id)
    submitted_count = sum(1 for item in (feedback_map or {}).values() if isinstance(item, dict) and _feedback_is_final(item))
    selection = _load_frozen_active_learning_targets(project_id, version_id)
    target_count = len(_frozen_target_values(selection))
    return {
        "review_target_count": target_count or int(stats.get("total_review_targets") or 0),
        "reviewed_targets": submitted_count or int(stats.get("reviewed_targets") or 0),
        "corrected_targets": int(stats.get("corrected_targets") or 0),
        "correction_rate": float(stats.get("correction_rate") or 0.0),
        "metrics_available": False
    }


def _versions_response(project_id):
    registry = _normalize_version_registry(project_id)
    versions = registry.get("versions") or {}
    enriched = {}
    for version_id, item in versions.items():
        record = dict(item or {})
        counts = _version_review_counts(project_id, version_id)
        record["metrics_available"] = bool(record.get("metrics_available") or counts.get("metrics_available"))
        record["review_target_count"] = max(int(record.get("review_target_count") or 0), int(counts.get("review_target_count") or 0))
        record["reviewed_targets"] = int(counts.get("reviewed_targets") or 0)
        record["corrected_targets"] = int(counts.get("corrected_targets") or 0)
        record["correction_rate"] = float(counts.get("correction_rate") or 0.0)
        enriched[version_id] = record
    registry["versions"] = enriched
    registry["total_versions"] = len(enriched)
    return registry


def _load_detection_version(project_id, version_id):
    version = rtdb.reference(_version_path(project_id, version_id)).get() or {}
    if not isinstance(version, dict):
        return {}, []
    rows_raw = version.get("rows") or {}
    rows = list(rows_raw.values()) if isinstance(rows_raw, dict) else (rows_raw if isinstance(rows_raw, list) else [])
    return version.get("metadata") or {}, [row for row in rows if isinstance(row, dict)]


def _version_has_detection_rows(project_id, version_id):
    # نستخدمها لمنع التصدير من نسخة لم يتم اعتماد نتائج الكشف لها بعد.
    metadata, rows = _load_detection_version(project_id, version_id)
    if rows:
        return True
    try:
        return int((metadata or {}).get("row_count") or 0) > 0
    except Exception:
        return False


def _current_version_with_rows(project_id):
    registry = _normalize_version_registry(project_id)
    current = _safe_str(registry.get("current_version"))
    if current and _version_has_detection_rows(project_id, current):
        return current, registry, True

    versions = registry.get("versions") or {}
    ordered = sorted(versions.keys(), key=_version_number, reverse=True) if isinstance(versions, dict) else []
    for version_id in ordered:
        if _version_has_detection_rows(project_id, version_id):
            return version_id, registry, True

    if _version_has_detection_rows(project_id, "v1"):
        return "v1", registry, True

    return current, registry, False


def _evaluation_snapshot_debug_context(project_id, version_id, metadata=None, rows=None):
    metadata = metadata if isinstance(metadata, dict) else {}
    rows = rows if isinstance(rows, list) else []
    sample = rows[0] if rows and isinstance(rows[0], dict) else {}
    return {
        "project_id": project_id,
        "version_id": version_id,
        "project_type": metadata.get("project_type"),
        "model_key": metadata.get("model_key"),
        "row_count": len(rows),
        "sample_keys": sorted(list(sample.keys()))[:20] if sample else []
    }


def _ensure_detection_rows_for_evaluation(project_id, version_id, metadata, rows):
    project = get_project_basic_info(project_id) or {}
    project_type = (metadata or {}).get("project_type") or _official_project_type(project)
    model_key = (metadata or {}).get("model_key") or _get_project_selected_model(project_id, project, get_project_tasks(project_id)).get("key")
    model_key = model_key or ("logistic" if project_type == "news" else CONV_LOGREG_KEY)
    model_name = (metadata or {}).get("model_name") or _official_model_name(model_key)
    task_id = (metadata or {}).get("task_id")
    created_at = (metadata or {}).get("created_at") or _now_utc_iso()
    if rows:
        repaired_metadata = dict(metadata or {})
        repaired_metadata.setdefault("project_id", project_id)
        repaired_metadata.setdefault("project_type", project_type)
        repaired_metadata.setdefault("model_key", model_key)
        repaired_metadata.setdefault("model_name", model_name)
        repaired_metadata.setdefault("version_id", version_id)
        repaired_metadata.setdefault("target_unit", "chunk" if project_type == "news" else "turn")
        repaired_metadata.setdefault("row_count", len(rows))
        return repaired_metadata, rows

    rebuilt_rows = []
    analysis_run_id = (metadata or {}).get("analysis_run_id")

    if project_type == "news":
        rebuilt_rows = _news_detection_rows(project_id, project, model_key, version_id, task_id, created_at)
        target_unit = "chunk"
    elif project_type == "uploaded_conversation":
        model_key = _uploaded_feedback_model_key(model_key)
        rebuilt_rows, found_run_id = _uploaded_conversation_detection_rows(project_id, project, model_key, version_id, task_id, created_at)
        analysis_run_id = analysis_run_id or found_run_id
        target_unit = "turn"
    elif project_type == "generated_conversation":
        rebuilt_rows, found_run_id = _generated_conversation_detection_rows(project_id, project, model_key, version_id, task_id, created_at)
        analysis_run_id = analysis_run_id or found_run_id
        target_unit = "turn"
    else:
        target_unit = ""

    if not rebuilt_rows:
        return metadata or {}, []

    repaired_metadata = dict(metadata or {})
    repaired_metadata.update({
        "project_id": project_id,
        "project_type": project_type,
        "model_key": model_key,
        "model_name": model_name,
        "version_id": version_id,
        "task_id": task_id,
        "analysis_run_id": analysis_run_id,
        "target_unit": target_unit,
        "row_count": len(rebuilt_rows)
    })
    rtdb.reference(_version_path(project_id, version_id)).update({
        "metadata": _owner_json_value(repaired_metadata),
        "rows": _owner_json_value(_version_rows_to_map(rebuilt_rows))
    })
    return repaired_metadata, rebuilt_rows


def _load_version_feedback(project_id, version_id):
    rows = rtdb.reference(_feedback_versions_path(project_id, version_id)).get() or {}
    if not isinstance(rows, dict):
        rows = {}
    if not rows and _safe_str(version_id) == "v1":
        fallback = rtdb.reference(f"active_learning_feedback/{project_id}").get() or {}
        rows = {
            _snapshot_row_key(row.get("sample_id")): {
                "project_id": project_id,
                "version_id": version_id,
                "sample_id": row.get("sample_id"),
                "corrected_label": _label_text_only(row.get("corrected_label_text") or row.get("corrected_label")),
                "feedback_explanation": row.get("feedback_explanation"),
                "examiner_uid": row.get("examiner_uid"),
                "examiner_name": row.get("examiner_name"),
                "submitted_at": row.get("submitted_at"),
                "updated_at": row.get("updated_at") or row.get("submitted_at"),
                "status": "accepted"
            }
            for row in (fallback.values() if isinstance(fallback, dict) else [])
            if isinstance(row, dict) and _safe_str(row.get("sample_id"))
        }
    return rows


def _feedback_from_map_or_active(project_id, feedback_map, sample_id, version_id=None):
    sample_id = _safe_str(sample_id)
    if not sample_id:
        return {}
    mapped = (feedback_map or {}).get(_snapshot_row_key(sample_id))
    if isinstance(mapped, dict) and mapped:
        return mapped
    if version_id and _safe_str(version_id) != "v1":
        return {}
    active = rtdb.reference(f"active_learning_feedback/{project_id}/{sample_id}").get()
    return active if isinstance(active, dict) else {}


def _build_enhanced_dataset(project_id, version_id):
    metadata, rows = _load_detection_version(project_id, version_id)
    version_record = _version_record(project_id, version_id)
    if rows or metadata or version_record:
        metadata, rows = _ensure_detection_rows_for_evaluation(project_id, version_id, metadata, rows)
    feedback_map = _load_version_feedback(project_id, version_id)
    targets = _row_review_targets(project_id, version_id)
    dataset_version = _safe_str(metadata.get("version_id") or version_id or "v1") or "v1"
    model_key = metadata.get("model_key")
    model_used = metadata.get("model_name") or _official_model_name(model_key)
    model_version = _model_version_label(model_key, dataset_version)
    created_at = _safe_str(metadata.get("created_at")) or _now_utc_iso()
    enhanced = []
    article_display = {}
    for row in rows:
        item = dict(row)
        sample_id = _safe_str(item.get("sample_id"))
        target = targets.get(sample_id) or {}
        feedback = _feedback_from_map_or_active(project_id, feedback_map, sample_id, version_id)
        feedback_label = _feedback_label_from_record(feedback, item.get("prediction"))
        feedback_final = isinstance(feedback, dict) and _feedback_is_final(feedback) and bool(feedback_label)
        prediction = _label_text_only(item.get("prediction")) or _label_text_only(item.get("prediction_label")) or ""
        final_label = feedback_label if feedback_final else prediction

        if metadata.get("project_type") == "news":
            source_article_id = _safe_str(item.get("source_id") or item.get("article_id"))
            if source_article_id not in article_display:
                article_display[source_article_id] = len(article_display) + 1
            article_number = article_display[source_article_id]
            chunk_index = item.get("chunk_index")
            if chunk_index is None:
                chunk_index = 1
            clean_row = {
                "article_id": f"article_{article_number:03d}",
                "chunk_id": f"chunk_{article_number:03d}_{chunk_index}",
                "chunk_index": int(chunk_index or 1),
                "title": item.get("title") or "",
                "text": item.get("text") or "",
                "ground_truth": _label_text_only(item.get("ground_truth")),
                "prediction_label": prediction,
                "prediction_int": _first_present(item.get("prediction_int"), _prediction_int_from_label(prediction)),
                "uncertainty": _owner_decimal(item.get("uncertainty")),
                "review_target": bool(target),
                "review_priority_rank": target.get("selection_rank") if target else None,
                "active_learning_state": _clean_review_state(target, feedback),
                "feedback_label": feedback_label,
                "feedback_explanation": _safe_str((feedback or {}).get("feedback_explanation") or (feedback or {}).get("explanation")),
                "reviewed_by": _reviewed_by_from_feedback(feedback, target),
                "reviewed_at": _reviewed_at_from_feedback(feedback, target),
                "final_label": final_label,
                "model_used": model_used,
                "model_version": model_version,
                "sample_id": sample_id,
                "source_id": source_article_id,
                "dataset_version": dataset_version,
                "created_at": _safe_str(item.get("created_at") or item.get("timestamp")) or created_at,
                "updated_at": _safe_str((feedback or {}).get("updated_at") or (feedback or {}).get("submitted_at") or item.get("updated_at") or item.get("timestamp")) or created_at
            }
        else:
            clean_row = {
                "dialogue_id": _safe_str(item.get("dialogue_id") or item.get("source_id")),
                "turn_index": int(item.get("turn_index") or 0),
                "sender": item.get("sender") or "",
                "text": item.get("text") or "",
                "previous_turn": _first_present(item.get("previous_turn"), item.get("previous_text"), item.get("prev_text")) or "",
                "ground_truth": _label_text_only(item.get("ground_truth")),
                "prediction_label": prediction,
                "prediction_int": _first_present(item.get("prediction_int"), _prediction_int_from_label(prediction)),
                "uncertainty": _owner_decimal(item.get("uncertainty")),
                "review_target": bool(target),
                "review_priority_rank": target.get("selection_rank") if target else None,
                "active_learning_state": _clean_review_state(target, feedback),
                "feedback_label": feedback_label,
                "feedback_explanation": _safe_str((feedback or {}).get("feedback_explanation") or (feedback or {}).get("explanation")),
                "reviewed_by": _reviewed_by_from_feedback(feedback, target),
                "reviewed_at": _reviewed_at_from_feedback(feedback, target),
                "final_label": final_label,
                "model_used": model_used,
                "model_version": model_version,
                "sample_id": sample_id,
                "source_id": _safe_str(item.get("source_id") or item.get("row_id") or item.get("conversation_id") or item.get("dialogue_id")),
                "dataset_version": dataset_version,
                "created_at": _safe_str(item.get("created_at") or item.get("timestamp")) or created_at,
                "updated_at": _safe_str((feedback or {}).get("updated_at") or (feedback or {}).get("submitted_at") or item.get("updated_at") or item.get("timestamp")) or created_at
            }
        enhanced.append(clean_row)

    enhanced.sort(key=lambda row: (
        row.get("review_priority_rank") is None,
        int(row.get("review_priority_rank") or 0),
        _owner_decimal(row.get("uncertainty")) is None,
        _owner_decimal(row.get("uncertainty")) if _owner_decimal(row.get("uncertainty")) is not None else 1.0,
        _safe_str(row.get("sample_id"))
    ))
    return {
        "ok": True,
        "project_id": project_id,
        "version_id": version_id,
        "metadata": metadata,
        "row_count": len(enhanced),
        "final_label_policy": "Rows without submitted or locked feedback keep the model prediction as final_label.",
        "rows": enhanced
    }


def _enhanced_dataset_columns(project_type):
    if project_type == "news":
        return [
            "article_id", "chunk_id", "chunk_index", "title", "text",
            "ground_truth", "prediction_label", "prediction_int", "uncertainty",
            "review_target", "review_priority_rank", "active_learning_state",
            "feedback_label", "feedback_explanation", "reviewed_by", "reviewed_at",
            "final_label", "model_used", "model_version", "sample_id", "source_id",
            "dataset_version", "created_at", "updated_at"
        ]
    return [
        "dialogue_id", "turn_index", "sender", "text", "previous_turn",
        "ground_truth", "prediction_label", "prediction_int", "uncertainty",
        "review_target", "review_priority_rank", "active_learning_state",
        "feedback_label", "feedback_explanation", "reviewed_by", "reviewed_at",
        "final_label", "model_used", "model_version", "sample_id", "source_id",
        "dataset_version", "created_at", "updated_at"
    ]


def _reopened_rows_from_enhanced(enhanced_rows, new_version_id, created_at):
    rows = []
    for row in enhanced_rows or []:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        final_label = _label_text_only(item.get("final_label") or item.get("prediction_label") or item.get("ground_truth"))
        item["ground_truth"] = final_label
        item["prediction_label"] = ""
        item["prediction_int"] = None
        item["feedback_label"] = None
        item["final_label"] = ""
        item["review_target"] = False
        item["active_learning_state"] = "pending"
        item["dataset_version"] = new_version_id
        item["created_at"] = created_at
        item["updated_at"] = created_at
        for key in ("feedback_explanation", "reviewed_by", "reviewed_at"):
            item.pop(key, None)
        rows.append(item)
    return rows


def _review_candidates_from_rows(rows, project_type):
    candidates = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        sample_id = _safe_str(row.get("sample_id"))
        if not sample_id:
            continue
        target_unit = "chunk" if project_type == "news" else "turn"
        candidate = {
            "sample_id": sample_id,
            "source_id": row.get("source_id") or row.get("article_id") or row.get("dialogue_id") or sample_id,
            "target_unit": target_unit,
            "prediction": row.get("prediction_label") or row.get("prediction"),
            "prediction_int": row.get("prediction_int"),
            "ground_truth": row.get("ground_truth"),
            "confidence": row.get("confidence"),
            "uncertainty": _owner_decimal(row.get("uncertainty")) if _owner_decimal(row.get("uncertainty")) is not None else 0.5,
            "title": row.get("title"),
            "text": row.get("text"),
            "article_id": row.get("source_id") or row.get("article_id"),
            "chunk_id": row.get("chunk_id"),
            "chunk_index": row.get("chunk_index"),
            "dialogue_id": row.get("dialogue_id"),
            "turn_index": row.get("turn_index"),
            "row_id": row.get("row_id") or row.get("source_id")
        }
        candidates.append(candidate)
    return candidates


def _project_owner_required(project_id):
    if not session.get("idToken"):
        return None, (jsonify({"error": "Unauthorized"}), 401)
    project = get_project_basic_info(project_id)
    if not project:
        return None, (jsonify({"error": "Project not found"}), 404)
    if project.get("owner_id") != session.get("uid"):
        return None, (jsonify({"error": "Forbidden"}), 403)
    return project, None


def _reset_labeling_tasks_for_reopen(project_id, version_id):
    updated = []
    for doc in db.collection("tasks").where("project_ID", "==", project_id).where("task_type", "==", "labeling").stream():
        db.collection("tasks").document(doc.id).update({
            "status": "pending",
            "active_version": version_id,
            "reopened_at": _now_utc_iso()
        })
        updated.append(doc.id)
    return updated


def _metric_value(metrics, key):
    return metrics.get(key) if isinstance(metrics, dict) and metrics.get("available") else None


def _evaluation_from_rows(rows, project_type, level_name):
    y_true = []
    y_pred = []
    for row in rows or []:
        true_value = _normalize_binary_label(row.get("ground_truth"))
        pred_value = _normalize_binary_label(row.get("prediction"))
        if true_value is not None and pred_value is not None:
            y_true.append(true_value)
            y_pred.append(pred_value)
    metrics = _compute_standard_metrics(y_true, y_pred)
    perf = _owner_performance_details(metrics if isinstance(metrics, dict) else {})
    support = metrics.get("support") if isinstance(metrics, dict) else {}
    return {
        "level": level_name,
        "available": bool(metrics.get("available")) if isinstance(metrics, dict) else False,
        "accuracy": _metric_value(metrics, "accuracy"),
        "precision_macro": _metric_value(metrics, "precision_macro"),
        "recall_macro": _metric_value(metrics, "recall_macro"),
        "f1_macro": _metric_value(metrics, "f1_macro"),
        "precision_ai": _metric_value(metrics, "precision_ai"),
        "recall_ai": _metric_value(metrics, "recall_ai"),
        "f1_ai": _metric_value(metrics, "f1_ai"),
        "false_positive_rate": metrics.get("false_positive_rate") if isinstance(metrics, dict) else None,
        "false_negative_rate": metrics.get("false_negative_rate") if isinstance(metrics, dict) else None,
        "confusion_matrix": metrics.get("confusion_matrix") if isinstance(metrics, dict) else None,
        "support_human": (support or {}).get("human"),
        "support_ai": (support or {}).get("ai"),
        "total_errors": perf.get("total_errors"),
        "correct_predictions": perf.get("correct_predictions"),
        "reason": metrics.get("reason") if isinstance(metrics, dict) else None
    }


def _evaluation_metrics_for_label(rows, prediction_key, level_name, evaluation_level):
    y_true = []
    y_pred = []
    for row in rows or []:
        true_value = _normalize_binary_label((row or {}).get("ground_truth"))
        pred_value = _normalize_binary_label((row or {}).get(prediction_key))
        if true_value is not None and pred_value is not None:
            y_true.append(true_value)
            y_pred.append(pred_value)

    metrics = _compute_standard_metrics(y_true, y_pred)
    support = metrics.get("support") if isinstance(metrics, dict) else {}
    cm = metrics.get("confusion_matrix") if isinstance(metrics, dict) else None
    total = int((support or {}).get("total_labeled") or 0)
    error_count = None
    if cm:
        values = _owner_confusion_values(cm)
        if values:
            error_count = int(values["fp"]) + int(values["fn"])
    error_rate = (float(error_count) / total) if total and error_count is not None else None

    return {
        "level": level_name,
        "evaluation_level": evaluation_level,
        "prediction_source": prediction_key,
        "available": bool(metrics.get("available")) if isinstance(metrics, dict) else False,
        "accuracy": _metric_value(metrics, "accuracy"),
        "macro_precision": _metric_value(metrics, "precision_macro"),
        "macro_recall": _metric_value(metrics, "recall_macro"),
        "macro_f1": _metric_value(metrics, "f1_macro"),
        "ai_precision": _metric_value(metrics, "precision_ai"),
        "ai_recall": _metric_value(metrics, "recall_ai"),
        "ai_f1": _metric_value(metrics, "f1_ai"),
        "false_positive_rate": metrics.get("false_positive_rate") if isinstance(metrics, dict) else None,
        "false_negative_rate": metrics.get("false_negative_rate") if isinstance(metrics, dict) else None,
        "confusion_matrix": cm,
        "support_human": (support or {}).get("human"),
        "support_ai": (support or {}).get("ai"),
        "support_total": total,
        "error_rate": error_rate,
        "error_count": error_count,
        "reason": metrics.get("reason") if isinstance(metrics, dict) else None
    }


def _metric_delta(enhanced, baseline, key):
    enhanced_value = (enhanced or {}).get(key)
    baseline_value = (baseline or {}).get(key)
    if enhanced_value is None or baseline_value is None:
        return None
    try:
        return float(enhanced_value) - float(baseline_value)
    except (TypeError, ValueError):
        return None


def _metrics_comparison(baseline, enhanced):
    return {
        "accuracy_delta": _metric_delta(enhanced, baseline, "accuracy"),
        "macro_f1_delta": _metric_delta(enhanced, baseline, "macro_f1"),
        "ai_f1_delta": _metric_delta(enhanced, baseline, "ai_f1"),
        "fpr_delta": _metric_delta(enhanced, baseline, "false_positive_rate"),
        "fnr_delta": _metric_delta(enhanced, baseline, "false_negative_rate"),
        "error_rate_delta": _metric_delta(enhanced, baseline, "error_rate")
    }


def _ai_score_from_row(row, key):
    row = row or {}
    if key != "prediction":
        normalized = _normalize_binary_label(row.get(key))
        if normalized is not None:
            return float(normalized)
    for prob_key in ("ai_probability", "p_machine", "ai", "ai_percentage"):
        value = _owner_decimal(row.get(prob_key))
        if value is not None:
            return value / 100.0 if value > 1 else value
    normalized = _normalize_binary_label(row.get(key))
    if normalized is not None:
        return float(normalized)
    return None


def _label_from_average_score(scores):
    valid = [float(score) for score in scores if score is not None]
    if not valid:
        return ""
    return "AI" if (sum(valid) / len(valid)) >= 0.5 else "Human"


def _derive_group_rows(rows, group_key):
    groups = {}
    for row in rows or []:
        key = _safe_str(row.get(group_key))
        if not key:
            continue
        groups.setdefault(key, []).append(row)
    derived = []
    for key, items in groups.items():
        pred_votes = [_normalize_binary_label(item.get("prediction")) for item in items]
        gt_votes = [_normalize_binary_label(item.get("ground_truth")) for item in items]
        pred_votes = [value for value in pred_votes if value is not None]
        gt_votes = [value for value in gt_votes if value is not None]
        if not pred_votes:
            continue
        pred = 1 if sum(pred_votes) >= (len(pred_votes) / 2.0) else 0
        gt = None
        if gt_votes:
            gt = 1 if sum(gt_votes) >= (len(gt_votes) / 2.0) else 0
        derived.append({
            group_key: key,
            "prediction": _label_text_from_int(pred),
            "ground_truth": _label_text_from_int(gt)
        })
    return derived


def _derive_evaluation_group_rows(rows, group_key):
    groups = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        key = _safe_str(row.get(group_key))
        if not key:
            continue
        groups.setdefault(key, []).append(row)

    derived = []
    for key, items in groups.items():
        baseline_scores = [_ai_score_from_row(item, "prediction_label") for item in items]
        enhanced_scores = [_ai_score_from_row(item, "final_label") for item in items]
        truth_scores = [_ai_score_from_row({"ground_truth": item.get("ground_truth")}, "ground_truth") for item in items]
        derived.append({
            group_key: key,
            "ground_truth": _label_from_average_score(truth_scores),
            "prediction_label": _label_from_average_score(baseline_scores),
            "final_label": _label_from_average_score(enhanced_scores),
            "row_count": len(items)
        })
    return derived


def _evaluation_pair(rows, level_name, evaluation_level):
    baseline = _evaluation_metrics_for_label(rows, "prediction_label", level_name, evaluation_level)
    enhanced = _evaluation_metrics_for_label(rows, "final_label", level_name, evaluation_level)
    return {
        "evaluation_level": evaluation_level,
        "baseline_metrics": baseline,
        "enhanced_metrics": enhanced,
        "comparison": _metrics_comparison(baseline, enhanced)
    }


def _review_statistics(enhanced_rows):
    review_rows = [row for row in enhanced_rows or [] if row.get("review_target")]
    feedback_rows = [
        row for row in enhanced_rows or []
        if row.get("active_learning_state") in ("submitted", "locked") and _label_text_only(row.get("feedback_label"))
    ]
    corrected_rows = [
        row for row in feedback_rows
        if _label_text_only(row.get("feedback_label")) != _label_text_only(row.get("prediction_label"))
    ]
    reviewed_targets = len(feedback_rows)
    corrected_targets = len(corrected_rows)
    return {
        "total_review_targets": len(review_rows),
        "reviewed_targets": reviewed_targets,
        "pending_targets": max(len(review_rows) - reviewed_targets, 0),
        "corrected_targets": corrected_targets,
        "correction_rate": (float(corrected_targets) / reviewed_targets) if reviewed_targets else 0.0,
        "human_to_ai_corrections": sum(1 for row in corrected_rows if _label_text_only(row.get("prediction_label")) == "Human" and _label_text_only(row.get("feedback_label")) == "AI"),
        "ai_to_human_corrections": sum(1 for row in corrected_rows if _label_text_only(row.get("prediction_label")) == "AI" and _label_text_only(row.get("feedback_label")) == "Human")
    }


def _examiner_impact_from_feedback(project_id, version_id, detection_rows):
    feedback_map = _load_version_feedback(project_id, version_id)
    row_map = {_snapshot_row_key(row.get("sample_id")): row for row in detection_rows or [] if isinstance(row, dict)}
    examiners = {}

    for key, feedback in (feedback_map.items() if isinstance(feedback_map, dict) else []):
        if not isinstance(feedback, dict):
            continue
        if not _feedback_is_final(feedback):
            continue
        corrected_label = _label_text_only(feedback.get("corrected_label"))
        if not corrected_label:
            continue
        examiner_uid = _safe_str(feedback.get("examiner_uid") or "unknown")
        examiner = examiners.setdefault(examiner_uid, {
            "examiner_uid": examiner_uid,
            "examiner_name": _safe_str(feedback.get("examiner_name")) or _short_uid(examiner_uid),
            "reviewed_count": 0,
            "corrected_count": 0,
            "correction_rate": 0.0
        })
        examiner["reviewed_count"] += 1
        prediction = _label_text_only((row_map.get(key) or {}).get("prediction_label") or (row_map.get(key) or {}).get("prediction"))
        if prediction and prediction != corrected_label:
            examiner["corrected_count"] += 1

    result = []
    for item in examiners.values():
        reviewed_count = int(item.get("reviewed_count") or 0)
        corrected_count = int(item.get("corrected_count") or 0)
        item["correction_rate"] = (float(corrected_count) / reviewed_count) if reviewed_count else 0.0
        result.append(item)
    result.sort(key=lambda item: (-int(item.get("reviewed_count") or 0), _safe_str(item.get("examiner_name"))))
    return result


def _evaluation_snapshot_complete(snapshot):
    if not isinstance(snapshot, dict) or not snapshot:
        return False
    if snapshot.get("stale") or snapshot.get("rebuild_required"):
        return False
    required = (
        "model_name",
        "version_id",
        "project_type",
        "baseline_metrics",
        "enhanced_metrics",
        "comparison",
        "review_statistics",
        "examiner_impact"
    )
    if any(key not in snapshot for key in required):
        return False
    if not isinstance(snapshot.get("baseline_metrics"), dict) or not snapshot.get("baseline_metrics"):
        return False
    if not isinstance(snapshot.get("enhanced_metrics"), dict) or not snapshot.get("enhanced_metrics"):
        return False
    project_type = _safe_str(snapshot.get("project_type"))
    if project_type == "news":
        return isinstance(snapshot.get("chunk_level_metrics"), dict) and isinstance(snapshot.get("article_level_metrics"), dict)
    return isinstance(snapshot.get("turn_level_metrics"), dict) and isinstance(snapshot.get("dialogue_level_metrics"), dict)


def _build_evaluation_snapshot(project_id, version_id, rebuild=False):
    if _version_is_closed(project_id, version_id):
        existing_closed = rtdb.reference(f"evaluation_snapshots/{project_id}/{version_id}").get()
        if _evaluation_snapshot_complete(existing_closed):
            return existing_closed
    if not rebuild:
        existing = rtdb.reference(f"evaluation_snapshots/{project_id}/{version_id}").get()
        if _evaluation_snapshot_complete(existing):
            return existing

    metadata, rows = _load_detection_version(project_id, version_id)
    metadata, rows = _ensure_detection_rows_for_evaluation(project_id, version_id, metadata, rows)
    if not rows:
        context = _evaluation_snapshot_debug_context(project_id, version_id, metadata, rows)
        raise ValueError(f"Detection version rows are missing. Submit/finalize detection before building evaluation. Context: {context}")
    project_type = metadata.get("project_type")
    enhanced = _build_enhanced_dataset(project_id, version_id).get("rows") or rows

    primary_level = "chunk" if project_type == "news" else "turn"
    primary_pair = _evaluation_pair(enhanced, primary_level, "direct")
    review_stats = _review_statistics(enhanced)
    existing_version_snapshot = rtdb.reference(_version_path(project_id, version_id)).child("evaluation_snapshot").get()
    created_at = _safe_str((existing_version_snapshot or {}).get("created_at")) if isinstance(existing_version_snapshot, dict) else ""
    now_iso = _now_utc_iso()

    snapshot = {
        "project_id": project_id,
        "version_id": version_id,
        "project_type": project_type,
        "model_key": metadata.get("model_key"),
        "model_name": metadata.get("model_name"),
        "created_at": created_at or now_iso,
        "rebuilt_at": now_iso if rebuild and created_at else None,
        "computed_at": now_iso,
        "primary_level": primary_level,
        "baseline_metrics": primary_pair["baseline_metrics"],
        "enhanced_metrics": primary_pair["enhanced_metrics"],
        "comparison": primary_pair["comparison"],
        "review_statistics": review_stats,
        "examiner_impact": _examiner_impact_from_feedback(project_id, version_id, rows),
        "reviewed_targets": review_stats["reviewed_targets"],
        "corrected_labels": review_stats["corrected_targets"],
        "corrected_targets": review_stats["corrected_targets"],
        "human_to_ai_corrections": review_stats["human_to_ai_corrections"],
        "ai_to_human_corrections": review_stats["ai_to_human_corrections"],
        "derived_chunk_metrics": False,
        "derived_article_metrics": project_type == "news",
        "derived_turn_metrics": False,
        "derived_dialogue_metrics": project_type in ("uploaded_conversation", "generated_conversation")
    }

    if project_type == "news":
        article_rows = _derive_evaluation_group_rows(enhanced, "article_id")
        snapshot["chunk_level_metrics"] = primary_pair
        snapshot["article_level_metrics"] = _evaluation_pair(article_rows, "article", "derived")
    else:
        dialogue_rows = _derive_evaluation_group_rows(enhanced, "dialogue_id")
        snapshot["turn_level_metrics"] = primary_pair
        snapshot["dialogue_level_metrics"] = _evaluation_pair(dialogue_rows, "dialogue", "derived")

    rtdb.reference(_version_path(project_id, version_id)).child("evaluation_snapshot").set(_owner_json_value(snapshot))
    rtdb.reference(f"evaluation_snapshots/{project_id}/{version_id}").set(_owner_json_value(snapshot))
    if not _version_is_closed(project_id, version_id):
        _update_project_version_status(project_id, version_id, "evaluation_ready", True)
    return snapshot


def _feedback_explanation_item(row):
    if not isinstance(row, dict):
        return None
    explanation = _safe_str(row.get("feedback_explanation") or row.get("explanation"))
    if not explanation:
        return None
    text = _safe_str(row.get("text"))
    prediction = _safe_str(row.get("model_prediction_label") or row.get("model_prediction") or row.get("prediction"))
    corrected_label = _safe_str(row.get("corrected_label_text") or row.get("corrected_label") or row.get("label"))
    agreed_with_model = row.get("agreed_with_model")
    if agreed_with_model is None and prediction and corrected_label:
        agreed_with_model = prediction.strip().lower() == corrected_label.strip().lower()

    return {
        "sample_id": _safe_str(row.get("sample_id") or row.get("source_id")),
        "project_type": _safe_str(row.get("project_type")),
        "title": _safe_str(row.get("title") or row.get("dialogue_id") or row.get("conversation_id") or row.get("task_id")),
        "text_preview": text[:220],
        "prediction": prediction,
        "corrected_label": corrected_label,
        "examiner_uid": _safe_str(row.get("examiner_uid") or row.get("uid") or row.get("examiner_id")),
        "examiner_name": _safe_str(row.get("examiner_name") or row.get("name")) or "Examiner",
        "feedback_explanation": explanation,
        "submitted_at": _safe_str(row.get("submitted_at")),
        "confidence": _owner_decimal(row.get("confidence")),
        "uncertainty": _owner_decimal(row.get("uncertainty")),
        "agreed_with_model": bool(agreed_with_model),
        "correction_changed_prediction": bool(prediction and corrected_label and prediction.strip().lower() != corrected_label.strip().lower()),
        "participation_status": _safe_str(row.get("participation_status")),
        "examiner_rating": row.get("examiner_rating")
    }


def _normalized_feedback_explanations(project_id, limit=20):
    raw = rtdb.reference(f"active_learning_feedback/{project_id}").get() or {}
    rows = list(raw.values()) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    items = []
    for row in rows:
        item = _feedback_explanation_item(row)
        if item:
            items.append(item)
    items.sort(key=lambda item: item.get("submitted_at", ""), reverse=True)
    return items[:limit]


def _old_feedback_explanations(project_id, project, result_type, limit=20):
    items = []
    try:
        if result_type == "news":
            dataset_id = project.get("dataset_id") or ""
            rows = rtdb.reference(f"datasets/uploaded_news/{dataset_id}").get() or {}
            selected = _get_project_selected_model(project_id, project, get_project_tasks(project_id))
            details = (_news_results_payload(project_id, selected.get("key") or "logistic") or {}).get("details") or []
            details = list(details.values()) if isinstance(details, dict) else (details if isinstance(details, list) else [])
            detail_map = {_safe_str(item.get("article_id") or item.get("id")): item for item in details if isinstance(item, dict)}
            for article_id, source in (rows.items() if isinstance(rows, dict) else []):
                source = source if isinstance(source, dict) else {}
                payload = source.get("payload") if isinstance(source.get("payload"), dict) else source
                detail = detail_map.get(_safe_str(article_id), {})
                title, text = _text_from_news_payload(payload if isinstance(payload, dict) else {}, detail)
                feedbacks = _feedback_items(source.get("feedback") or {})
                feedbacks.extend(_feedback_items(source.get("examiner_feedbacks") or {}))
                for feedback in feedbacks:
                    item = _feedback_explanation_item({
                        "sample_id": _make_active_learning_sample_id("news", article_id=article_id),
                        "project_type": "news",
                        "title": title,
                        "text": text,
                        "model_prediction_label": detail.get("prediction"),
                        "corrected_label_text": feedback.get("label") if feedback.get("agreed_with_model") is False else detail.get("prediction"),
                        "examiner_uid": feedback.get("examiner_uid"),
                        "examiner_name": feedback.get("examiner_name"),
                        "feedback_explanation": feedback.get("explanation"),
                        "submitted_at": feedback.get("submitted_at"),
                        "confidence": detail.get("confidence"),
                        "uncertainty": detail.get("uncertainty")
                    })
                    if item:
                        items.append(item)
        elif result_type == "uploaded_conversation":
            selected = _get_project_selected_model(project_id, project, get_project_tasks(project_id))
            model_key = _uploaded_feedback_model_key(selected.get("key") or CONV_LOGREG_KEY)
            _, run_ref, _ = _uploaded_conversation_run(project_id, model_key)
            if run_ref:
                all_turns = run_ref.child("dialogue_turns").get() or {}
                feedback_root = run_ref.child("turn_feedbacks").get() or {}
                for safe_key, turns_raw in (all_turns.items() if isinstance(all_turns, dict) else []):
                    turns = _turns_to_list(turns_raw)
                    for idx, turn in enumerate(turns):
                        if not isinstance(turn, dict):
                            continue
                        ui_turn_index = idx + 1
                        turn_feedbacks = feedback_root.get(safe_key, {}).get(str(ui_turn_index), {}) if isinstance(feedback_root, dict) else {}
                        for feedback in _feedback_items(turn_feedbacks):
                            item = _feedback_explanation_item({
                                "sample_id": _make_active_learning_sample_id("uploaded_conversation", row_id=turn.get("row_id") or turn.get("source_row_id"), dialogue_id=turn.get("dialogue_id") or safe_key, turn_index=ui_turn_index),
                                "project_type": "uploaded_conversation",
                                "dialogue_id": turn.get("dialogue_id") or safe_key,
                                "text": turn.get("text"),
                                "model_prediction_label": "AI" if _is_ai_label(turn.get("prediction")) else "Human",
                                "corrected_label_text": feedback.get("label") if feedback.get("agreed_with_model") is False else ("AI" if _is_ai_label(turn.get("prediction")) else "Human"),
                                "examiner_uid": feedback.get("examiner_uid"),
                                "examiner_name": feedback.get("examiner_name"),
                                "feedback_explanation": feedback.get("explanation"),
                                "submitted_at": feedback.get("submitted_at"),
                                "confidence": turn.get("confidence"),
                                "uncertainty": turn.get("uncertainty")
                            })
                            if item:
                                items.append(item)
        elif result_type == "generated_conversation":
            selected = _get_project_selected_model(project_id, project, get_project_tasks(project_id))
            raw = _generated_conversation_results_payload(project_id, selected.get("key") or CONV_LOGREG_KEY) or {}
            for task_id, node in (raw.items() if isinstance(raw, dict) else []):
                if not isinstance(node, dict):
                    continue
                turns = _turns_to_list(node.get("turns") or {})
                feedback_root = node.get("turn_feedbacks") or {}
                for turn in turns:
                    if not isinstance(turn, dict):
                        continue
                    turn_index = int(turn.get("turn_index", 0) or 0)
                    turn_feedbacks = feedback_root.get(str(turn_index), {}) if isinstance(feedback_root, dict) else {}
                    for feedback in _feedback_items(turn_feedbacks):
                        item = _feedback_explanation_item({
                            "sample_id": _make_active_learning_sample_id("generated_conversation", task_id=task_id, conversation_id=task_id, turn_index=turn_index),
                            "project_type": "generated_conversation",
                            "conversation_id": task_id,
                            "text": turn.get("text"),
                            "model_prediction_label": "AI" if _is_ai_label(turn.get("prediction")) else "Human",
                            "corrected_label_text": feedback.get("label") if feedback.get("agreed_with_model") is False else ("AI" if _is_ai_label(turn.get("prediction")) else "Human"),
                            "examiner_uid": feedback.get("examiner_uid"),
                            "examiner_name": feedback.get("examiner_name"),
                            "feedback_explanation": feedback.get("explanation"),
                            "submitted_at": feedback.get("submitted_at"),
                            "confidence": turn.get("confidence"),
                            "uncertainty": turn.get("uncertainty")
                        })
                        if item:
                            items.append(item)
    except Exception as e:
        app.logger.exception("Failed to build fallback feedback explanations: %s", e)
    items.sort(key=lambda item: item.get("submitted_at", ""), reverse=True)
    return items[:limit]


def _owner_feedback_explanations(project_id, project, result_type, limit=20):
    items = _normalized_feedback_explanations(project_id, limit)
    if items:
        return items
    return _old_feedback_explanations(project_id, project, result_type, limit)


def _project_logistic_task_model(project_id):
    for task in db.collection("tasks").where("project_ID", "==", project_id).stream():
        data = task.to_dict() or {}
        if data.get("task_type") == "model_selection" and data.get("selected_model"):
            return data.get("selected_model")
    return None


def _export_news_active_learning_rows(project_id, proj_data):
    selected_model = _project_logistic_task_model(project_id)
    if not _uses_frozen_active_learning(selected_model):
        return []
    dataset_id = proj_data.get("dataset_id")
    if not dataset_id:
        return []

    safe_pid = project_id.replace(".", "_").replace("#", "_").replace("$", "_").replace("[", "_").replace("]", "_")
    results_data = rtdb.reference(f"analysis_results/{safe_pid}/{selected_model}").get() or {}
    details = results_data.get("details") or []
    if not isinstance(details, list):
        details = list(details.values()) if isinstance(details, dict) else []

    dataset_rows = rtdb.reference(f"datasets/uploaded_news/{dataset_id}").get() or {}
    detail_map = {
        _safe_str(item.get("article_id") or item.get("id")): item
        for item in details
        if isinstance(item, dict)
    }
    model_version = _active_learning_model_version(project_id, selected_model)
    frozen_selection = _freeze_active_learning_targets_if_missing(
        project_id,
        "news",
        selected_model,
        model_version,
        _news_active_learning_candidates(project_id, selected_model, details)
    )
    rows = []

    for target in sorted(_frozen_target_values(frozen_selection), key=lambda item: int(item.get("selection_rank") or 0)):
        article_id = _safe_str(target.get("article_id") or target.get("parent_article_id") or target.get("source_id"))
        if ":chunk:" in article_id:
            article_id = article_id.split(":chunk:", 1)[0]
        source = dataset_rows.get(article_id) or {}
        payload = source.get("payload", {}) if isinstance(source, dict) else {}
        detail = detail_map.get(article_id, {})
        title = payload.get("title") or payload.get("Title") or detail.get("title", "")
        article = payload.get("Article") or payload.get("article") or payload.get("content") or detail.get("content", "")
        target_unit = _safe_str(target.get("target_unit") or "article")
        chunk_index = target.get("chunk_index")
        chunk = _news_chunk_by_index(detail, chunk_index) if target_unit == "chunk" else None
        feedback = _news_first_chunk_feedback(source, chunk_index) if target_unit == "chunk" else (source.get("feedback", {}) if isinstance(source, dict) else {})
        model_prediction = (chunk or {}).get("prediction") if chunk else detail.get("prediction")
        label = feedback.get("label") if isinstance(feedback, dict) and not feedback.get("agreed_with_model") else model_prediction
        label_int = _label_to_int(label)

        rows.append({
            "project_id": project_id,
            "dataset_id": dataset_id,
            "article_id": article_id,
            "sample_id": target.get("sample_id"),
            "selection_run_id": target.get("selection_run_id"),
            "selection_rank": target.get("selection_rank"),
            "target_status": target.get("target_status"),
            "target_unit": target_unit,
            "chunk_index": chunk_index,
            "chunk_id": target.get("chunk_id"),
            "active_learning_selected": True,
            "title": title,
            "Article": article,
            "text": (chunk or {}).get("chunk_text") or target.get("chunk_text") or target.get("text") or (f"{title}. {article}" if title else article),
            "parent_article_prediction": detail.get("prediction"),
            "model_prediction": model_prediction,
            "MachineGen": label_int,
            "label": label_int,
            "feedback_label": feedback.get("label") if isinstance(feedback, dict) else None,
            "agreed_with_model": bool(feedback.get("agreed_with_model", False)) if isinstance(feedback, dict) else None,
            "confidence": (chunk or {}).get("confidence") if chunk else detail.get("confidence"),
            "uncertainty": (chunk or {}).get("uncertainty") if chunk else detail.get("uncertainty"),
            "examiner_uid": feedback.get("examiner_uid") if isinstance(feedback, dict) else None,
            "submitted_at": feedback.get("submitted_at") if isinstance(feedback, dict) else None
        })

    return rows
    return rows


def _first_turn_feedback(feedback_root, turn_index):
    if not isinstance(feedback_root, dict):
        return None
    turn_feedbacks = feedback_root.get(str(turn_index), {}) or {}
    if not isinstance(turn_feedbacks, dict) or not turn_feedbacks:
        return None
    return next(iter(turn_feedbacks.values()))


def _export_generated_conversation_active_learning_rows(project_id, model_key=CONV_LOGREG_KEY):
    raw = _generated_conversation_results_payload(project_id, model_key) or {}
    if not isinstance(raw, dict):
        return []

    model_version = _active_learning_model_version(project_id, model_key)
    frozen_selection = _freeze_active_learning_targets_if_missing(
        project_id,
        "generated_conversation",
        model_key,
        model_version,
        _generated_conversation_active_learning_candidates(project_id, model_key, raw)
    )
    target_ids = {
        _safe_str(target.get("source_id") or f"{target.get('conversation_id')}:{target.get('turn_index')}"): target
        for target in _frozen_target_values(frozen_selection)
    }

    rows = []
    for node_key, node_val in raw.items():
        node = node_val or {}
        meta = node.get("meta", {}) or {}
        conversation_id = meta.get("task_id") or node_key
        turns_raw = node.get("turns", {}) or {}
        turns = list(turns_raw.values()) if isinstance(turns_raw, dict) else (turns_raw if isinstance(turns_raw, list) else [])
        feedback_root = node.get("turn_feedbacks") or {}

        for turn in turns:
            if not isinstance(turn, dict):
                continue
            turn_index = int(turn.get("turn_index", 0) or 0)
            target = target_ids.get(f"{conversation_id}:{turn_index}")
            if not target:
                continue
            feedback = _first_turn_feedback(feedback_root, turn_index)
            label_int = _label_to_int(feedback.get("label")) if feedback else None

            rows.append({
                "project_id": project_id,
                "sample_id": target.get("sample_id"),
                "selection_run_id": target.get("selection_run_id"),
                "selection_rank": target.get("selection_rank"),
                "target_status": target.get("target_status"),
                "active_learning_selected": True,
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "text": turn.get("text", ""),
                "prev_text": turn.get("prev_text", ""),
                "MachineGen": label_int,
                "label": label_int,
                "feedback_label": feedback.get("label") if feedback else None,
                "agreed_with_model": bool(feedback.get("agreed_with_model", False)) if feedback else None,
                "confidence": turn.get("confidence"),
                "uncertainty": turn.get("uncertainty"),
                "examiner_uid": feedback.get("examiner_uid") if feedback else None,
                "submitted_at": feedback.get("submitted_at") if feedback else None
            })

    rows.sort(key=lambda item: int(item.get("selection_rank") or 0))
    return rows


def _export_uploaded_conversation_active_learning_rows(project_id, model_key=CONV_LOGREG_KEY):
    run_id, run_ref, _ = _uploaded_conversation_run(project_id, model_key)
    if not run_ref:
        return []

    dialogue_turns = run_ref.child("dialogue_turns").get() or {}
    feedback_root = run_ref.child("turn_feedbacks").get() or {}
    key_map = run_ref.child("dialogue_key_map").get() or {}
    reverse_key_map = {v: k for k, v in key_map.items()} if isinstance(key_map, dict) else {}
    model_version = _active_learning_model_version(project_id, model_key)
    frozen_selection = _freeze_active_learning_targets_if_missing(
        project_id,
        "uploaded_conversation",
        model_key,
        model_version,
        _uploaded_conversation_active_learning_candidates(project_id, model_key, run_id, run_ref)
    )
    target_keys = {
        (_safe_str(target.get("safe_key")), int(target.get("turn_index") or 0)): target
        for target in _frozen_target_values(frozen_selection)
    }

    rows = []
    for safe_key, turns_raw in (dialogue_turns.items() if isinstance(dialogue_turns, dict) else []):
        turns = list(turns_raw.values()) if isinstance(turns_raw, dict) else (turns_raw if isinstance(turns_raw, list) else [])
        dialogue_id = reverse_key_map.get(safe_key, safe_key)
        dialogue_feedbacks = feedback_root.get(safe_key, {}) if isinstance(feedback_root, dict) else {}

        for idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            turn_index = idx + 1
            target = target_keys.get((safe_key, turn_index))
            if not target:
                continue
            feedback = _first_turn_feedback(dialogue_feedbacks, turn_index)
            label_int = _label_to_int(feedback.get("label")) if feedback else None

            rows.append({
                "project_id": project_id,
                "run_id": run_id,
                "sample_id": target.get("sample_id"),
                "selection_run_id": target.get("selection_run_id"),
                "selection_rank": target.get("selection_rank"),
                "target_status": target.get("target_status"),
                "active_learning_selected": True,
                "dialogue_id": dialogue_id,
                "turn_index": turn.get("turn_index", turn_index),
                "text": turn.get("text", ""),
                "prev_text": turn.get("previous_text", ""),
                "MachineGen": label_int,
                "label": label_int,
                "feedback_label": feedback.get("label") if feedback else None,
                "agreed_with_model": bool(feedback.get("agreed_with_model", False)) if feedback else None,
                "confidence": turn.get("confidence"),
                "uncertainty": turn.get("uncertainty"),
                "examiner_uid": feedback.get("examiner_uid") if feedback else None,
                "submitted_at": feedback.get("submitted_at") if feedback else None
            })

    rows.sort(key=lambda item: int(item.get("selection_rank") or 0))
    return rows


# =========================
# Owner Results Summary Helpers
# =========================
# These helpers build a read-only summary from the Firebase paths already used by
# analysis and feedback pages. They do not create or update any Firebase data.

def _owner_empty_summary():
    return {
        "total_items": 0,
        "total_turns": 0,
        "human_count": 0,
        "ai_count": 0,
        "avg_confidence": 0,
        "avg_uncertainty": 0,
        "review_targets": 0,
        "reviewed_targets": 0,
        "corrected_labels": 0
    }


def _owner_empty_metrics():
    return {
        "available": False,
        "reason": "Ground truth labels are missing or invalid.",
        "accuracy": None,
        "precision_macro": None,
        "recall_macro": None,
        "f1_macro": None,
        "precision_ai": None,
        "recall_ai": None,
        "f1_ai": None,
        "confusion_matrix": None
    }


def _owner_decimal(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if number > 1.0:
        number = number / 100.0
    if number < 0.0:
        number = 0.0
    if number > 1.0:
        number = 1.0
    return number


def _owner_prediction_is_ai(value):
    return _normalize_binary_label(value) == 1


def _owner_model_key_for_results(model_key, result_type):
    normalized = _safe_str(model_key).strip().lower()
    if result_type in ("uploaded_conversation", "generated_conversation"):
        if normalized in ("rnn", CONV_RNN_KEY, "baseline_rnn", "conv_rnn"):
            return CONV_RNN_KEY
        return CONV_LOGREG_KEY
    if normalized in ("rnn",):
        return "rnn"
    if normalized in ("logreg", "tfidf_logreg", "logistic"):
        return "logistic"
    return normalized


def _owner_task_selected_model(tasks, result_type):
    candidates = []
    for task in tasks:
        if task.get("task_type") != "model_selection":
            continue
        model_key = _safe_str(task.get("selected_model")).strip()
        model_name = _safe_str(task.get("selected_model_name")).strip()
        if not model_key and not model_name:
            continue
        candidates.append(task)

    if not candidates:
        return {
            "key": "",
            "name": "",
            "version": "v1",
            "selected_at": "",
            "selected_by": ""
        }

    candidates.sort(key=lambda item: _safe_str(item.get("selected_at")), reverse=True)
    picked = candidates[0]
    key = _owner_model_key_for_results(picked.get("selected_model"), result_type)
    name = _safe_str(picked.get("selected_model_name"))
    if not name:
        name = "RNN" if key == "rnn" else "Logistic Regression"

    return {
        "key": key,
        "name": name,
        "version": "v1",
        "selected_at": _safe_str(picked.get("selected_at")),
        "selected_by": _safe_str(picked.get("selected_by"))
    }


def _owner_feedback_bucket(examiners, feedback):
    if not isinstance(feedback, dict) or not feedback:
        return

    uid = _safe_str(feedback.get("examiner_uid") or feedback.get("uid") or feedback.get("examiner_id"))
    if not uid:
        uid = "unknown"

    name = _safe_str(feedback.get("examiner_name")) or _feedback_examiner_name(uid)
    if uid not in examiners:
        examiners[uid] = {
            "examiner_id": uid,
            "examiner_name": name,
            "feedback_count": 0,
            "corrected_count": 0
        }

    examiners[uid]["feedback_count"] += 1
    if feedback.get("agreed_with_model") is False:
        examiners[uid]["corrected_count"] += 1


def _owner_iter_feedbacks(feedback_node):
    if not isinstance(feedback_node, dict) or not feedback_node:
        return []
    if "examiner_uid" in feedback_node or "agreed_with_model" in feedback_node:
        return [feedback_node]
    return [item for item in feedback_node.values() if isinstance(item, dict)]


def _owner_metrics_from_pairs(y_true, y_pred):
    return _compute_standard_metrics(y_true, y_pred)


def _owner_samples_from_items(items, limit=10):
    samples = []
    for item in items:
        if not isinstance(item, dict):
            continue
        samples.append({
            "id": _safe_str(item.get("id") or item.get("article_id") or item.get("row_id") or item.get("dialogue_id") or item.get("conversation_id")),
            "title": _safe_str(item.get("title") or item.get("task_name") or item.get("dialogue_id")),
            "text": _safe_str(item.get("text") or item.get("content") or item.get("text_preview"))[:220],
            "prediction": _safe_str(item.get("prediction")),
            "confidence": _owner_decimal(item.get("confidence")),
            "uncertainty": _owner_decimal(item.get("uncertainty")),
            "review_target": bool(item.get("review_target", False))
        })
    samples.sort(key=lambda item: (item.get("uncertainty") is None, item.get("uncertainty") if item.get("uncertainty") is not None else 1))
    return samples[:limit]


def _owner_json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_owner_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _owner_json_value(val) for key, val in value.items()}
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return _safe_str(value)


def get_project_basic_info(project_id):
    project_doc = db.collection("projects").document(project_id).get()
    if not project_doc.exists:
        return None

    project = project_doc.to_dict() or {}
    project["project_id"] = project.get("project_id") or project.get("project_ID") or project_id
    return {
        "project_id": project["project_id"],
        "project_name": project.get("project_name", ""),
        "category": project.get("category", ""),
        "dataset_id": project.get("dataset_id", ""),
        "generated_from_scratch": bool(project.get("generated_from_scratch", False)),
        "status": project.get("status", ""),
        "owner_id": project.get("owner_id", "")
    }


def get_project_tasks(project_id):
    tasks = []
    for doc in db.collection("tasks").where("project_ID", "==", project_id).stream():
        data = doc.to_dict() or {}
        data["id"] = doc.id
        data["task_ID"] = data.get("task_ID") or doc.id
        tasks.append(_owner_json_value(data))
    tasks.sort(key=lambda item: _safe_str(item.get("created_at") or item.get("task_ID")))
    return tasks


def detect_project_result_type(project):
    category = _safe_str(project.get("category")).strip().lower()
    if not category:
        return "unknown"
    is_conversation = category in ("conversation", "conversations", "chat", "chats")
    if not is_conversation:
        return "news"
    if bool(project.get("generated_from_scratch", False)):
        return "generated_conversation"
    return "uploaded_conversation"


def normalize_news_results(project_id, project):
    warnings = []
    tasks = get_project_tasks(project_id)
    selected_model = _owner_task_selected_model(tasks, "news")
    model_key = selected_model.get("key") or "logistic"

    safe_pid = project_id.replace(".", "_").replace("#", "_").replace("$", "_").replace("[", "_").replace("]", "_")
    results_data = rtdb.reference(f"analysis_results/{safe_pid}/{model_key}").get()
    if not results_data and safe_pid != project_id:
        results_data = rtdb.reference(f"analysis_results/{project_id}/{model_key}").get()
    if not results_data:
        warnings.append("News analysis result path is missing.")
        results_data = {}

    details = results_data.get("details") or results_data.get("results") or []
    if isinstance(details, dict):
        details = list(details.values())
    if not isinstance(details, list):
        details = []

    dataset_id = project.get("dataset_id")
    dataset_rows = {}
    if dataset_id:
        dataset_rows = rtdb.reference(f"datasets/uploaded_news/{dataset_id}").get() or {}
        if not dataset_rows:
            warnings.append("News dataset path is missing or empty.")
    else:
        warnings.append("Project dataset_id is missing.")

    normalized = []
    confidence_values = []
    uncertainty_values = []
    human_count = 0
    ai_count = 0
    y_true = []
    y_pred = []

    selected_ids = set()
    frozen_by_sample = {}
    if _uses_frozen_active_learning(model_key):
        model_version = _active_learning_model_version(project_id, model_key)
        frozen_selection = _freeze_active_learning_targets_if_missing(
            project_id,
            "news",
            model_key,
            model_version,
            _news_active_learning_candidates(project_id, model_key, details)
        )
        for target in _frozen_target_values(frozen_selection):
            sample_key = _safe_str(target.get("sample_id"))
            if sample_key:
                selected_ids.add(sample_key)
                frozen_by_sample[sample_key] = target

    reviewed_targets = 0
    corrected_labels = 0
    examiners = {}
    reviewed_samples = set()

    for item in details:
        if not isinstance(item, dict):
            continue
        article_id = _safe_str(item.get("article_id") or item.get("id"))
        prediction = item.get("prediction")
        prediction_int = _normalize_binary_label(_first_present(item.get("prediction_int"), prediction))
        if _owner_prediction_is_ai(prediction):
            ai_count += 1
        else:
            human_count += 1

        payload = {}
        if isinstance(dataset_rows, dict) and article_id in dataset_rows and isinstance(dataset_rows[article_id], dict):
            payload = _news_source_payload(dataset_rows[article_id])
        ground_truth = _first_present(item.get("ground_truth"), _extract_ground_truth_from_payload(payload))
        if ground_truth is not None and prediction_int is not None:
            y_true.append(ground_truth)
            y_pred.append(prediction_int)

        confidence = _owner_decimal(item.get("confidence"))
        uncertainty = _owner_decimal(item.get("uncertainty"))
        if confidence is not None:
            confidence_values.append(confidence)
        if uncertainty is not None:
            uncertainty_values.append(uncertainty)

        source = dataset_rows.get(article_id, {}) if isinstance(dataset_rows, dict) and article_id in dataset_rows and isinstance(dataset_rows[article_id], dict) else {}
        feedback = {}
        if isinstance(dataset_rows, dict) and article_id in dataset_rows and isinstance(dataset_rows[article_id], dict):
            feedback = dataset_rows[article_id].get("feedback") or {}
        feedbacks = _owner_iter_feedbacks(feedback)
        for fb in feedbacks:
            _owner_feedback_bucket(examiners, fb)
            if fb.get("agreed_with_model") is False:
                corrected_labels += 1
        for target in frozen_by_sample.values():
            if _safe_str(target.get("article_id")) != article_id:
                continue
            sample_id = _safe_str(target.get("sample_id"))
            chunk_index = target.get("chunk_index")
            if _safe_str(target.get("target_unit")) == "chunk":
                chunk_feedbacks = _owner_iter_feedbacks(_news_chunk_feedbacks(source, chunk_index))
                frozen_reviewed = target.get("target_status") == "reviewed" or int(target.get("feedback_count") or 0) > 0
                if chunk_feedbacks or frozen_reviewed:
                    reviewed_samples.add(sample_id)
                for fb in chunk_feedbacks:
                    _owner_feedback_bucket(examiners, fb)
                    if fb.get("agreed_with_model") is False:
                        corrected_labels += 1
            elif feedbacks or target.get("target_status") == "reviewed" or int(target.get("feedback_count") or 0) > 0:
                reviewed_samples.add(sample_id)

        normalized.append({
            "id": article_id,
            "article_id": article_id,
            "title": item.get("title", ""),
            "content": item.get("content", ""),
            "prediction": prediction,
            "prediction_int": prediction_int,
            "ground_truth": ground_truth,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "review_target": any(_safe_str(target.get("article_id")) == article_id for target in frozen_by_sample.values())
        })

    reviewed_targets = len(reviewed_samples)

    summary = _owner_empty_summary()
    summary.update({
        "total_items": len(normalized),
        "human_count": human_count,
        "ai_count": ai_count,
        "avg_confidence": round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else 0,
        "avg_uncertainty": round(sum(uncertainty_values) / len(uncertainty_values), 4) if uncertainty_values else 0,
        "review_targets": len(selected_ids),
        "reviewed_targets": reviewed_targets,
        "corrected_labels": corrected_labels
    })

    stored_metrics = results_data.get("metrics") or (results_data.get("summary") or {}).get("metrics")
    metrics = stored_metrics if isinstance(stored_metrics, dict) and "available" in stored_metrics else _owner_metrics_from_pairs(y_true, y_pred)

    return {
        "selected_model": selected_model,
        "summary": summary,
        "metrics": metrics,
        "examiners": list(examiners.values()),
        "most_uncertain_samples": _owner_samples_from_items(normalized),
        "warnings": warnings
    }


def normalize_uploaded_conversation_results(project_id, project):
    warnings = []
    tasks = get_project_tasks(project_id)
    selected_model = _owner_task_selected_model(tasks, "uploaded_conversation")
    model_key = selected_model.get("key") or CONV_LOGREG_KEY

    base_ref = rtdb.reference(f"analysis_results/conversations/{model_key}/{project_id}")
    run_id = _safe_str(base_ref.child("latest_run_id").get())
    if not run_id and model_key != CONV_LOGREG_KEY:
        model_key = CONV_LOGREG_KEY
        base_ref = rtdb.reference(f"analysis_results/conversations/{model_key}/{project_id}")
        run_id = _safe_str(base_ref.child("latest_run_id").get())
    if not run_id:
        warnings.append("Uploaded conversation latest_run_id is missing.")
        return {
            "selected_model": selected_model,
            "summary": _owner_empty_summary(),
            "metrics": _owner_empty_metrics(),
            "examiners": [],
            "most_uncertain_samples": [],
            "warnings": warnings
        }

    run_ref = base_ref.child("runs").child(run_id)
    summary_raw = run_ref.child("summary").get() or {}
    dialogue_turns = run_ref.child("dialogue_turns").get() or {}
    feedback_root = run_ref.child("turn_feedbacks").get() or {}
    if not summary_raw:
        warnings.append("Uploaded conversation analysis summary is missing.")

    all_turns = []
    confidence_values = []
    uncertainty_values = []
    human_count = 0
    ai_count = 0
    y_true = []
    y_pred = []

    for safe_key, turns_raw in (dialogue_turns.items() if isinstance(dialogue_turns, dict) else []):
        turns = list(turns_raw.values()) if isinstance(turns_raw, dict) else (turns_raw if isinstance(turns_raw, list) else [])
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            pred_int = turn.get("prediction_int")
            if pred_int is None:
                pred_int = 1 if _owner_prediction_is_ai(turn.get("prediction")) else 0
            pred_int = int(pred_int or 0)
            if pred_int == 1:
                ai_count += 1
            else:
                human_count += 1

            gt_int = _label_to_int(turn.get("ground_truth"))
            if gt_int is not None:
                y_true.append(gt_int)
                y_pred.append(pred_int)

            confidence = _owner_decimal(turn.get("confidence"))
            uncertainty = _owner_decimal(turn.get("uncertainty"))
            if confidence is not None:
                confidence_values.append(confidence)
            if uncertainty is not None:
                uncertainty_values.append(uncertainty)

            all_turns.append({
                "id": turn.get("row_id") or f"{safe_key}:{turn.get('turn_index')}",
                "dialogue_id": turn.get("dialogue_id") or safe_key,
                "row_id": turn.get("row_id"),
                "text": turn.get("text"),
                "prediction": turn.get("prediction"),
                "confidence": confidence,
                "uncertainty": uncertainty,
                "safe_key": safe_key,
                "turn_index": int(turn.get("turn_index", 0) or 0)
            })

    frozen_by_key = {}
    review_target_keys = set()
    if _uses_frozen_active_learning(model_key):
        model_version = _active_learning_model_version(project_id, model_key)
        frozen_selection = _freeze_active_learning_targets_if_missing(
            project_id,
            "uploaded_conversation",
            model_key,
            model_version,
            _uploaded_conversation_active_learning_candidates(project_id, model_key, run_id, run_ref)
        )
        for target in _frozen_target_values(frozen_selection):
            key = (_safe_str(target.get("safe_key")), int(target.get("turn_index") or 0))
            review_target_keys.add(key)
            frozen_by_key[key] = target
    for item in all_turns:
        item["review_target"] = (item.get("safe_key"), item.get("turn_index")) in review_target_keys

    reviewed_targets = 0
    corrected_labels = 0
    examiners = {}
    for safe_key, turns_feedback in (feedback_root.items() if isinstance(feedback_root, dict) else []):
        if not isinstance(turns_feedback, dict):
            continue
        for turn_index, feedbacks_raw in turns_feedback.items():
            feedbacks = _owner_iter_feedbacks(feedbacks_raw)
            target_key = (safe_key, int(turn_index or 0))
            frozen_target = frozen_by_key.get(target_key, {})
            frozen_reviewed = frozen_target.get("target_status") == "reviewed" or int(frozen_target.get("feedback_count") or 0) > 0
            if target_key in review_target_keys and (feedbacks or frozen_reviewed):
                reviewed_targets += 1
            for fb in feedbacks:
                _owner_feedback_bucket(examiners, fb)
                if fb.get("agreed_with_model") is False:
                    corrected_labels += 1

    metrics = summary_raw.get("metrics") if isinstance(summary_raw.get("metrics"), dict) else None
    if not metrics:
        macro = summary_raw.get("macro_metrics") or {}
        metrics = {
            "available": bool(macro),
            "accuracy": macro.get("accuracy"),
            "precision_macro": macro.get("precision_macro"),
            "recall_macro": macro.get("recall_macro"),
            "f1_macro": macro.get("f1_macro"),
            "precision_ai": macro.get("precision_ai"),
            "recall_ai": macro.get("recall_ai"),
            "f1_ai": macro.get("f1_ai"),
            "confusion_matrix": summary_raw.get("confusion_matrix")
        }
    if not metrics.get("available"):
        metrics = _owner_metrics_from_pairs(y_true, y_pred)

    summary = _owner_empty_summary()
    summary.update({
        "total_items": int(summary_raw.get("total_dialogues") or 0),
        "total_turns": len(all_turns),
        "human_count": human_count,
        "ai_count": ai_count,
        "avg_confidence": round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else 0,
        "avg_uncertainty": round(sum(uncertainty_values) / len(uncertainty_values), 4) if uncertainty_values else 0,
        "review_targets": len(review_target_keys),
        "reviewed_targets": reviewed_targets,
        "corrected_labels": corrected_labels
    })

    return {
        "selected_model": selected_model,
        "summary": summary,
        "metrics": metrics,
        "examiners": list(examiners.values()),
        "most_uncertain_samples": _owner_samples_from_items(all_turns),
        "warnings": warnings
    }


def normalize_generated_conversation_results(project_id, project):
    warnings = []
    tasks = get_project_tasks(project_id)
    selected_model = _owner_task_selected_model(tasks, "generated_conversation")
    model_key = selected_model.get("key") or CONV_LOGREG_KEY

    raw = _generated_conversation_results_payload(project_id, model_key)
    if not raw and model_key != CONV_LOGREG_KEY:
        model_key = CONV_LOGREG_KEY
        raw = _generated_conversation_results_payload(project_id, model_key)
    if not isinstance(raw, dict) or not raw:
        warnings.append("Generated conversation analysis result path is missing.")
        raw = {}

    all_turns = []
    confidence_values = []
    uncertainty_values = []
    human_count = 0
    ai_count = 0
    y_true = []
    y_pred = []
    examiners = {}

    for task_id, node in raw.items():
        if not isinstance(node, dict):
            continue
        meta = node.get("meta", {}) or {}
        turns_raw = node.get("turns", {}) or {}
        turns = list(turns_raw.values()) if isinstance(turns_raw, dict) else (turns_raw if isinstance(turns_raw, list) else [])
        feedback_root = node.get("turn_feedbacks") or {}
        task_name = meta.get("task_name") or task_id

        for turn in turns:
            if not isinstance(turn, dict):
                continue
            prediction = turn.get("prediction")
            pred_int = 1 if _owner_prediction_is_ai(prediction) else 0
            if pred_int == 1:
                ai_count += 1
            else:
                human_count += 1

            gt_int = _normalize_binary_label(_first_present(turn.get("ground_truth"), turn.get("gt")))
            if gt_int is not None:
                y_true.append(gt_int)
                y_pred.append(pred_int)

            confidence = _owner_decimal(turn.get("confidence"))
            uncertainty = _owner_decimal(turn.get("uncertainty"))
            if confidence is not None:
                confidence_values.append(confidence)
            if uncertainty is not None:
                uncertainty_values.append(uncertainty)

            turn_index = int(turn.get("turn_index", 0) or 0)
            all_turns.append({
                "id": f"{task_id}:{turn_index}",
                "conversation_id": task_id,
                "task_name": task_name,
                "text": turn.get("text"),
                "prediction": prediction,
                "prediction_int": pred_int,
                "ground_truth": gt_int,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "turn_index": turn_index
            })

            turn_feedbacks = {}
            if isinstance(feedback_root, dict):
                turn_feedbacks = feedback_root.get(str(turn_index), {}) or {}
            for fb in _owner_iter_feedbacks(turn_feedbacks):
                _owner_feedback_bucket(examiners, fb)

    frozen_by_id = {}
    review_target_ids = set()
    if _uses_frozen_active_learning(model_key):
        model_version = _active_learning_model_version(project_id, model_key)
        frozen_selection = _freeze_active_learning_targets_if_missing(
            project_id,
            "generated_conversation",
            model_key,
            model_version,
            _generated_conversation_active_learning_candidates(project_id, model_key, raw)
        )
        for target in _frozen_target_values(frozen_selection):
            target_id = _safe_str(target.get("source_id") or f"{target.get('conversation_id')}:{target.get('turn_index')}")
            if target_id:
                review_target_ids.add(target_id)
                frozen_by_id[target_id] = target
    for item in all_turns:
        item["review_target"] = item.get("id") in review_target_ids

    reviewed_targets = 0
    corrected_labels = 0
    for task_id, node in raw.items():
        if not isinstance(node, dict):
            continue
        feedback_root = node.get("turn_feedbacks") or {}
        if not isinstance(feedback_root, dict):
            continue
        for turn_index, feedbacks_raw in feedback_root.items():
            feedbacks = _owner_iter_feedbacks(feedbacks_raw)
            target_id = f"{task_id}:{turn_index}"
            frozen_target = frozen_by_id.get(target_id, {})
            frozen_reviewed = frozen_target.get("target_status") == "reviewed" or int(frozen_target.get("feedback_count") or 0) > 0
            if target_id in review_target_ids and (feedbacks or frozen_reviewed):
                reviewed_targets += 1
            for fb in feedbacks:
                if fb.get("agreed_with_model") is False:
                    corrected_labels += 1

    summary = _owner_empty_summary()
    summary.update({
        "total_items": len(raw),
        "total_turns": len(all_turns),
        "human_count": human_count,
        "ai_count": ai_count,
        "avg_confidence": round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else 0,
        "avg_uncertainty": round(sum(uncertainty_values) / len(uncertainty_values), 4) if uncertainty_values else 0,
        "review_targets": len(review_target_ids),
        "reviewed_targets": reviewed_targets,
        "corrected_labels": corrected_labels
    })

    return {
        "selected_model": selected_model,
        "summary": summary,
        "metrics": _owner_metrics_from_pairs(y_true, y_pred),
        "examiners": list(examiners.values()),
        "most_uncertain_samples": _owner_samples_from_items(all_turns),
        "warnings": warnings
    }


def get_feedback_summary_from_existing_paths(project_id, project):
    result_type = detect_project_result_type(project)
    if result_type == "news":
        normalized = normalize_news_results(project_id, project)
    elif result_type == "uploaded_conversation":
        normalized = normalize_uploaded_conversation_results(project_id, project)
    elif result_type == "generated_conversation":
        normalized = normalize_generated_conversation_results(project_id, project)
    else:
        return {
            "reviewed_targets": 0,
            "corrected_labels": 0,
            "examiners": [],
            "warnings": ["Project result type is unknown."]
        }

    return {
        "reviewed_targets": normalized["summary"].get("reviewed_targets", 0),
        "corrected_labels": normalized["summary"].get("corrected_labels", 0),
        "examiners": normalized.get("examiners", []),
        "warnings": normalized.get("warnings", [])
    }


def _owner_empty_examiner(examiner_id):
    return {
        "examiner_id": _safe_str(examiner_id),
        "examiner_name": "",
        "email": "",
        "invitation_status": "unknown",
        "project_role": "examiner",
        "assigned_task_count": 0,
        "assigned_tasks": [],
        "model_selection_assigned": 0,
        "labeling_assigned": 0,
        "conversation_assigned": 0,
        "feedback_submitted": 0,
        "corrected_labels": 0,
        "pending_feedback_estimate": 0,
        "rating": None,
        "participation_status": "no_feedback_required"
    }


def _owner_task_type_for_summary(task):
    task_type = _safe_str(task.get("task_type")).strip().lower()
    conversation_type = _safe_str(task.get("conversation_type")).strip().lower()
    if task_type in ("model_selection", "labeling"):
        return task_type
    if conversation_type in ("human-ai", "human-human"):
        return "unknown"
    return "unknown"


def _owner_add_examiner(examiners, examiner_id, name="", email="", invitation_status=None):
    examiner_id = _safe_str(examiner_id)
    if not examiner_id:
        return None

    if examiner_id not in examiners:
        examiners[examiner_id] = _owner_empty_examiner(examiner_id)

    row = examiners[examiner_id]
    if name and not row.get("examiner_name"):
        row["examiner_name"] = _safe_str(name)
    if email and not row.get("email"):
        row["email"] = _safe_str(email)
    if invitation_status:
        row["invitation_status"] = _safe_str(invitation_status).strip().lower() or "unknown"
    return row


def _owner_user_lookup(examiner_id):
    if not examiner_id or examiner_id == "unknown":
        return "", ""
    try:
        user_doc = db.collection("users").document(examiner_id).get()
        if not user_doc.exists:
            return "", ""
        data = user_doc.to_dict() or {}
        profile = data.get("profile", {}) or {}
        name = f"{profile.get('firstName','')} {profile.get('lastName','')}".strip()
        email = data.get("email", "")
        return name, email
    except Exception:
        return "", ""


def _owner_feedback_is_corrected(feedback, model_prediction=None):
    if not isinstance(feedback, dict):
        return False
    if feedback.get("agreed_with_model") is False:
        return True

    label = _safe_str(feedback.get("label")).strip().lower()
    prediction = _safe_str(model_prediction).strip().lower()
    if not label or not prediction:
        return False

    label_is_ai = _owner_prediction_is_ai(label)
    prediction_is_ai = _owner_prediction_is_ai(prediction)
    return label_is_ai != prediction_is_ai


def _owner_add_feedback_event(feedback_events, examiners, event_key, feedback, model_prediction=None):
    if not isinstance(feedback, dict) or not feedback:
        return

    examiner_id = _safe_str(feedback.get("examiner_uid") or feedback.get("uid") or feedback.get("examiner_id"))
    if not examiner_id:
        return

    row = _owner_add_examiner(
        examiners,
        examiner_id,
        name=feedback.get("examiner_name") or feedback.get("name"),
        email=feedback.get("examiner_email") or feedback.get("email")
    )
    if row is None:
        return

    feedback_events.setdefault(examiner_id, {})
    if event_key not in feedback_events[examiner_id]:
        feedback_events[examiner_id][event_key] = _owner_feedback_is_corrected(feedback, model_prediction)
    elif _owner_feedback_is_corrected(feedback, model_prediction):
        feedback_events[examiner_id][event_key] = True


def _owner_selected_model_for_read(project_id, result_type):
    tasks = get_project_tasks(project_id)
    selected = _owner_task_selected_model(tasks, result_type)
    if result_type == "news":
        return selected.get("key") or "logistic"
    if result_type in ("uploaded_conversation", "generated_conversation"):
        return selected.get("key") or CONV_LOGREG_KEY
    return selected.get("key") or ""


def _owner_news_feedback_events(project_id, project, examiners):
    feedback_events = {}
    dataset_id = project.get("dataset_id")
    if not dataset_id:
        return feedback_events

    model_key = _owner_selected_model_for_read(project_id, "news")
    safe_pid = project_id.replace(".", "_").replace("#", "_").replace("$", "_").replace("[", "_").replace("]", "_")
    results_data = rtdb.reference(f"analysis_results/{safe_pid}/{model_key}").get()
    if not results_data and safe_pid != project_id:
        results_data = rtdb.reference(f"analysis_results/{project_id}/{model_key}").get()
    details = (results_data or {}).get("details") or []
    if isinstance(details, dict):
        details = list(details.values())
    prediction_by_article = {
        _safe_str(item.get("article_id") or item.get("id")): item.get("prediction")
        for item in details
        if isinstance(item, dict)
    }

    dataset_rows = rtdb.reference(f"datasets/uploaded_news/{dataset_id}").get() or {}
    for article_id, article_data in (dataset_rows.items() if isinstance(dataset_rows, dict) else []):
        if not isinstance(article_data, dict):
            continue
        article_id = _safe_str(article_id)
        event_key = f"news:{article_id}"
        model_prediction = prediction_by_article.get(article_id)

        feedback = article_data.get("feedback") or {}
        for fb in _owner_iter_feedbacks(feedback):
            _owner_add_feedback_event(feedback_events, examiners, event_key, fb, model_prediction)

        old_feedbacks = article_data.get("examiner_feedbacks") or {}
        for uid, fb in (old_feedbacks.items() if isinstance(old_feedbacks, dict) else []):
            if not isinstance(fb, dict):
                continue
            fb = dict(fb)
            fb.setdefault("examiner_uid", uid)
            _owner_add_feedback_event(feedback_events, examiners, event_key, fb, model_prediction)

    return feedback_events


def _owner_uploaded_feedback_events(project_id, examiners):
    feedback_events = {}
    model_key = _owner_selected_model_for_read(project_id, "uploaded_conversation")
    model_keys = [model_key] if model_key else [CONV_LOGREG_KEY, CONV_RNN_KEY]
    if CONV_LOGREG_KEY not in model_keys:
        model_keys.append(CONV_LOGREG_KEY)

    for mk in model_keys:
        mk = _uploaded_feedback_model_key(mk)
        base_ref = rtdb.reference(f"analysis_results/conversations/{mk}/{project_id}")
        run_id = _safe_str(base_ref.child("latest_run_id").get())
        if not run_id:
            continue

        run_ref = base_ref.child("runs").child(run_id)
        dialogue_turns = run_ref.child("dialogue_turns").get() or {}
        feedback_root = run_ref.child("turn_feedbacks").get() or {}
        prediction_map = {}
        for safe_key, turns_raw in (dialogue_turns.items() if isinstance(dialogue_turns, dict) else []):
            turns = list(turns_raw.values()) if isinstance(turns_raw, dict) else (turns_raw if isinstance(turns_raw, list) else [])
            for idx, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    continue
                ui_turn_index = idx + 1
                prediction_map[(safe_key, ui_turn_index)] = turn.get("prediction")
                try:
                    source_turn_index = int(turn.get("turn_index"))
                    prediction_map[(safe_key, source_turn_index)] = turn.get("prediction")
                except Exception:
                    pass

        for safe_key, dialogue_feedbacks in (feedback_root.items() if isinstance(feedback_root, dict) else []):
            if not isinstance(dialogue_feedbacks, dict):
                continue
            for turn_index_raw, turn_feedbacks in dialogue_feedbacks.items():
                try:
                    turn_index = int(turn_index_raw)
                except Exception:
                    turn_index = 0
                event_key = f"uploaded:{run_id}:{safe_key}:{turn_index}"
                model_prediction = prediction_map.get((safe_key, turn_index))
                for fb in _owner_iter_feedbacks(turn_feedbacks):
                    _owner_add_feedback_event(feedback_events, examiners, event_key, fb, model_prediction)
        break

    return feedback_events


def _owner_generated_feedback_events(project_id, examiners):
    feedback_events = {}
    model_key = _owner_selected_model_for_read(project_id, "generated_conversation")
    model_keys = [model_key] if model_key else [CONV_LOGREG_KEY, CONV_RNN_KEY]
    if CONV_LOGREG_KEY not in model_keys:
        model_keys.append(CONV_LOGREG_KEY)

    for mk in model_keys:
        raw = _generated_conversation_results_payload(project_id, mk) or {}
        if not isinstance(raw, dict) or not raw:
            continue

        for conversation_id, node in raw.items():
            if not isinstance(node, dict):
                continue
            turns_raw = node.get("turns", {}) or {}
            turns = list(turns_raw.values()) if isinstance(turns_raw, dict) else (turns_raw if isinstance(turns_raw, list) else [])
            prediction_map = {}
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                turn_index = int(turn.get("turn_index", 0) or 0)
                prediction_map[turn_index] = turn.get("prediction")

            feedback_root = node.get("turn_feedbacks") or {}
            if isinstance(feedback_root, dict):
                feedback_items = feedback_root.items()
            elif isinstance(feedback_root, list):
                feedback_items = enumerate(feedback_root)
            else:
                feedback_items = []

            for turn_index_raw, turn_feedbacks in feedback_items:
                try:
                    turn_index = int(turn_index_raw)
                except Exception:
                    turn_index = 0
                event_key = f"generated:{conversation_id}:{turn_index}"
                model_prediction = prediction_map.get(turn_index)
                for fb in _owner_iter_feedbacks(turn_feedbacks):
                    _owner_add_feedback_event(feedback_events, examiners, event_key, fb, model_prediction)
        break

    return feedback_events


def _owner_collect_feedback_events(project_id, project, result_type, examiners):
    if result_type == "news":
        return _owner_news_feedback_events(project_id, project, examiners)
    if result_type == "uploaded_conversation":
        return _owner_uploaded_feedback_events(project_id, examiners)
    if result_type == "generated_conversation":
        return _owner_generated_feedback_events(project_id, examiners)
    return {}


def _owner_counts_from_summary(result_type, summary):
    total_items = int(summary.get("total_items") or 0)
    total_turns = int(summary.get("total_turns") or 0)
    return {
        "total_articles": total_items if result_type == "news" else 0,
        "total_dialogues": total_items if result_type == "uploaded_conversation" else 0,
        "total_conversation_tasks": total_items if result_type == "generated_conversation" else 0,
        "total_turns_analyzed": total_turns if result_type in ("uploaded_conversation", "generated_conversation") else 0,
        "feedback_targets": int(summary.get("review_targets") or 0),
        "reviewed_feedback_targets": int(summary.get("reviewed_targets") or 0)
    }


def _owner_participation_status(examiner):
    assigned_tasks = examiner.get("assigned_tasks") or []
    feedback_submitted = int(examiner.get("feedback_submitted") or 0)
    pending_feedback = int(examiner.get("pending_feedback_estimate") or 0)
    labeling_assigned = int(examiner.get("labeling_assigned") or 0)

    if not assigned_tasks and feedback_submitted == 0:
        return "no_feedback_required"
    if assigned_tasks and feedback_submitted == 0 and labeling_assigned > 0:
        return "not_started"
    if feedback_submitted > 0 and pending_feedback > 0:
        return "in_progress"
    if feedback_submitted > 0 and pending_feedback == 0:
        return "completed"

    statuses = [_safe_str(task.get("task_status")).strip().lower() for task in assigned_tasks]
    if statuses and all(status == "completed" for status in statuses):
        return "completed"
    if statuses and any(status in ("progress", "completed", "active") for status in statuses):
        return "in_progress"
    if statuses:
        return "not_started"
    return "no_feedback_required"


def build_owner_examiners_summary(project_id, project, tasks, summary, result_type):
    examiners = {}

    invitations = db.collection("invitations").where("project_id", "==", project_id).stream()
    for inv in invitations:
        data = inv.to_dict() or {}
        examiner_id = data.get("examiner_id")
        _owner_add_examiner(
            examiners,
            examiner_id,
            name=data.get("examiner_name") or data.get("examiner_name"),
            email=data.get("examiner_email"),
            invitation_status=data.get("status") or "unknown"
        )

    for task in tasks:
        examiner_ids = task.get("examiner_ids") or []
        if not isinstance(examiner_ids, list):
            continue

        task_type = _owner_task_type_for_summary(task)
        conversation_type = _safe_str(task.get("conversation_type")).strip().lower() or None
        task_item = {
            "task_id": _safe_str(task.get("task_ID") or task.get("id")),
            "task_name": _safe_str(task.get("task_name")),
            "task_type": task_type,
            "conversation_type": conversation_type,
            "task_status": _safe_str(task.get("status")),
            "assigned": True
        }

        for examiner_id in examiner_ids:
            row = _owner_add_examiner(examiners, examiner_id)
            if row is None:
                continue
            row["assigned_tasks"].append(dict(task_item))
            if task_type == "model_selection":
                row["model_selection_assigned"] += 1
            elif task_type == "labeling":
                row["labeling_assigned"] += 1
            if conversation_type in ("human-ai", "human-human"):
                row["conversation_assigned"] += 1

    feedback_events = _owner_collect_feedback_events(project_id, project, result_type, examiners)
    for examiner_id, events in feedback_events.items():
        row = _owner_add_examiner(examiners, examiner_id)
        if row is None:
            continue
        row["feedback_submitted"] = len(events)
        row["corrected_labels"] = sum(1 for corrected in events.values() if corrected)

    rating_docs = db.collection("projects").document(project_id).collection("assigned_examiners").stream()
    for rating_doc in rating_docs:
        rating = rating_doc.to_dict() or {}
        examiner_id = rating.get("examiner_id") or rating_doc.id
        row = _owner_add_examiner(examiners, examiner_id)
        if row is None:
            continue
        row["rating"] = {
            "stars": rating.get("stars"),
            "comment": rating.get("comment", ""),
            "rated_at": _owner_json_value(rating.get("rated_at"))
        }

    shared_pending = max(0, int(summary.get("review_targets") or 0) - int(summary.get("reviewed_targets") or 0))
    for row in examiners.values():
        if row.get("labeling_assigned", 0) > 0:
            row["pending_feedback_estimate"] = shared_pending
        else:
            row["pending_feedback_estimate"] = 0

        row["assigned_task_count"] = len(row.get("assigned_tasks") or [])

        if row.get("invitation_status") == "unknown" and not row["assigned_tasks"] and (row.get("feedback_submitted", 0) > 0 or row.get("rating")):
            row["invitation_status"] = "removed"

        if not row.get("examiner_name") or not row.get("email"):
            name, email = _owner_user_lookup(row.get("examiner_id"))
            if name and not row.get("examiner_name"):
                row["examiner_name"] = name
            if email and not row.get("email"):
                row["email"] = email

        if not row.get("examiner_name"):
            row["examiner_name"] = "Examiner"

        row["participation_status"] = _owner_participation_status(row)

    return sorted(examiners.values(), key=lambda item: (
        item.get("invitation_status") != "accepted",
        item.get("examiner_name", ""),
        item.get("email", ""),
        item.get("examiner_id", "")
    ))


def _owner_summary_from_enhanced(rows, result_type):
    summary = _owner_empty_summary()
    final_labels = [_label_text_only(row.get("final_label")) for row in rows or [] if isinstance(row, dict)]
    summary["human_count"] = sum(1 for label in final_labels if label == "Human")
    summary["ai_count"] = sum(1 for label in final_labels if label == "AI")
    summary["review_targets"] = sum(1 for row in rows or [] if isinstance(row, dict) and row.get("review_target"))
    summary["reviewed_targets"] = sum(1 for row in rows or [] if isinstance(row, dict) and row.get("active_learning_state") in ("submitted", "locked"))
    summary["corrected_labels"] = sum(
        1 for row in rows or []
        if isinstance(row, dict)
        and row.get("active_learning_state") in ("submitted", "locked")
        and _label_text_only(row.get("feedback_label"))
        and _label_text_only(row.get("feedback_label")) != _label_text_only(row.get("prediction_label"))
    )
    uncertainty_values = [
        _owner_decimal(row.get("uncertainty"))
        for row in rows or []
        if isinstance(row, dict) and _owner_decimal(row.get("uncertainty")) is not None
    ]
    summary["avg_uncertainty"] = round(sum(uncertainty_values) / len(uncertainty_values), 4) if uncertainty_values else 0
    if result_type == "news":
        summary["total_items"] = len({row.get("article_id") for row in rows or [] if isinstance(row, dict) and row.get("article_id")})
    else:
        summary["total_turns"] = len(rows or [])
        summary["total_items"] = len({row.get("dialogue_id") for row in rows or [] if isinstance(row, dict) and row.get("dialogue_id")})
    return summary


def _owner_samples_from_enhanced(rows, limit=10):
    samples = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        samples.append({
            "id": _safe_str(row.get("sample_id")),
            "title": _safe_str(row.get("title") or row.get("dialogue_id")),
            "text": _safe_str(row.get("text"))[:220],
            "prediction": _safe_str(row.get("prediction_label")),
            "confidence": None,
            "uncertainty": _owner_decimal(row.get("uncertainty")),
            "review_target": bool(row.get("review_target"))
        })
    samples.sort(key=lambda item: (
        not item.get("review_target"),
        item.get("uncertainty") is None,
        item.get("uncertainty") if item.get("uncertainty") is not None else 1.0,
        item.get("id")
    ))
    return samples[:limit]


def _owner_feedback_from_enhanced(rows, limit=20):
    items = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if row.get("active_learning_state") not in ("submitted", "locked"):
            continue
        explanation = _safe_str(row.get("feedback_explanation"))
        if not explanation:
            continue
        reviewer = _safe_str(row.get("reviewed_by")) or "unknown"
        items.append({
            "sample_id": _safe_str(row.get("sample_id")),
            "project_type": "",
            "title": _safe_str(row.get("title") or row.get("dialogue_id")),
            "text_preview": _safe_str(row.get("text"))[:220],
            "prediction": _safe_str(row.get("prediction_label")),
            "corrected_label": _safe_str(row.get("feedback_label")),
            "examiner_uid": reviewer,
            "examiner_name": _reviewer_display_name(reviewer),
            "feedback_explanation": explanation,
            "submitted_at": _safe_str(row.get("reviewed_at")),
            "confidence": None,
            "uncertainty": _owner_decimal(row.get("uncertainty")),
            "agreed_with_model": _label_text_only(row.get("feedback_label")) == _label_text_only(row.get("prediction_label")),
            "correction_changed_prediction": _label_text_only(row.get("feedback_label")) != _label_text_only(row.get("prediction_label"))
        })
    items.sort(key=lambda item: item.get("submitted_at", ""), reverse=True)
    return items[:limit]


def build_owner_results_summary(project_id):
    project = get_project_basic_info(project_id)
    if not project:
        return None

    result_type = detect_project_result_type(project)
    warnings = []

    if result_type == "news":
        normalized = normalize_news_results(project_id, project)
    elif result_type == "uploaded_conversation":
        normalized = normalize_uploaded_conversation_results(project_id, project)
    elif result_type == "generated_conversation":
        normalized = normalize_generated_conversation_results(project_id, project)
    else:
        normalized = {
            "selected_model": {
                "key": "",
                "name": "",
                "version": "v1",
                "selected_at": "",
                "selected_by": ""
            },
            "summary": _owner_empty_summary(),
            "metrics": _owner_empty_metrics(),
            "examiners": [],
            "most_uncertain_samples": [],
            "warnings": ["Project result type is unknown."]
        }

    warnings.extend(normalized.get("warnings", []))
    if not normalized.get("selected_model", {}).get("key"):
        warnings.append("Selected model is missing.")

    enhanced_rows = []
    enhanced_metadata = {}
    try:
        current_version, _, detection_ready = _current_version_with_rows(project_id)
        if detection_ready:
            enhanced_payload = _build_enhanced_dataset(project_id, current_version)
            enhanced_rows = enhanced_payload.get("rows") or []
            enhanced_metadata = enhanced_payload.get("metadata") or {}
    except Exception as e:
        app.logger.warning("Owner Results used legacy summary fallback because enhanced dataset was unavailable: %s", e)

    tasks = get_project_tasks(project_id)
    summary_data = _owner_summary_from_enhanced(enhanced_rows, result_type) if enhanced_rows else normalized.get("summary", _owner_empty_summary())
    counts = _owner_counts_from_summary(result_type, summary_data)
    examiners = build_owner_examiners_summary(project_id, project, tasks, summary_data, result_type)

    if any(examiner.get("labeling_assigned", 0) > 0 for examiner in examiners):
        warnings.append("Feedback targets are shared across assigned labeling examiners; pending count is an estimate.")

    examiner_lookup = {item.get("examiner_id"): item for item in examiners if item.get("examiner_id")}
    feedback_explanations = _owner_feedback_from_enhanced(enhanced_rows) if enhanced_rows else _owner_feedback_explanations(project_id, project, result_type)
    for item in feedback_explanations:
        examiner = examiner_lookup.get(item.get("examiner_uid")) or {}
        item["participation_status"] = item.get("participation_status") or examiner.get("participation_status", "unknown")
        item["examiner_rating"] = item.get("examiner_rating") or examiner.get("rating")
        prediction = _safe_str(item.get("prediction")).strip().lower()
        corrected = _safe_str(item.get("corrected_label")).strip().lower()
        item["correction_changed_prediction"] = bool(prediction and corrected and prediction != corrected)
        item["agreed_with_model"] = not item["correction_changed_prediction"]

    project_public = {
        "project_id": project.get("project_id", ""),
        "project_name": project.get("project_name", ""),
        "category": project.get("category", ""),
        "dataset_id": project.get("dataset_id", ""),
        "generated_from_scratch": bool(project.get("generated_from_scratch", False)),
        "status": project.get("status", "")
    }
    selected_model_payload = normalized.get("selected_model") or {}
    if enhanced_metadata:
        selected_model_payload = {
            **selected_model_payload,
            "key": enhanced_metadata.get("model_key") or selected_model_payload.get("key"),
            "name": enhanced_metadata.get("model_name") or selected_model_payload.get("name"),
            "version": enhanced_metadata.get("version_id") or selected_model_payload.get("version") or "v1"
        }
    selected_model_key = selected_model_payload.get("key") or selected_model_payload.get("name")
    review_workflow = {
        "review_targets_enabled": _uses_frozen_active_learning(selected_model_key),
        "active_learning_enabled": _active_learning_retraining_supported(selected_model_key),
        "retraining_supported": _active_learning_retraining_supported(selected_model_key)
    }

    return {
        "ok": True,
        "project": project_public,
        "result_type": result_type,
        "selected_model": selected_model_payload,
        "review_workflow": review_workflow,
        "summary": summary_data,
        "counts": counts,
        "metrics": normalized.get("metrics", _owner_empty_metrics()),
        "examiners": examiners,
        "tasks": tasks,
        "most_uncertain_samples": _owner_samples_from_enhanced(enhanced_rows) if enhanced_rows else normalized.get("most_uncertain_samples", []),
        "feedback_explanations": feedback_explanations,
        "warnings": list(dict.fromkeys(warnings))
    }


@app.route("/api/project/<project_id>/results_summary", methods=["GET"])
def api_project_results_summary(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    project = get_project_basic_info(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    if project.get("owner_id") != session.get("uid"):
        return jsonify({"error": "Forbidden"}), 403

    try:
        summary = build_owner_results_summary(project_id)
        if not summary:
            return jsonify({"error": "Project not found"}), 404
        return jsonify(summary), 200
    except Exception as e:
        app.logger.exception("Owner results summary failed: %s", e)
        return jsonify({"error": "Failed to build results summary"}), 500


@app.route("/api/project/<project_id>/active_learning_feedback_summary", methods=["GET"])
def api_active_learning_feedback_summary(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    project = get_project_basic_info(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    if project.get("owner_id") != session.get("uid"):
        return jsonify({"error": "Forbidden"}), 403

    rows_raw = rtdb.reference(f"active_learning_feedback/{project_id}").get() or {}
    rows = list(rows_raw.values()) if isinstance(rows_raw, dict) else (rows_raw if isinstance(rows_raw, list) else [])

    total_feedback_rows = 0
    logistic_feedback_rows = 0
    used_for_retraining = 0
    corrected_labels = 0
    recent_explanations = []
    by_examiner = {}

    for row in rows:
        if not isinstance(row, dict):
            continue
        total_feedback_rows += 1
        if _is_logistic_model_key(row.get("selected_model_key") or row.get("selected_model_name")):
            logistic_feedback_rows += 1
        if bool(row.get("used_for_retraining", False)):
            used_for_retraining += 1
        if row.get("agreed_with_model") is False:
            corrected_labels += 1

        explanation = _safe_str(row.get("feedback_explanation"))
        if explanation:
            recent_explanations.append({
                "sample_id": _safe_str(row.get("sample_id")),
                "examiner_name": _safe_str(row.get("examiner_name")) or "Examiner",
                "corrected_label_text": _safe_str(row.get("corrected_label_text")),
                "feedback_explanation": explanation,
                "submitted_at": _safe_str(row.get("submitted_at"))
            })

        examiner_uid = _safe_str(row.get("examiner_uid")) or "unknown"
        if examiner_uid not in by_examiner:
            by_examiner[examiner_uid] = {
                "examiner_uid": examiner_uid,
                "examiner_name": row.get("examiner_name") or "Examiner",
                "feedback_rows": 0,
                "corrected_labels": 0
            }
        by_examiner[examiner_uid]["feedback_rows"] += 1
        if row.get("agreed_with_model") is False:
            by_examiner[examiner_uid]["corrected_labels"] += 1

    recent_explanations.sort(key=lambda item: item.get("submitted_at", ""), reverse=True)

    return jsonify({
        "ok": True,
        "project_id": project_id,
        "total_feedback_rows": total_feedback_rows,
        "logistic_feedback_rows": logistic_feedback_rows,
        "used_for_retraining": used_for_retraining,
        "unused_for_retraining": max(0, total_feedback_rows - used_for_retraining),
        "corrected_labels": corrected_labels,
        "explanations_count": len(recent_explanations),
        "recent_explanations": recent_explanations[:10],
        "by_examiner": sorted(by_examiner.values(), key=lambda item: (item.get("examiner_name", ""), item.get("examiner_uid", "")))
    }), 200


# =========================
# Owner Final Dataset Export Helpers
# =========================
# Read-only export helpers for Owner Results. They combine source rows,
# detection output, and optional examiner feedback without writing to Firebase.

NEWS_EXPORT_COLUMNS = [
    "project_id", "dataset_id", "article_id", "chunk_id", "source_id", "sample_id",
    "title", "text", "ground_truth", "prediction", "feedback_label",
    "final_label", "prediction_int", "human_probability", "ai_probability",
    "confidence", "uncertainty", "selected_model", "model_version",
    "active_learning_selected", "active_learning_state", "source_type", "examiner_uid", "examiner_name",
    "agreed_with_model", "feedback_explanation", "submitted_at", "used_for_retraining"
]

UPLOADED_CONVERSATION_EXPORT_COLUMNS = [
    "project_id", "dataset_id", "run_id", "dialogue_id", "turn_index",
    "sender", "text", "previous_text", "prediction", "prediction_int",
    "p_machine", "confidence", "uncertainty", "selected_model",
    "model_version", "active_learning_selected", "ground_truth",
    "source_row_id", "examiner_uid", "examiner_name", "agreed_with_model",
    "corrected_label", "corrected_MachineGen", "feedback_explanation",
    "submitted_at", "used_for_retraining"
]

GENERATED_CONVERSATION_EXPORT_COLUMNS = [
    "project_id", "task_id", "conversation_id", "conversation_type",
    "turn_index", "sender", "text", "previous_text", "prediction",
    "prediction_int", "ground_truth", "p_machine", "confidence", "uncertainty",
    "selected_model", "model_version", "active_learning_selected",
    "examiner_uid", "examiner_name", "agreed_with_model", "corrected_label",
    "corrected_MachineGen", "feedback_explanation", "submitted_at",
    "used_for_retraining"
]


def _label_to_machinegen(label):
    normalized = _normalize_binary_label(label)
    if normalized is not None:
        return normalized
    value = _safe_str(label).strip().lower()
    if value in ("human-written", "human written", "false"):
        return 0
    if value in ("true", "ai-generated", "ai generated"):
        return 1
    return None


def _first_present(*values):
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _flatten_json_for_csv(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(_owner_json_value(value), ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _excel_safe_csv_cell(value):
    value = _flatten_json_for_csv(value)
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@"):
        return "'" + value
    return value


def _safe_csv_response(rows, filename, columns=None):
    output = io.StringIO()
    if columns is None:
        columns = []
        for row in rows:
            for key in row.keys():
                if key not in columns:
                    columns.append(key)

    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _excel_safe_csv_cell(row.get(key)) for key in columns})

    return Response(
        "\ufeff" + output.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


def _owner_export_filename(project, stage, fmt):
    name = _safe_str(project.get("project_name") or project.get("project_id") or "project").lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_") or "project"
    return f"trustlens_{name}_{stage}.{fmt}"


def _owner_report_filename(project):
    name = _safe_str(project.get("project_name") or project.get("project_id") or "project").lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_") or "project"
    return f"trustlens_{name}_report.pdf"


def _owner_confusion_values(confusion_matrix):
    if not confusion_matrix:
        return None
    try:
        if isinstance(confusion_matrix, list) and len(confusion_matrix) == 2:
            return {
                "tn": int(confusion_matrix[0][0] or 0),
                "fp": int(confusion_matrix[0][1] or 0),
                "fn": int(confusion_matrix[1][0] or 0),
                "tp": int(confusion_matrix[1][1] or 0)
            }
        if isinstance(confusion_matrix, dict):
            return {
                "tn": int(confusion_matrix.get("true_negative") or confusion_matrix.get("tn") or 0),
                "fp": int(confusion_matrix.get("false_positive") or confusion_matrix.get("fp") or 0),
                "fn": int(confusion_matrix.get("false_negative") or confusion_matrix.get("fn") or 0),
                "tp": int(confusion_matrix.get("true_positive") or confusion_matrix.get("tp") or 0)
            }
    except Exception:
        return None
    return None


def _owner_performance_details(metrics):
    values = _owner_confusion_values((metrics or {}).get("confusion_matrix"))
    if not values:
        return {
            "false_positive_rate": None,
            "false_negative_rate": None,
            "total_errors": None,
            "correct_predictions": None,
            "confusion": None
        }

    tn = values["tn"]
    fp = values["fp"]
    fn = values["fn"]
    tp = values["tp"]
    return {
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
        "false_negative_rate": round(fn / (fn + tp), 4) if (fn + tp) else None,
        "total_errors": fp + fn,
        "correct_predictions": tp + tn,
        "confusion": values
    }


def _owner_report_recommendation(summary_data, metrics, selected_model):
    is_logistic = _is_logistic_model_key((selected_model or {}).get("key") or (selected_model or {}).get("name"))
    if not is_logistic:
        return "Frozen review targets are enabled for standard review; retraining is not supported for this model."
    if isinstance(metrics, dict) and metrics.get("available") is False:
        return "Model performance metrics need valid ground truth labels before F1-based decisions can be made."

    feedback_targets = int((summary_data or {}).get("review_targets") or 0)
    reviewed_targets = int((summary_data or {}).get("reviewed_targets") or 0)
    if reviewed_targets < feedback_targets:
        return "More examiner feedback is recommended before stopping."
    return "Review the reported F1, support, and confusion matrix before making a final owner decision."


def _owner_metric_text(value):
    number = _owner_decimal(value)
    if number is None:
        return "-"
    return f"{number * 100:.1f}%"


def _owner_plain_text(value, fallback="-"):
    text = _safe_str(value).strip()
    return text if text else fallback


def _owner_report_owner_info(uid):
    name = ""
    email = ""
    try:
        user_doc = db.collection("users").document(uid).get()
        if user_doc.exists:
            data = user_doc.to_dict() or {}
            profile = data.get("profile", {}) or {}
            name = f"{profile.get('firstName','')} {profile.get('lastName','')}".strip()
            email = data.get("email", "")
    except Exception:
        pass
    return name or "Project Owner", email or "-"


def _owner_report_dataset_source(project):
    if bool(project.get("generated_from_scratch", False)):
        return "Generated from scratch"
    if project.get("dataset_id"):
        return "Uploaded dataset"
    return "No dataset"


def _owner_build_pdf_report(summary_payload, owner_name, owner_email):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#155cfb"), fontSize=22, leading=28))
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading2"], textColor=colors.HexColor("#0e162b"), fontSize=14, leading=18, spaceBefore=10))
    styles.add(ParagraphStyle(name="SmallText", parent=styles["BodyText"], fontSize=8, leading=10))

    project = summary_payload.get("project") or {}
    summary_data = summary_payload.get("summary") or {}
    counts = summary_payload.get("counts") or {}
    selected = summary_payload.get("selected_model") or {}
    metrics = summary_payload.get("metrics") or {}
    performance = _owner_performance_details(metrics)
    confusion = performance.get("confusion")
    is_logistic = _is_logistic_model_key(selected.get("key") or selected.get("name"))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    recommendation = _owner_report_recommendation(summary_data, metrics, selected)

    def paragraph(text, style="BodyText"):
        return Paragraph(escape(_safe_str(text)), styles[style])

    def section(title):
        return Paragraph(escape(title), styles["SectionTitle"])

    def table(data, widths=None):
        wrapped = []
        for row in data:
            wrapped.append([Paragraph(escape(_owner_plain_text(cell)), styles["SmallText"]) for cell in row])
        result = Table(wrapped, colWidths=widths, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef5ff")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0e162b")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d0d5dd")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fbff")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return result

    story = [
        Paragraph("TrustLens Professional Report", styles["ReportTitle"]),
        paragraph("AI-generated text detection project report", "BodyText"),
        Spacer(1, 0.18 * inch),
        section("A) Cover / Project Summary"),
        table([
            ["Field", "Value"],
            ["Project name", project.get("project_name")],
            ["Project category", "Conversation" if "conversation" in _safe_str(project.get("category")).lower() else "News"],
            ["Result type", summary_payload.get("result_type")],
            ["Dataset source", _owner_report_dataset_source(project)],
            ["Selected model", selected.get("name")],
            ["Model version", selected.get("version") or "v1"],
            ["Project status", project.get("status")],
            ["Generated date/time", generated_at],
            ["Owner", owner_name],
            ["Owner email", owner_email],
        ], [2.0 * inch, 4.5 * inch]),
        Spacer(1, 0.12 * inch),
        section("B) Detection Summary"),
        table([
            ["Metric", "Value"],
            ["Total articles", counts.get("total_articles")],
            ["Total dialogues", counts.get("total_dialogues")],
            ["Total conversation tasks", counts.get("total_conversation_tasks")],
            ["Total turns analyzed", counts.get("total_turns_analyzed")],
            ["Human count", summary_data.get("human_count")],
            ["AI count", summary_data.get("ai_count")],
            ["Average confidence", _owner_metric_text(summary_data.get("avg_confidence"))],
            ["Average uncertainty", _owner_metric_text(summary_data.get("avg_uncertainty"))],
            ["Feedback targets", counts.get("feedback_targets")],
            ["Reviewed feedback targets", counts.get("reviewed_feedback_targets")],
            ["Corrected labels", summary_data.get("corrected_labels")],
        ], [2.4 * inch, 4.1 * inch]),
        Spacer(1, 0.12 * inch),
        section("C) Performance Evaluation"),
    ]

    has_metrics = metrics.get("available") is True or (
        "available" not in metrics and any(metrics.get(key) is not None for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"))
    )
    if has_metrics:
        story.extend([
            table([
                ["Metric", "Value"],
                ["Accuracy", _owner_metric_text(metrics.get("accuracy"))],
                ["Precision Macro", _owner_metric_text(metrics.get("precision_macro"))],
                ["Recall Macro", _owner_metric_text(metrics.get("recall_macro"))],
                ["F1 Macro", _owner_metric_text(metrics.get("f1_macro"))],
                ["AI F1", _owner_metric_text(metrics.get("f1_ai"))],
                ["False Positive Rate", _owner_metric_text(performance.get("false_positive_rate"))],
                ["False Negative Rate", _owner_metric_text(performance.get("false_negative_rate"))],
                ["Labeled support", (metrics.get("support") or {}).get("total_labeled")],
                ["Human support", (metrics.get("support") or {}).get("human")],
                ["AI support", (metrics.get("support") or {}).get("ai")],
            ], [2.4 * inch, 4.1 * inch])
        ])
        if confusion:
            story.extend([
                Spacer(1, 0.08 * inch),
                table([
                    ["Confusion Matrix", "Predicted Human", "Predicted AI/Machine"],
                    ["Actual Human", confusion["tn"], confusion["fp"]],
                    ["Actual AI/Machine", confusion["fn"], confusion["tp"]],
                ], [2.0 * inch, 2.2 * inch, 2.2 * inch])
            ])
    else:
        story.append(paragraph(metrics.get("reason") or "Performance metrics are not available because ground truth labels are missing or invalid."))

    story.extend([
        Spacer(1, 0.12 * inch),
        section("D) Error Analysis"),
    ])
    if confusion:
        story.extend([
            table([
                ["Item", "Value"],
                ["True positives", confusion["tp"]],
                ["True negatives", confusion["tn"]],
                ["False positives", confusion["fp"]],
                ["False negatives", confusion["fn"]],
                ["Total errors", performance.get("total_errors")],
                ["Correct predictions", performance.get("correct_predictions")],
            ], [2.4 * inch, 4.1 * inch]),
            Spacer(1, 0.08 * inch),
            paragraph("False Positive means human text incorrectly predicted as AI/Machine-generated. False Negative means AI/Machine-generated text incorrectly predicted as Human. This interpretation follows the stored confusion matrix order [[TN, FP], [FN, TP]].")
        ])
    else:
        story.append(paragraph("Error analysis is not available because the confusion matrix is missing."))

    story.extend([
        Spacer(1, 0.12 * inch),
        section("E) Active Learning Summary"),
    ])
    if is_logistic:
        story.extend([
            table([
                ["Field", "Value"],
                ["Active Learning Enabled", "Yes"],
                ["Strategy", "Uncertainty sampling / least confident samples"],
                ["Formula", "uncertainty = abs(probability - 0.5)"],
                ["Average uncertainty", _owner_metric_text(summary_data.get("avg_uncertainty"))],
                ["Review targets", summary_data.get("review_targets")],
                ["Reviewed targets", summary_data.get("reviewed_targets")],
                ["Corrected labels", summary_data.get("corrected_labels")],
                ["Most uncertain sample count", len(summary_payload.get("most_uncertain_samples") or [])],
            ], [2.4 * inch, 4.1 * inch]),
            Spacer(1, 0.08 * inch),
            paragraph("Lower uncertainty value means the probability is closer to 0.5, so the model is less certain and the sample is more useful for feedback review. This report only summarizes detection and feedback.")
        ])
    else:
        story.append(paragraph("This report includes standard detection and feedback results for the selected model."))

    samples = (summary_payload.get("most_uncertain_samples") or [])[:10]
    story.extend([Spacer(1, 0.12 * inch), section("F) Most Uncertain Samples")])
    if samples:
        sample_rows = [["ID", "Title/Dialog", "Text preview", "Prediction", "Confidence", "Uncertainty", "Review Target"]]
        for sample in samples:
            sample_id = _owner_plain_text(sample.get("id"))
            sample_rows.append([
                sample_id[:6] + "..." + sample_id[-5:] if len(sample_id) > 14 else sample_id,
                sample.get("title") or sample.get("dialogue_id") or "-",
                _safe_str(sample.get("text"))[:120],
                sample.get("prediction"),
                _owner_metric_text(sample.get("confidence")),
                _owner_metric_text(sample.get("uncertainty")),
                "Yes" if sample.get("review_target") else "No",
            ])
        story.append(table(sample_rows, [0.75 * inch, 1.0 * inch, 2.1 * inch, 0.8 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch]))
    else:
        story.append(paragraph("No uncertain samples are available yet."))

    examiners = summary_payload.get("examiners") or []
    story.extend([Spacer(1, 0.12 * inch), section("G) Examiner Participation")])
    story.append(paragraph(f"Total examiners connected to the project: {len(examiners)}"))
    if examiners:
        examiner_rows = [["Name", "Email", "Invitation", "Tasks", "Feedback", "Corrected", "Pending", "Status"]]
        for examiner in examiners:
            examiner_rows.append([
                examiner.get("examiner_name"),
                examiner.get("email"),
                examiner.get("invitation_status"),
                examiner.get("assigned_task_count"),
                examiner.get("feedback_submitted"),
                examiner.get("corrected_labels"),
                examiner.get("pending_feedback_estimate"),
                examiner.get("participation_status"),
            ])
        story.append(table(examiner_rows, [1.0 * inch, 1.25 * inch, 0.75 * inch, 0.5 * inch, 0.65 * inch, 0.65 * inch, 0.6 * inch, 1.0 * inch]))
    else:
        story.append(paragraph("No examiner participation data is available yet."))

    explanations = (summary_payload.get("feedback_explanations") or [])[:10]
    story.extend([Spacer(1, 0.12 * inch), section("Examiner Feedback Explanations")])
    if explanations:
        explanation_rows = [["Examiner", "Prediction", "Corrected", "Text preview", "Explanation"]]
        for item in explanations:
            explanation_rows.append([
                item.get("examiner_name"),
                item.get("prediction"),
                item.get("corrected_label"),
                _safe_str(item.get("text_preview"))[:100],
                item.get("feedback_explanation"),
            ])
        story.append(table(explanation_rows, [1.0 * inch, 0.85 * inch, 0.85 * inch, 2.0 * inch, 1.8 * inch]))
    else:
        story.append(paragraph("No examiner feedback explanations were submitted."))

    story.extend([
        Spacer(1, 0.12 * inch),
        section("H) Final Recommendation / Owner Decision Support"),
        paragraph(recommendation)
    ])

    warnings = summary_payload.get("warnings") or []
    if warnings:
        story.extend([Spacer(1, 0.12 * inch), section("Notes")])
        for warning in warnings:
            story.append(paragraph(f"- {warning}"))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def _safe_json_export_response(payload, filename):
    return Response(
        json.dumps(_owner_json_value(payload), ensure_ascii=False, indent=2),
        mimetype="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


def _get_project_selected_model(project_id, project, tasks):
    result_type = detect_project_result_type(project)
    selected = _owner_task_selected_model(tasks, result_type)
    model_key = selected.get("key") or _owner_selected_model_for_read(project_id, result_type)
    model_name = selected.get("name") or ("RNN" if model_key == CONV_RNN_KEY or model_key == "rnn" else "Logistic Regression")
    return {
        "key": model_key,
        "name": model_name,
        "version": _active_learning_model_version(project_id, model_key),
        "result_type": result_type
    }


def _feedback_items(feedback_node):
    if not isinstance(feedback_node, dict) or not feedback_node:
        return []
    if "examiner_uid" in feedback_node or "agreed_with_model" in feedback_node or "label" in feedback_node:
        item = dict(feedback_node)
        item.setdefault("examiner_uid", item.get("uid") or item.get("examiner_id"))
        return [item]

    items = []
    for uid, feedback in feedback_node.items():
        if not isinstance(feedback, dict):
            continue
        item = dict(feedback)
        item.setdefault("examiner_uid", uid)
        items.append(item)
    return items


def _feedback_fields(feedback, prediction):
    if not feedback:
        return {
            "examiner_uid": None,
            "examiner_name": None,
            "agreed_with_model": None,
            "corrected_label": None,
            "corrected_MachineGen": None,
            "feedback_explanation": "",
            "submitted_at": None,
            "used_for_retraining": False
        }

    agreed = feedback.get("agreed_with_model")
    if agreed is True:
        corrected_label = prediction
    elif agreed is False:
        corrected_label = feedback.get("label")
    else:
        corrected_label = feedback.get("label") or prediction

    return {
        "examiner_uid": feedback.get("examiner_uid") or feedback.get("uid") or feedback.get("examiner_id"),
        "examiner_name": feedback.get("examiner_name") or feedback.get("name"),
        "agreed_with_model": agreed,
        "corrected_label": corrected_label,
        "corrected_MachineGen": _label_to_machinegen(corrected_label),
        "feedback_explanation": _safe_str(feedback.get("explanation") or feedback.get("feedback_explanation")),
        "submitted_at": feedback.get("submitted_at"),
        "used_for_retraining": False
    }


def _rows_with_feedback(base_row, feedbacks, prediction, stage):
    if stage != "feedback":
        return [base_row]
    if not feedbacks:
        row = dict(base_row)
        row.update(_feedback_fields(None, prediction))
        return [row]

    rows = []
    for feedback in feedbacks:
        row = dict(base_row)
        row.update(_feedback_fields(feedback, prediction))
        rows.append(row)
    return rows


def _has_detection_results(rows):
    for row in rows:
        if row.get("prediction") is not None and row.get("prediction") != "":
            return True
    return False


def _news_results_payload(project_id, model_key):
    safe_pid = project_id.replace(".", "_").replace("#", "_").replace("$", "_").replace("[", "_").replace("]", "_")
    results_data = rtdb.reference(f"analysis_results/{safe_pid}/{model_key}").get()
    if not results_data and safe_pid != project_id:
        results_data = rtdb.reference(f"analysis_results/{project_id}/{model_key}").get()
    return results_data or {}


def _news_source_payload(source):
    if not isinstance(source, dict):
        return {}
    payload = source.get("payload")
    return payload if isinstance(payload, dict) else source


def _text_from_news_payload(payload, detail):
    title = (
        _payload_value(payload, "title", "Title", "headline", "Headline")
        or detail.get("title") or ""
    )
    text = (
        _payload_value(payload, "Article", "article", "content", "text", "Text")
        or detail.get("content") or ""
    )
    return title, text


def _news_chunk_by_index(detail, chunk_index):
    try:
        wanted = int(chunk_index)
    except Exception:
        return None
    for pos, chunk in enumerate((detail or {}).get("chunks") or [], start=1):
        if not isinstance(chunk, dict):
            continue
        current = int(chunk.get("chunk_index") or pos)
        if current == wanted:
            return chunk
    return None


def _news_chunk_feedbacks(source, chunk_index):
    if not isinstance(source, dict) or chunk_index is None:
        return {}
    chunk_root = source.get("chunk_feedback") or source.get("chunk_feedbacks") or {}
    if not isinstance(chunk_root, dict):
        return {}
    return chunk_root.get(str(chunk_index), {}) or {}


def _news_first_chunk_feedback(source, chunk_index):
    feedbacks = _news_chunk_feedbacks(source, chunk_index)
    if not isinstance(feedbacks, dict) or not feedbacks:
        return None
    return next(iter(feedbacks.values()))


def _news_chunks_export_payload(detail, source, active_ids):
    chunks = []
    article_id = _safe_str(detail.get("article_id") or detail.get("id"))
    for pos, chunk in enumerate(detail.get("chunks") or [], start=1):
        if not isinstance(chunk, dict):
            continue
        chunk_index = int(chunk.get("chunk_index") or pos)
        sample_id = _make_active_learning_sample_id("news", article_id=article_id, chunk_index=chunk_index)
        feedback = _news_first_chunk_feedback(source, chunk_index)
        chunks.append({
            "chunk_index": chunk_index,
            "chunk_id": chunk.get("chunk_id") or f"chunk:{chunk_index}",
            "chunk_text": chunk.get("chunk_text") or chunk.get("text") or "",
            "prediction": chunk.get("prediction"),
            "prediction_int": _first_present(chunk.get("prediction_int"), _label_to_machinegen(chunk.get("prediction"))),
            "confidence": chunk.get("confidence"),
            "uncertainty": chunk.get("uncertainty"),
            "active_learning_selected": sample_id in active_ids,
            "target_status": "reviewed" if feedback else "pending",
            "feedback": feedback
        })
    return chunks


def _news_export_feedback_fields(feedback_node, prediction):
    feedback = feedback_node if isinstance(feedback_node, dict) else None
    if feedback and not ("examiner_uid" in feedback or "corrected_label" in feedback or "label" in feedback):
        feedbacks = _feedback_items(feedback)
        feedback = feedbacks[0] if feedbacks else None
    prediction_label = _normalized_visible_label(prediction)
    if not feedback:
        return {
            "examiner_uid": None,
            "examiner_name": None,
            "agreed_with_model": None,
            "feedback_label": "",
            "final_label": prediction_label,
            "feedback_explanation": "",
            "submitted_at": None,
            "used_for_retraining": False
        }

    agreed = feedback.get("agreed_with_model")
    feedback_label = prediction_label if agreed is True else _normalized_visible_label(
        feedback.get("corrected_label_text")
        or feedback.get("corrected_label")
        or feedback.get("label")
    )
    if not feedback_label:
        feedback_label = prediction_label
    return {
        "examiner_uid": feedback.get("examiner_uid") or feedback.get("uid") or feedback.get("examiner_id"),
        "examiner_name": _reviewer_display_name(
            feedback.get("examiner_uid") or feedback.get("uid") or feedback.get("examiner_id"),
            feedback.get("examiner_name") or feedback.get("reviewed_by_name")
        ),
        "agreed_with_model": agreed,
        "feedback_label": feedback_label,
        "final_label": feedback_label,
        "feedback_explanation": _safe_str(feedback.get("explanation") or feedback.get("feedback_explanation")),
        "submitted_at": feedback.get("submitted_at") or feedback.get("reviewed_at"),
        "used_for_retraining": False
    }


def _feedback_for_current_or_first(feedback_node, uid=None):
    if not isinstance(feedback_node, dict) or not feedback_node:
        return None
    current = _feedback_record_for_uid(feedback_node, uid) if uid else None
    if current:
        return current
    if "examiner_uid" in feedback_node or "corrected_label" in feedback_node or "label" in feedback_node:
        return feedback_node
    for item in feedback_node.values():
        if isinstance(item, dict):
            return item
    return None


def _news_feedback_by_sample(project_id, version_id, sample_id, source=None, chunk_index=None, uid=None):
    sample_id = _safe_str(sample_id)
    if not sample_id:
        return None

    version_feedback = rtdb.reference(_feedback_versions_path(project_id, version_id)).child(_snapshot_row_key(sample_id)).get()
    if isinstance(version_feedback, dict) and version_feedback:
        return version_feedback

    if _safe_str(version_id) == "v1":
        normalized_feedback = rtdb.reference(f"active_learning_feedback/{project_id}/{sample_id}").get()
        if isinstance(normalized_feedback, dict) and normalized_feedback:
            return normalized_feedback
    else:
        return None

    if isinstance(source, dict):
        if chunk_index is not None:
            dataset_feedback = _feedback_for_current_or_first(_news_chunk_feedbacks(source, chunk_index), uid)
        else:
            dataset_feedback = _feedback_for_current_or_first(source.get("feedback") or source.get("examiner_feedbacks") or {}, uid)
        if dataset_feedback:
            return dataset_feedback

    return None


def _news_feedback_label(feedback, prediction):
    if not isinstance(feedback, dict) or not feedback:
        return ""
    label = _normalized_visible_label(
        feedback.get("corrected_label_text")
        or feedback.get("corrected_label")
        or feedback.get("label")
    )
    if label:
        return label
    if feedback.get("agreed_with_model") is True:
        return _normalized_visible_label(prediction)
    return ""


def _news_export_probability_pair(chunk, detail):
    human_probability = _probability_percent(_first_present(
        chunk.get("human"),
        chunk.get("human_percentage"),
        detail.get("human_probability"),
        detail.get("human_percentage")
    ))
    ai_probability = _probability_percent(_first_present(
        chunk.get("ai"),
        chunk.get("ai_percentage"),
        detail.get("ai_probability"),
        detail.get("ai_percentage")
    ))
    if human_probability is None and ai_probability is not None:
        human_probability = round(100.0 - ai_probability, 6)
    if ai_probability is None and human_probability is not None:
        ai_probability = round(100.0 - human_probability, 6)
    return human_probability, ai_probability


def _export_news_dataset(project_id, project, stage):
    tasks = get_project_tasks(project_id)
    selected = _get_project_selected_model(project_id, project, tasks)
    model_key = selected.get("key") or "logistic"
    dataset_id = project.get("dataset_id") or ""
    dataset_rows = rtdb.reference(f"datasets/uploaded_news/{dataset_id}").get() or {}
    results_data = _news_results_payload(project_id, model_key)
    details = results_data.get("details") or results_data.get("results") or []
    if isinstance(details, dict):
        details = list(details.values())
    if not isinstance(details, list):
        details = []

    detail_map = {
        _safe_str(item.get("article_id") or item.get("id")): item
        for item in details
        if isinstance(item, dict)
    }

    model_version = selected.get("version") or _active_learning_model_version(project_id, model_key)
    active_ids = set()
    if _uses_frozen_active_learning(model_key):
        frozen_selection = _freeze_active_learning_targets_if_missing(
            project_id,
            "news",
            model_key,
            model_version,
            _news_active_learning_candidates(project_id, model_key, details)
        )
        active_ids = {
            _safe_str(target.get("sample_id"))
            for target in _frozen_target_values(frozen_selection)
        }

    article_ids = list(dataset_rows.keys()) if isinstance(dataset_rows, dict) else []
    for article_id in detail_map.keys():
        if article_id not in article_ids:
            article_ids.append(article_id)

    rows = []
    for article_number, article_id in enumerate(article_ids, start=1):
        source = dataset_rows.get(article_id, {}) if isinstance(dataset_rows, dict) else {}
        payload = _news_source_payload(source)
        detail = detail_map.get(article_id, {})
        title, text = _text_from_news_payload(payload, detail)
        display_article_id = f"article_{article_number:03d}"
        original_label = _normalized_visible_label(_first_present(
            detail.get("ground_truth"),
            _extract_ground_truth_from_payload(payload),
            payload.get("MachineGen"),
            payload.get("label"),
            payload.get("target"),
            payload.get("ground_truth")
        ))
        chunks = detail.get("chunks") if isinstance(detail.get("chunks"), list) else []
        if not chunks:
            chunks = [{
                "chunk_index": None,
                "chunk_text": text,
                "prediction": detail.get("prediction"),
                "prediction_int": detail.get("prediction_int"),
                "confidence": detail.get("confidence"),
                "uncertainty": detail.get("uncertainty"),
                "human_percentage": detail.get("human_percentage"),
                "ai_percentage": detail.get("ai_percentage")
            }]

        for pos, chunk in enumerate(chunks, start=1):
            if not isinstance(chunk, dict):
                continue
            chunk_index = chunk.get("chunk_index")
            if chunk_index is not None:
                try:
                    chunk_index = int(chunk_index)
                except Exception:
                    chunk_index = pos
            sample_id = _make_active_learning_sample_id("news", article_id=article_id, chunk_index=chunk_index)
            prediction = _normalized_visible_label(chunk.get("prediction") or detail.get("prediction"))
            human_probability, ai_probability = _news_export_probability_pair(chunk, detail)
            feedback_node = _news_feedback_by_sample(
                project_id,
                model_version,
                sample_id,
                source=source,
                chunk_index=chunk_index
            )
            feedback_fields = _news_export_feedback_fields(feedback_node, prediction)
            has_feedback = bool(feedback_fields.get("feedback_label"))
            selected_for_review = sample_id in active_ids
            rows.append({
                "project_id": project_id,
                "dataset_id": dataset_id,
                "article_id": display_article_id,
                "chunk_id": f"chunk_{article_number:03d}_{chunk_index or pos}",
                "source_id": article_id,
                "sample_id": sample_id,
                "title": title,
                "text": chunk.get("chunk_text") or chunk.get("text") or text,
                "ground_truth": original_label,
                "prediction": prediction,
                "prediction_int": _first_present(chunk.get("prediction_int"), _label_to_machinegen(prediction)),
                "human_probability": human_probability,
                "ai_probability": ai_probability,
                "confidence": _probability_percent(chunk.get("confidence")),
                "uncertainty": _chunk_uncertainty_percent(chunk),
                "selected_model": selected.get("name"),
                "model_version": model_version,
                "active_learning_selected": selected_for_review,
                "active_learning_state": "reviewed" if has_feedback else ("selected_for_review" if selected_for_review else "not_selected_for_review"),
                "source_type": "uploaded_news",
                **feedback_fields
            })

    return rows, NEWS_EXPORT_COLUMNS


def _turns_to_list(turns_raw):
    if isinstance(turns_raw, dict):
        return list(turns_raw.values())
    if isinstance(turns_raw, list):
        return turns_raw
    return []


def _source_rows_by_id(dataset_id):
    rows = rtdb.reference(f"datasets/uploaded_conversations/{dataset_id}").get() or {}
    if not isinstance(rows, dict):
        return {}, {}
    by_row_id = {}
    by_order = {}
    for idx, (key, value) in enumerate(rows.items()):
        payload = value.get("payload") if isinstance(value, dict) and isinstance(value.get("payload"), dict) else value
        if not isinstance(payload, dict):
            payload = {}
        by_order[idx] = payload
        by_row_id[_safe_str(key)] = payload
        explicit_id = _safe_str(payload.get("row_id") or payload.get("id") or payload.get("source_row_id"))
        if explicit_id:
            by_row_id[explicit_id] = payload
    return by_row_id, by_order


def _export_uploaded_conversation_dataset(project_id, project, stage):
    tasks = get_project_tasks(project_id)
    selected = _get_project_selected_model(project_id, project, tasks)
    model_key = _uploaded_feedback_model_key(selected.get("key") or CONV_LOGREG_KEY)
    dataset_id = project.get("dataset_id") or ""
    run_id, run_ref, _ = _uploaded_conversation_run(project_id, model_key)
    if not run_ref and model_key != CONV_LOGREG_KEY:
        model_key = CONV_LOGREG_KEY
        run_id, run_ref, _ = _uploaded_conversation_run(project_id, model_key)

    if not run_ref:
        return [], UPLOADED_CONVERSATION_EXPORT_COLUMNS

    dialogue_turns = run_ref.child("dialogue_turns").get() or {}
    feedback_root = run_ref.child("turn_feedbacks").get() or {}
    key_map = run_ref.child("dialogue_key_map").get() or {}
    reverse_key_map = {v: k for k, v in key_map.items()} if isinstance(key_map, dict) else {}
    source_by_row_id, source_by_order = _source_rows_by_id(dataset_id)

    all_turn_refs = []
    for safe_key, turns_raw in (dialogue_turns.items() if isinstance(dialogue_turns, dict) else []):
        turns = _turns_to_list(turns_raw)
        for idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            all_turn_refs.append((safe_key, idx, turn))

    active_keys = set()
    if _uses_frozen_active_learning(model_key):
        model_version = _active_learning_model_version(project_id, model_key)
        frozen_selection = _freeze_active_learning_targets_if_missing(
            project_id,
            "uploaded_conversation",
            model_key,
            model_version,
            _uploaded_conversation_active_learning_candidates(project_id, model_key, run_id, run_ref)
        )
        active_keys = {
            (_safe_str(target.get("safe_key")), int(target.get("turn_index") or 0))
            for target in _frozen_target_values(frozen_selection)
        }

    rows = []
    source_order_index = 0
    for safe_key, idx, turn in all_turn_refs:
        turn_index = int(turn.get("turn_index", idx + 1) or idx + 1)
        row_id = _safe_str(turn.get("row_id") or turn.get("source_row_id"))
        source_payload = source_by_row_id.get(row_id) or source_by_order.get(source_order_index, {})
        source_order_index += 1
        dialogue_id = turn.get("dialogue_id") or reverse_key_map.get(safe_key, safe_key)
        prediction = turn.get("prediction")
        p_machine = _owner_decimal(_first_present(turn.get("p_machine"), turn.get("ai_probability"), turn.get("probability")))

        base_row = {
            "project_id": project_id,
            "dataset_id": dataset_id,
            "run_id": run_id,
            "dialogue_id": dialogue_id,
            "turn_index": turn_index,
            "sender": _first_present(turn.get("sender"), source_payload.get("sender"), source_payload.get("role"), source_payload.get("author")),
            "text": _first_present(turn.get("text"), source_payload.get("text"), source_payload.get("message"), source_payload.get("content")),
            "previous_text": _first_present(turn.get("previous_text"), turn.get("prev_text"), source_payload.get("prev_text")),
            "prediction": prediction,
            "prediction_int": turn.get("prediction_int"),
            "p_machine": p_machine,
            "confidence": _owner_decimal(turn.get("confidence")),
            "uncertainty": _owner_decimal(turn.get("uncertainty")),
            "selected_model": selected.get("name"),
            "model_version": selected.get("version") or "v1",
            "active_learning_selected": (safe_key, turn_index) in active_keys,
            "ground_truth": _first_present(turn.get("ground_truth"), _extract_ground_truth_from_payload(source_payload), source_payload.get("ground_truth"), source_payload.get("label"), source_payload.get("target")),
            "source_row_id": row_id
        }

        dialogue_feedbacks = feedback_root.get(safe_key, {}) if isinstance(feedback_root, dict) else {}
        feedbacks = []
        if isinstance(dialogue_feedbacks, dict):
            feedbacks.extend(_feedback_items(dialogue_feedbacks.get(str(turn_index)) or {}))
            if not feedbacks:
                feedbacks.extend(_feedback_items(dialogue_feedbacks.get(str(idx + 1)) or {}))
        rows.extend(_rows_with_feedback(base_row, feedbacks, prediction, stage))

    return rows, UPLOADED_CONVERSATION_EXPORT_COLUMNS


def _generated_task_type_map(tasks):
    mapping = {}
    for task in tasks:
        task_id = _safe_str(task.get("task_ID") or task.get("id"))
        if task_id:
            mapping[task_id] = _safe_str(task.get("conversation_type"))
    return mapping


def _generated_messages_previous_text(task_id, conversation_type):
    branch = "llm_conversations" if conversation_type == "human-ai" else "hh_conversations"
    raw = rtdb.reference(f"{branch}/{task_id}/messages").get() or {}
    messages = list(raw.values()) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    messages.sort(key=lambda item: item.get("timestamp", "") if isinstance(item, dict) else "")
    previous_by_index = {}
    prev_text = ""
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        previous_by_index[idx] = prev_text
        prev_text = msg.get("message") or prev_text
    return previous_by_index


def _export_generated_conversation_dataset(project_id, project, stage):
    tasks = get_project_tasks(project_id)
    selected = _get_project_selected_model(project_id, project, tasks)
    model_key = selected.get("key") or CONV_LOGREG_KEY
    raw = _generated_conversation_results_payload(project_id, model_key)
    if not raw and model_key != CONV_LOGREG_KEY:
        model_key = CONV_LOGREG_KEY
        raw = _generated_conversation_results_payload(project_id, model_key)
    if not isinstance(raw, dict):
        return [], GENERATED_CONVERSATION_EXPORT_COLUMNS

    type_map = _generated_task_type_map(tasks)
    all_turn_refs = []
    for task_id, node in raw.items():
        if not isinstance(node, dict):
            continue
        turns = _turns_to_list(node.get("turns") or {})
        for idx, turn in enumerate(turns):
            if isinstance(turn, dict):
                all_turn_refs.append((task_id, idx, turn))

    active_ids = set()
    if _uses_frozen_active_learning(model_key):
        model_version = _active_learning_model_version(project_id, model_key)
        frozen_selection = _freeze_active_learning_targets_if_missing(
            project_id,
            "generated_conversation",
            model_key,
            model_version,
            _generated_conversation_active_learning_candidates(project_id, model_key, raw)
        )
        active_ids = {
            _safe_str(target.get("source_id") or f"{target.get('conversation_id')}:{target.get('turn_index')}")
            for target in _frozen_target_values(frozen_selection)
        }

    rows = []
    previous_cache = {}
    for task_id, idx, turn in all_turn_refs:
        node = raw.get(task_id) or {}
        meta = node.get("meta") or {}
        conversation_id = meta.get("task_id") or task_id
        conversation_type = meta.get("conversation_type") or type_map.get(task_id)
        turn_index = int(turn.get("turn_index", idx) or idx)
        previous_key = (task_id, conversation_type)
        if previous_key not in previous_cache:
            previous_cache[previous_key] = _generated_messages_previous_text(task_id, conversation_type)
        prediction = turn.get("prediction")
        base_row = {
            "project_id": project_id,
            "task_id": task_id,
            "conversation_id": conversation_id,
            "conversation_type": conversation_type,
            "turn_index": turn_index,
            "sender": turn.get("sender"),
            "text": turn.get("text"),
            "previous_text": _first_present(turn.get("previous_text"), turn.get("prev_text"), previous_cache[previous_key].get(idx, "")),
            "prediction": prediction,
            "prediction_int": _first_present(turn.get("prediction_int"), _label_to_machinegen(prediction)),
            "ground_truth": _first_present(turn.get("ground_truth"), _normalize_binary_label(turn.get("gt"))),
            "p_machine": _owner_decimal(_first_present(turn.get("p_machine"), turn.get("ai_probability"), turn.get("probability"))),
            "confidence": _owner_decimal(turn.get("confidence")),
            "uncertainty": _owner_decimal(turn.get("uncertainty")),
            "selected_model": selected.get("name"),
            "model_version": selected.get("version") or "v1",
            "active_learning_selected": f"{task_id}:{turn_index}" in active_ids
        }

        feedback_root = node.get("turn_feedbacks") or {}
        feedbacks = []
        if isinstance(feedback_root, dict):
            feedbacks.extend(_feedback_items(feedback_root.get(str(turn_index)) or {}))
        rows.extend(_rows_with_feedback(base_row, feedbacks, prediction, stage))

    rows.sort(key=lambda item: (item.get("task_id", ""), int(item.get("turn_index") or 0)))
    return rows, GENERATED_CONVERSATION_EXPORT_COLUMNS


@app.route("/api/project/<project_id>/final_dataset_export", methods=["GET"])
def api_project_final_dataset_export(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Forbidden"}), 403

    project = get_project_basic_info(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    if project.get("owner_id") != session.get("uid"):
        return jsonify({"error": "Forbidden"}), 403

    fmt = _safe_str(request.args.get("format") or "json").strip().lower()
    stage = _safe_str(request.args.get("stage") or "detection").strip().lower()
    if fmt not in ("csv", "json"):
        return jsonify({"error": "Invalid format. Use csv or json."}), 400
    if stage not in ("detection", "feedback"):
        return jsonify({"error": "Invalid stage. Use detection or feedback."}), 400

    try:
        current_version, _, detection_ready = _current_version_with_rows(project_id)
        if not detection_ready:
            return jsonify({"ok": False, "error": "No detection results found"}), 404

        payload = _build_enhanced_dataset(project_id, current_version)
        rows = payload.get("rows") or []
        columns = _enhanced_dataset_columns((payload.get("metadata") or {}).get("project_type"))
        if not rows:
            return jsonify({"ok": False, "error": "No detection results found"}), 404

        if fmt == "csv":
            return _safe_csv_response(rows, _owner_export_filename(project, "enhanced_dataset", "csv"), columns)

        response_payload = {
            "ok": True,
            "project_id": project_id,
            "stage": "enhanced_dataset",
            "format": "json",
            "version_id": current_version,
            "metadata": payload.get("metadata") or {},
            "row_count": len(rows),
            "rows": rows
        }
        return _safe_json_export_response(response_payload, _owner_export_filename(project, "enhanced_dataset", "json"))
    except Exception as e:
        app.logger.exception("Final dataset export failed: %s", e)
        return jsonify({"error": "Failed to export final dataset"}), 500


@app.route("/api/project/<project_id>/version/<version_id>/enhanced_dataset", methods=["GET"])
def api_project_version_enhanced_dataset(project_id, version_id):
    if not session.get("idToken"):
        return jsonify({"error": "Forbidden"}), 403

    project = get_project_basic_info(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    fmt = _safe_str(request.args.get("format") or "json").strip().lower()
    if fmt not in ("csv", "json"):
        return jsonify({"error": "Invalid format. Use csv or json."}), 400

    try:
        payload = _build_enhanced_dataset(project_id, version_id)
        if not payload.get("rows"):
            return jsonify({
                "ok": False,
                "error": "Submit detection results before exporting enhanced dataset.",
                "details": "Detection version is empty or missing."
            }), 404

        if fmt == "csv":
            return _safe_csv_response(
                payload["rows"],
                f"trustlens_{project_id}_{version_id}_enhanced_dataset.csv",
                columns=_enhanced_dataset_columns((payload.get("metadata") or {}).get("project_type"))
            )

        return _safe_json_export_response(
            payload,
            f"trustlens_{project_id}_{version_id}_enhanced_dataset.json"
        )
    except Exception as e:
        app.logger.exception("Enhanced dataset build failed: %s", e)
        return jsonify({"error": "Failed to build enhanced dataset"}), 500


@app.route("/api/project/<project_id>/current_version", methods=["GET"])
def api_project_current_version(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Forbidden"}), 403

    project = get_project_basic_info(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    current_version, registry, detection_ready = _current_version_with_rows(project_id)
    return jsonify({
        "ok": True,
        "project_id": project_id,
        "current_version": current_version,
        "detection_ready": detection_ready,
        "message": "" if detection_ready else "Submit detection results before exporting enhanced dataset.",
        "versions": registry.get("versions") if isinstance(registry.get("versions"), dict) else {}
    })


@app.route("/api/project/<project_id>/versions", methods=["GET"])
def api_project_versions(project_id):
    project, error = _project_owner_required(project_id)
    if error:
        return error
    return jsonify({
        "ok": True,
        "project_id": project_id,
        "registry": _versions_response(project_id)
    }), 200


@app.route("/api/project/<project_id>/version/<version_id>/submit_feedback", methods=["POST"])
def api_project_version_submit_feedback(project_id, version_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401
    project = get_project_basic_info(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    try:
        result = _finalize_feedback_for_examiner(project_id, version_id, session.get("uid"))
        return jsonify({"ok": True, "project_id": project_id, "version_id": version_id, **result}), 200
    except PermissionError as e:
        return jsonify({"ok": False, "error": str(e)}), 403
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 409
    except Exception as e:
        app.logger.exception("Feedback final submit failed: %s", e)
        return jsonify({"ok": False, "error": "Failed to submit feedback task"}), 500


@app.route("/api/project/<project_id>/version/<version_id>/owner_notes", methods=["GET", "POST"])
def api_project_version_owner_notes(project_id, version_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401
    project = get_project_basic_info(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    notes_ref = rtdb.reference(f"owner_notes/{project_id}/{version_id}")
    if request.method == "GET":
        notes = notes_ref.get() or {}
        return jsonify({"ok": True, "project_id": project_id, "version_id": version_id, "notes": notes if isinstance(notes, dict) else {}}), 200

    if project.get("owner_id") != session.get("uid"):
        return jsonify({"error": "Forbidden"}), 403
    payload = request.get_json() or {}
    text = _safe_str(payload.get("text") or payload.get("note")).strip()
    note = {
        "text": text,
        "updated_at": _now_utc_iso(),
        "updated_by": session.get("uid")
    }
    notes_ref.set(_owner_json_value(note))
    return jsonify({"ok": True, "project_id": project_id, "version_id": version_id, "notes": note}), 200


@app.route("/api/project/<project_id>/versions/<version_id>/close", methods=["POST"])
def api_project_version_close(project_id, version_id):
    project, error = _project_owner_required(project_id)
    if error:
        return error

    registry = _normalize_version_registry(project_id)
    versions = registry.get("versions") or {}
    if version_id not in versions:
        return jsonify({"error": "Version not found"}), 404

    snapshot = rtdb.reference(f"evaluation_snapshots/{project_id}/{version_id}").get()
    if not isinstance(snapshot, dict) or not snapshot:
        return jsonify({"error": "Evaluation snapshot is required before closing this version"}), 409

    now_iso = _now_utc_iso()
    record = dict(versions.get(version_id) or {})
    record["version_id"] = version_id
    record["status"] = "evaluation_closed"
    record["closed_at"] = record.get("closed_at") or now_iso
    record["closed_by"] = session.get("uid")
    record["metrics_available"] = True
    versions[version_id] = record
    registry["versions"] = versions
    registry["latest_closed_version"] = version_id
    _save_version_registry(project_id, registry)
    return jsonify({"ok": True, "project_id": project_id, "version_id": version_id, "status": "evaluation_closed"}), 200


@app.route("/api/project/<project_id>/versions/<version_id>/reopen", methods=["POST"])
def api_project_version_reopen(project_id, version_id):
    project, error = _project_owner_required(project_id)
    if error:
        return error

    registry = _normalize_version_registry(project_id)
    versions = registry.get("versions") or {}
    source_record = versions.get(version_id)
    if not isinstance(source_record, dict):
        return jsonify({"error": "Version not found"}), 404
    if _safe_str(source_record.get("status")) != "evaluation_closed":
        return jsonify({"error": "Only closed evaluation versions can be reopened"}), 409
    if not _active_learning_retraining_supported(source_record.get("model_key")):
        return jsonify({"error": "Reopen cycles are currently supported for Logistic Regression only"}), 409

    enhanced_payload = _build_enhanced_dataset(project_id, version_id)
    enhanced_rows = enhanced_payload.get("rows") or []
    if not enhanced_rows:
        return jsonify({"error": "Enhanced dataset is empty for the selected version"}), 409

    metadata, _old_rows = _load_detection_version(project_id, version_id)
    project_type = metadata.get("project_type") or _official_project_type(project)
    new_version_id = _next_version_id(project_id)
    now_iso = _now_utc_iso()
    model_key = source_record.get("model_key") or metadata.get("model_key") or "logistic"
    model_name = source_record.get("model_name") or metadata.get("model_name") or _official_model_name(model_key)
    rows = _reopened_rows_from_enhanced(enhanced_rows, new_version_id, now_iso)
    candidates = _review_candidates_from_rows(rows, project_type)
    assigned_examiner_ids = source_record.get("examiner_ids")
    if not isinstance(assigned_examiner_ids, list):
        assigned_examiner_ids = []
        source_task_id = source_record.get("task_id") or metadata.get("task_id")
        if source_task_id:
            task_doc = db.collection("tasks").document(source_task_id).get()
            if task_doc.exists:
                assigned_examiner_ids = (task_doc.to_dict() or {}).get("examiner_ids") or []

    frozen_selection = _freeze_active_learning_targets_if_missing(
        project_id,
        project_type,
        model_key,
        new_version_id,
        candidates,
        assigned_examiner_ids
    )
    selected_ids = _frozen_target_sample_ids(frozen_selection)
    reopened_task_ids = _reset_labeling_tasks_for_reopen(project_id, new_version_id)
    for row in rows:
        if _safe_str(row.get("sample_id")) in selected_ids:
            row["review_target"] = True
            row["active_learning_state"] = "pending"
        else:
            row["review_target"] = False
            row["active_learning_state"] = "not_selected_for_review"

    snapshot = _create_detection_version_snapshot(
        project_id,
        source_record.get("task_id") or metadata.get("task_id"),
        model_key,
        model_name,
        session.get("uid"),
        analysis_run_id=metadata.get("analysis_run_id"),
        version_id=new_version_id,
        rows_override=rows,
        status="feedback_in_progress"
    )

    new_record = _version_record(project_id, new_version_id)
    new_record.update({
        "version_id": new_version_id,
        "status": "feedback_in_progress",
        "previous_version": version_id,
        "reopened_from": version_id,
        "reopened_at": now_iso,
        "reopened_by": session.get("uid"),
        "examiner_ids": assigned_examiner_ids,
        "labeling_task_ids": reopened_task_ids,
        "review_target_count": len(selected_ids),
        "metrics_available": False,
        "active_learning_supported": True
    })
    versions = _normalize_version_registry(project_id).get("versions") or {}
    versions[new_version_id] = new_record
    registry = _normalize_version_registry(project_id)
    registry["versions"] = versions
    registry["previous_version"] = version_id
    registry["current_version"] = new_version_id
    registry["latest_closed_version"] = version_id
    _save_version_registry(project_id, registry)

    return jsonify({
        "ok": True,
        "project_id": project_id,
        "previous_version": version_id,
        "current_version": new_version_id,
        "new_version": new_version_id,
        "review_target_count": len(selected_ids),
        "labeling_task_ids": reopened_task_ids,
        "snapshot_row_count": len((snapshot.get("rows") or {}))
    }), 200


@app.route("/api/project/<project_id>/version/<version_id>/evaluation_snapshot", methods=["GET"])
def api_project_version_evaluation_snapshot(project_id, version_id):
    if not session.get("idToken"):
        return jsonify({"error": "Forbidden"}), 403

    project = get_project_basic_info(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    try:
        rebuild = _safe_str(request.args.get("rebuild")).strip().lower() in ("1", "true", "yes")
        version_record = _version_record(project_id, version_id)
        version_status = _safe_str(version_record.get("status")).strip().lower()
        if rebuild and version_status == "feedback_in_progress" and _frozen_target_values(_load_frozen_active_learning_targets(project_id, version_id)):
            return jsonify({
                "ok": False,
                "error": "Feedback is still in progress. Submit the feedback task before building evaluation."
            }), 409
        if not rebuild:
            existing = rtdb.reference(f"evaluation_snapshots/{project_id}/{version_id}").get()
            if isinstance(existing, dict) and (existing.get("stale") or existing.get("rebuild_required")):
                return jsonify({
                    "ok": False,
                    "error": "Rebuild required after new submitted feedback.",
                    "rebuild_required": True,
                    "stale_reason": existing.get("stale_reason")
                }), 409
            if not _evaluation_snapshot_complete(existing):
                return jsonify({
                    "ok": False,
                    "error": "Evaluation snapshot is not available yet. Run evaluation after feedback is completed."
                }), 404
        snapshot = _build_evaluation_snapshot(project_id, version_id, rebuild=rebuild)
        return jsonify({"ok": True, "snapshot": snapshot}), 200
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 409
    except Exception as e:
        metadata, rows = _load_detection_version(project_id, version_id)
        context = _evaluation_snapshot_debug_context(project_id, version_id, metadata, rows)
        app.logger.exception("Evaluation snapshot failed for %s/%s: %s | context=%s", project_id, version_id, e, context)
        return jsonify({
            "ok": False,
            "error": "Failed to build evaluation snapshot",
            "detail": str(e),
            "context": context
        }), 500


@app.route("/api/project/<project_id>/report.pdf", methods=["GET"])
def api_project_pdf_report(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Forbidden"}), 403

    project = get_project_basic_info(project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404
    if project.get("owner_id") != session.get("uid"):
        return jsonify({"error": "Forbidden"}), 403

    try:
        summary_payload = build_owner_results_summary(project_id)
        if not summary_payload:
            return jsonify({"error": "Project not found"}), 404

        owner_name, owner_email = _owner_report_owner_info(session.get("uid"))
        pdf_bytes = _owner_build_pdf_report(summary_payload, owner_name, owner_email)
        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{_owner_report_filename(project)}"'}
        )
    except ImportError:
        app.logger.exception("PDF report generation failed because reportlab is missing.")
        return jsonify({"error": "PDF library is not installed. Install reportlab from requirements.txt."}), 500
    except Exception as e:
        app.logger.exception("PDF report generation failed: %s", e)
        return jsonify({"error": "Failed to generate PDF report"}), 500


@app.route("/api/project/<project_id>/active_learning_export", methods=["GET"])
def api_project_active_learning_export(project_id):
    ctx, err = _ensure_project_access(project_id)
    if err:
        return err

    current_version, _, detection_ready = _current_version_with_rows(project_id)
    if not detection_ready:
        return jsonify({
            "ok": False,
            "project_id": project_id,
            "error": "Submit detection results before exporting enhanced dataset.",
            "rows": [],
            "total_rows": 0
        }), 404

    payload = _build_enhanced_dataset(project_id, current_version)
    rows = payload.get("rows") or []
    metadata = payload.get("metadata") or {}

    return jsonify({
        "ok": True,
        "project_id": project_id,
        "version_id": current_version,
        "project_type": metadata.get("project_type"),
        "model_key": metadata.get("model_key"),
        "exported_at": _now_utc_iso(),
        "rows": rows,
        "total_rows": len(rows)
    }), 200


@app.route("/api/task/<task_id>/uploaded_conversation_feedback_list", methods=["GET"])
def api_uploaded_conversation_feedback_list(task_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    uid = session.get("uid")

    task_doc = db.collection("tasks").document(task_id).get()
    if not task_doc.exists:
        return jsonify({"error": "Task not found"}), 404

    task_data = task_doc.to_dict() or {}
    if task_data.get("task_type") != "labeling":
        return jsonify({"error": "Task is not labeling"}), 400
    if uid not in (task_data.get("examiner_ids") or []):
        return jsonify({"error": "Forbidden"}), 403

    project_id = task_data.get("project_ID")
    selected_model, model_key, model_label = _pick_conversation_model_for_project(project_id)
    if not selected_model:
        return jsonify({
            "waiting": True,
            "message": "Waiting for model selection task to be completed"
        }), 200

    model_key = _uploaded_feedback_model_key(model_key)
    run_id, run_ref, summary = _uploaded_conversation_run(project_id, model_key)
    if not run_ref:
        return jsonify({
            "waiting": True,
            "message": "Analysis not ready yet, please run conversation detection first"
        }), 200

    dialogues = run_ref.child("dialogues").get() or []
    if isinstance(dialogues, dict):
        dialogues = list(dialogues.values())

    key_map = run_ref.child("dialogue_key_map").get() or {}
    all_turns = run_ref.child("dialogue_turns").get() or {}
    all_feedbacks = run_ref.child("turn_feedbacks").get() or {}

    items = []
    reviewed_dialogues = 0

    for order_index, dialogue in enumerate(dialogues):
        if not isinstance(dialogue, dict):
            continue

        dialogue_id = _safe_str(dialogue.get("dialogue_id")) or f"dialogue_{order_index + 1}"
        safe_key = key_map.get(dialogue_id) or _rtdb_safe_key(dialogue_id)

        turns_raw = all_turns.get(safe_key) or []
        turns = list(turns_raw.values()) if isinstance(turns_raw, dict) else (turns_raw if isinstance(turns_raw, list) else [])
        turns.sort(key=lambda x: int((x or {}).get("turn_index", 0) or 0))

        feedback_root = all_feedbacks.get(safe_key) or {}
        clean_turns = []
        feedback_users_map = {}
        reviewed_turns = 0
        confs = []
        uncertainties = []

        for idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue

            ui_turn_index = idx + 1
            raw_prediction = _safe_str(turn.get("prediction"))
            prediction = "AI" if raw_prediction.lower() in ("ai", "machine", "machine-generated") else "Human"

            confidence = turn.get("confidence")
            confidence_pct = None
            try:
                confidence_pct = float(confidence)
                if confidence_pct <= 1:
                    confidence_pct *= 100.0
                confs.append(confidence_pct)
            except Exception:
                confidence_pct = None

            uncertainty = turn.get("uncertainty")
            uncertainty_pct = None
            try:
                uncertainty_pct = float(uncertainty)
                if uncertainty_pct <= 1:
                    uncertainty_pct *= 100.0
                uncertainties.append(uncertainty_pct)
            except Exception:
                uncertainty_pct = None

            turn_feedbacks = feedback_root.get(str(ui_turn_index), {}) if isinstance(feedback_root, dict) else {}
            if not isinstance(turn_feedbacks, dict):
                turn_feedbacks = {}

            my_feedback = turn_feedbacks.get(uid)
            turn_feedback_users = []

            for feedback_uid, feedback_data in turn_feedbacks.items():
                feedback_data = feedback_data or {}
                feedback_name = feedback_data.get("examiner_name") or _feedback_examiner_name(feedback_uid)
                user_item = {
                    "uid": feedback_uid,
                    "examiner_uid": feedback_uid,
                    "name": feedback_name,
                    "examiner_name": feedback_name,
                    "label": feedback_data.get("label"),
                    "explanation": feedback_data.get("explanation", ""),
                    "agreed_with_model": bool(feedback_data.get("agreed_with_model", False)),
                    "submitted_at": feedback_data.get("submitted_at")
                }
                turn_feedback_users.append(user_item)
                feedback_users_map[feedback_uid] = {
                    "uid": feedback_uid,
                    "name": feedback_name
                }

            if turn_feedback_users:
                reviewed_turns += 1

            clean_turns.append({
                "turn_index": ui_turn_index,
                "source_turn_index": turn.get("turn_index"),
                "sender": turn.get("sender", ""),
                "text": turn.get("text", ""),
                "prediction": prediction,
                "gt": turn.get("ground_truth"),
                "confidence": round(confidence_pct, 2) if isinstance(confidence_pct, (int, float)) else None,
                "uncertainty": round(uncertainty_pct, 2) if isinstance(uncertainty_pct, (int, float)) else None,
                "turn_locked": bool(turn_feedback_users),
                "turn_feedback": turn_feedback_users[0] if turn_feedback_users else None,
                "my_feedback": my_feedback,
                "feedback_users": turn_feedback_users
            })

        total_turns = len(clean_turns)
        conversation_locked = total_turns > 0 and reviewed_turns >= total_turns
        if conversation_locked:
            reviewed_dialogues += 1

        items.append({
            "conversation_id": dialogue_id,
            "task_name": f"Dialogue {dialogue_id}",
            "order_index": order_index,
            "turns_count": total_turns,
            "ai_percentage": float(dialogue.get("ai_percentage") or 0),
            "human_percentage": float(dialogue.get("human_percentage") or 0),
            "confidence": round(sum(confs) / len(confs), 2) if confs else 0.0,
            "uncertainty": round(sum(uncertainties) / len(uncertainties), 2) if uncertainties else 0.0,
            "has_feedback": reviewed_turns > 0,
            "conversation_locked": conversation_locked,
            "feedback_users": list(feedback_users_map.values()),
            "turns": clean_turns
        })

    active_learning_enabled = _uses_frozen_active_learning(model_key)
    if active_learning_enabled:
        model_version = _active_learning_model_version(project_id, model_key)
        frozen_selection = _freeze_active_learning_targets_if_missing(
            project_id,
            "uploaded_conversation",
            model_key,
            model_version,
            _uploaded_conversation_active_learning_candidates(project_id, model_key, run_id, run_ref),
            task_data.get("examiner_ids") or []
        )
        items, active_learning_info = _apply_frozen_active_learning_turn_selection(
            items,
            frozen_selection,
            "uploaded_conversation"
        )
    else:
        items, active_learning_info = _apply_active_learning_turn_selection(items, False)

    total_dialogues = len(items)
    reviewed_dialogues = sum(1 for item in items if item.get("conversation_locked"))

    if active_learning_enabled:
        status_total = active_learning_info["total"]
        status_reviewed = active_learning_info["reviewed"]
    else:
        status_total = total_dialogues
        status_reviewed = reviewed_dialogues

    new_status = "completed" if status_total > 0 and status_reviewed >= status_total else ("progress" if status_reviewed > 0 else "pending")
    if (task_data.get("status") or "").strip().lower() != new_status:
        db.collection("tasks").document(task_id).update({"status": new_status})

    return jsonify({
        "waiting": False,
        "project_id": project_id,
        "task_id": task_id,
        "task_status": new_status,
        "feedback_locked": new_status == "completed",
        "source": "uploaded",
        "selected_model": selected_model,
        "selected_model_name": model_label,
        "run_id": run_id,
        "active_learning": active_learning_info,
        "progress": {
            "reviewed": status_reviewed,
            "total": status_total,
            "unit": active_learning_info.get("unit", "dialogues")
        },
        "items": items
    }), 200


@app.route("/api/task/<task_id>/uploaded_conversation_feedback/<dialogue_id>/turn/<int:turn_index>/submit", methods=["POST"])
def api_submit_uploaded_conversation_turn_feedback(task_id, dialogue_id, turn_index):
    try:
        if not session.get("idToken"):
            return jsonify({"error": "Unauthorized"}), 401

        uid = session.get("uid")
        data = request.get_json(silent=True) or {}

        agree_with_model = bool(data.get("agree_with_model", False))
        label = (data.get("label") or "").strip()
        explanation = (data.get("explanation") or "").strip()

        if turn_index <= 0:
            return jsonify({"error": "Invalid turn_index"}), 400

        task_doc = db.collection("tasks").document(task_id).get()
        if not task_doc.exists:
            return jsonify({"error": "Task not found"}), 404

        task_data = task_doc.to_dict() or {}
        if task_data.get("task_type") != "labeling":
            return jsonify({"error": "Task is not labeling"}), 400
        if uid not in (task_data.get("examiner_ids") or []):
            return jsonify({"error": "Forbidden"}), 403
        if _labeling_task_completed(task_data):
            return jsonify({"error": "Feedback is locked because this labeling task is completed"}), 423

        project_id = task_data.get("project_ID")
        selected_model, model_key, _ = _pick_conversation_model_for_project(project_id)
        if not selected_model:
            return jsonify({"error": "Model selection is not completed yet"}), 400
        if _version_is_closed(project_id, _active_learning_model_version(project_id, model_key)):
            return jsonify({"error": "Feedback is locked because this evaluation version is closed"}), 423

        model_key = _uploaded_feedback_model_key(model_key)
        run_id, run_ref, _ = _uploaded_conversation_run(project_id, model_key)
        if not run_ref:
            return jsonify({"error": "Analysis results not found"}), 404

        key_map = run_ref.child("dialogue_key_map").get() or {}
        safe_key = key_map.get(dialogue_id) or _rtdb_safe_key(dialogue_id)
        turns_raw = run_ref.child("dialogue_turns").child(safe_key).get() or []
        turns = list(turns_raw.values()) if isinstance(turns_raw, dict) else (turns_raw if isinstance(turns_raw, list) else [])

        if turn_index > len(turns):
            return jsonify({"error": "Turn not found"}), 404

        target_turn = turns[turn_index - 1] or {}
        model_prediction = "AI" if _safe_str(target_turn.get("prediction")).lower() in ("ai", "machine", "machine-generated") else "Human"

        if agree_with_model:
            label = model_prediction
            if not explanation:
                explanation = "Agreed with model prediction."
        else:
            if label not in ["Human", "AI"]:
                return jsonify({"error": "Invalid label"}), 400
            target_prediction = _label_text_only(target_turn.get("prediction"))
            if label != target_prediction and not explanation:
                return jsonify({"error": "Explanation is required when correcting the model"}), 400

        feedback_ref = run_ref.child("turn_feedbacks").child(safe_key).child(str(turn_index))
        existing_feedbacks = feedback_ref.get() or {}
        if not isinstance(existing_feedbacks, dict):
            existing_feedbacks = {}
        existing_owner_uid = _feedback_owner_uid(existing_feedbacks)
        existing_user_feedback = _feedback_record_for_uid(existing_feedbacks, uid)
        if existing_owner_uid and existing_owner_uid != uid:
            return jsonify({"error": "Feedback already submitted for this turn and can only be edited by the original reviewer"}), 409

        project = get_project_basic_info(project_id) or {}
        detection_snapshot = _find_detection_snapshot_for_feedback(
            "uploaded_conversation",
            project_id,
            project=project,
            dialogue_id=dialogue_id,
            turn_index=turn_index,
            target_turn=target_turn
        )
        sample_id = _make_active_learning_sample_id(
            "uploaded_conversation",
            row_id=target_turn.get("row_id") or target_turn.get("source_row_id"),
            dialogue_id=dialogue_id,
            turn_index=turn_index
        )
        try:
            _ensure_feedback_can_be_saved(project_id, detection_snapshot.get("version_id") or _active_learning_model_version(project_id, model_key), sample_id, uid)
        except ValueError as e:
            return jsonify({"error": str(e)}), 423

        examiner_name = _feedback_examiner_name(uid)
        now_iso = datetime.utcnow().isoformat() + "Z"
        payload = {
            "examiner_uid": uid,
            "examiner_name": examiner_name,
            "agreed_with_model": agree_with_model,
            "label": label,
            "explanation": explanation,
            "submitted_at": now_iso,
            "status": "draft_saved",
            "lifecycle_status": "draft_saved",
            "locked": False
        }
        payload, is_edit = _prepare_feedback_payload(existing_user_feedback, payload, uid, examiner_name, now_iso)

        feedback_ref.child(uid).set(payload)

        try:
            _write_active_learning_feedback(project_id, sample_id, detection_snapshot, payload)
            if _uses_frozen_active_learning(model_key):
                model_version = _active_learning_model_version(project_id, model_key)
                _mark_frozen_active_learning_target_reviewed(
                    project_id,
                    model_version,
                    sample_id,
                    uid,
                    payload.get("submitted_at"),
                    {"dialogue_id": dialogue_id, "turn_index": turn_index}
                )
        except Exception as normalized_error:
            app.logger.exception("Failed to write normalized uploaded conversation feedback: %s", normalized_error)

        return jsonify({
            "message": _feedback_write_message(is_edit),
            "updated": is_edit,
            "conversation_id": dialogue_id,
            "turn_index": turn_index,
            "selected_model": selected_model,
            "run_id": run_id
        }), 200

    except Exception as e:
        app.logger.exception("api_submit_uploaded_conversation_turn_feedback failed: %s", e)
        return jsonify({"error": "Server error while saving turn feedback"}), 500



@app.route("/project/<project_id>/analysis/examiner")
def analysis_examiner_redirect(project_id):
    return redirect(url_for("results_con", projectId=project_id))

@app.route("/api/project/<project_id>/examiner_progress", methods=["GET"])
def api_examiner_progress(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Project not found"}), 404

    proj_data = proj_doc.to_dict()
    dataset_id = proj_data.get("dataset_id")
    if not dataset_id:
        return jsonify({"progress": [], "total_articles": 0}), 200

    # نجيب كل المقالات ونحسب كم feedback لكل examiner
    try:
        ref = rtdb.reference(f"datasets/uploaded_news/{dataset_id}")
        snapshot = ref.get() or {}
        total = len(snapshot)

        # نجمع كل الـ examiner_uid اللي حطوا feedback
        examiner_counts = {}
        for push_id, article_data in snapshot.items():
            if not isinstance(article_data, dict):
                continue
            feedback = article_data.get("feedback")
            if not feedback or not isinstance(feedback, dict):
                continue
            uid = feedback.get("examiner_uid")
            if uid:
                examiner_counts[uid] = examiner_counts.get(uid, 0) + 1

        return jsonify({
            "total_articles": total,
            "examiner_counts": examiner_counts
        }), 200

    except Exception as e:
        app.logger.exception("examiner_progress failed: %s", e)
        return jsonify({"error": "Failed"}), 500
@app.route("/api/project/<project_id>/rate_examiner", methods=["POST"])
def api_rate_examiner(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401
    uid = session.get("uid")
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists or proj_doc.to_dict().get("owner_id") != uid:
        return jsonify({"error": "Forbidden"}), 403
    data = request.get_json() or {}
    examiner_id = data.get("examiner_id")
    stars = int(data.get("stars", 0))
    comment = (data.get("comment") or "").strip()
    if not examiner_id or not (1 <= stars <= 5):
        return jsonify({"error": "Invalid data"}), 400
    db.collection("projects").document(project_id)\
    .collection("assigned_examiners").document(examiner_id).set({
        "examiner_id": examiner_id,
        "stars": stars,
        "comment": comment,
        "rated_by": uid,
        "rated_at": datetime.utcnow().isoformat() + "Z"
    })
    return jsonify({"message": "Rating saved"}), 200


@app.route("/api/project/<project_id>/examiner_feedback_count", methods=["GET"])
def api_examiner_feedback_count(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401
    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return jsonify({"error": "Not found"}), 404
    proj_data = proj_doc.to_dict()
    category = (proj_data.get("category") or "").lower()
    is_generated = bool(proj_data.get("generated_from_scratch", False))
    counts = {}
    try:
        # Article
        if "article" in category or "news" in category:
            dataset_id = proj_data.get("dataset_id")
            if dataset_id:
                ref = rtdb.reference(f"datasets/uploaded_news/{dataset_id}")
                snapshot = ref.get() or {}
                total = len(snapshot)
                for push_id, article_data in snapshot.items():
                    if not isinstance(article_data, dict):
                        continue
                    feedback = article_data.get("feedback")
                    if feedback and isinstance(feedback, dict):
                        eid = feedback.get("examiner_uid")
                        if eid:
                            counts[eid] = counts.get(eid, 0) + 1
               # Conversation (Generated only)
        else:
            if is_generated:
                has_labeling_task = False

                tasks = db.collection("tasks").where("project_ID", "==", project_id).stream()
                for t in tasks:
                    td = t.to_dict() or {}
                    task_type = (td.get("task_type") or "").lower()
                    if task_type == "labeling":
                        has_labeling_task = True
                        break

                if has_labeling_task:
                    _, model_key, _ = _pick_conversation_model_for_project(project_id)

                    model_keys_to_try = [model_key] if model_key else [CONV_LOGREG_KEY, CONV_RNN_KEY]

                    for mk in model_keys_to_try:
                        raw = _generated_conversation_results_payload(project_id, mk) or {}

                        if not isinstance(raw, dict) or not raw:
                            continue

                        for _, node in raw.items():
                            if not isinstance(node, dict):
                                continue

                            tf_root = node.get("turn_feedbacks") or {}

                            if isinstance(tf_root, dict):
                                turn_feedback_items = tf_root.values()
                            elif isinstance(tf_root, list):
                                turn_feedback_items = [x for x in tf_root if isinstance(x, dict)]
                            else:
                                turn_feedback_items = []

                            for tf in turn_feedback_items:
                                for eid in tf.keys():
                                    counts[eid] = counts.get(eid, 0) + 1

                        break
            else:
                _, model_key, _ = _pick_conversation_model_for_project(project_id)
                model_keys_to_try = [model_key] if model_key else [CONV_LOGREG_KEY, CONV_RNN_KEY]

                for mk in model_keys_to_try:
                    mk = _uploaded_feedback_model_key(mk)
                    base_ref = rtdb.reference(f"analysis_results/conversations/{mk}/{project_id}")
                    run_id = _safe_str(base_ref.child("latest_run_id").get())
                    if not run_id:
                        continue

                    feedback_root = base_ref.child("runs").child(run_id).child("turn_feedbacks").get() or {}
                    if not isinstance(feedback_root, dict):
                        continue

                    for dialogue_feedbacks in feedback_root.values():
                        if not isinstance(dialogue_feedbacks, dict):
                            continue
                        for turn_feedbacks in dialogue_feedbacks.values():
                            if not isinstance(turn_feedbacks, dict):
                                continue
                            for eid in turn_feedbacks.keys():
                                counts[eid] = counts.get(eid, 0) + 1

                    break

        # ratings
        ratings = {}
        rating_docs = db.collection("projects").document(project_id)\
        .collection("assigned_examiners").stream()
        for r in rating_docs:
            rd = r.to_dict() or {}
            if rd.get("stars"):
              ratings[r.id] = {
                "stars": rd.get("stars", 0),
                "comment": rd.get("comment", "")
            }
        return jsonify({
            "counts": counts,
            "ratings": ratings
        }), 200
    except Exception as e:
        app.logger.exception("examiner_feedback_count failed: %s", e)
        return jsonify({"error": "Failed"}), 500
    
    
@app.route("/api/project/<project_id>/my_rating", methods=["GET"])
def api_my_rating(project_id):
    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401
    uid = session.get("uid")
    doc = db.collection("projects").document(project_id)\
        .collection("assigned_examiners").document(uid).get()
    if not doc.exists:
        return jsonify({"rated": False}), 200
    rd = doc.to_dict() or {}
    return jsonify({
        "rated": True,
        "stars": rd.get("stars", 0),
        "comment": rd.get("comment", "")
    }), 200


# ==================================================
# Open Uploaded Conversation Results Page
# ==================================================
@app.route("/projectdetailsexaminer/<project_id>/conversation-analysis")
def conversation_analysis_page_examiner(project_id):
    if not session.get("idToken"):
        return redirect(url_for("login_page"))

    examiner_uid = session.get("uid")
    task_id = request.args.get("taskId") or request.args.get("task_id") or ""

    inv = (
        db.collection("invitations")
        .where("project_id", "==", project_id)
        .where("examiner_id", "==", examiner_uid)
        .where("status", "==", "accepted")
        .limit(1)
        .get()
    )
    if not inv:
        abort(403)

    if task_id:
        task_doc = db.collection("tasks").document(task_id).get()
        if not task_doc.exists:
            abort(404)
        task_data = task_doc.to_dict() or {}
        examiner_ids = task_data.get("examiner_ids") or []
        if task_data.get("project_ID") != project_id or task_data.get("task_type") != "model_selection":
            abort(403)
        if len(examiner_ids) != 1 or examiner_uid not in examiner_ids:
            abort(403)

    user_doc = db.collection("users").document(examiner_uid).get()
    full_name = "User"
    if user_doc.exists:
        prof = user_doc.to_dict().get("profile", {})
        full_name = f"{prof.get('firstName','')} {prof.get('lastName','')}".strip() or "User"

    return render_template(
        "ConversationAnalysisResults.html",
        user_name=full_name,
        project_id=project_id,
        task_id=task_id
    )


# ===================================================================
# CONVERSATIONS Baseline Analysis (Uploaded dataset dashboard)
# ===================================================================

def _safe_str(x):
    try:
        return ("" if x is None else str(x)).strip()
    except Exception:
        return ""


def _as_int(x, default=0):
    try:
        return int(str(x).strip())
    except Exception:
        return default


def _normalize_gt(label):
    return _normalize_binary_label(label)


def _extract_conversation_fields(payload):
    dialogue_id = _payload_value(
        payload,
        "dialogue_id", "dialogueId", "Dialogue_ID",
        "conversation_id", "conversationId", "Conversation_ID",
        "chat_id", "Chat_ID",
        "thread_id", "Thread_ID",
        "session_id", "Session_ID",
        "id", "ID"
    )
    dialogue_id = _safe_str(dialogue_id)

    turn_index = _payload_value(
        payload,
        "turn_index", "turnIndex", "Turn_Index",
        "turn_id", "turnId", "Turn_ID",
        "turn_number", "turnNumber", "Turn_Number",
        "utterance_id", "Utterance_ID",
        "message_index", "messageIndex"
    )
    turn_index = None if turn_index is None or str(turn_index).strip() == "" else _as_int(turn_index, default=None)

    text = _payload_value(
        payload,
        "text", "Text",
        "utterance", "Utterance",
        "message", "Message",
        "turn", "Turn",
        "content", "Content",
        "reply", "Reply",
        "msg", "Msg"
    )
    text = _safe_str(text)

    previous_text = _safe_str(_payload_value(
        payload,
        "previous_text", "previousText", "Previous_Text",
        "previous_turn", "previousTurn", "Previous_Turn",
        "prev_text", "prevText", "Prev_Text"
    ))

    sender = _payload_value(
        payload,
        "sender", "Sender",
        "role", "Role",
        "author", "Author",
        "from", "From"
    )
    sender = _safe_str(sender)

    gt_norm = _extract_ground_truth_from_payload(payload)

    return dialogue_id, turn_index, text, gt_norm, sender, previous_text


def _build_prev_text_for_dialogue(turns_sorted, idx, max_chars=800):
    if idx <= 0:
        return ""
    prev = _safe_str(turns_sorted[idx - 1].get("text", ""))
    if len(prev) > max_chars:
        prev = prev[-max_chars:]
    return prev


def _final_label_rule(ai_pct):
    try:
        p = float(ai_pct)
    except Exception:
        p = 0.0

    if p <= 10:
        return "Human"
    if p >= 70:
        return "AI-heavy"
    return "Mixed"


def _now_utc_iso():
    return datetime.utcnow().isoformat() + "Z"


def _rtdb_safe_key(raw_id: str) -> str:
    s = _safe_str(raw_id) or "unknown_dialogue"
    s_clean = re.sub(r'[.#$\[\]/]', '_', s) or "unknown_dialogue"
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return f"{s_clean}__{h}"


def _ensure_project_access(project_id):
    if not session.get("idToken"):
        return None, (jsonify({"error": "Unauthorized"}), 401)

    uid = session.get("uid")

    proj_doc = db.collection("projects").document(project_id).get()
    if not proj_doc.exists:
        return None, (jsonify({"error": "Project not found"}), 404)

    proj_data = proj_doc.to_dict()
    is_owner = (proj_data.get("owner_id") == uid)

    is_examiner = False
    if not is_owner:
        inv_docs = list(
            db.collection("invitations")
            .where("project_id", "==", project_id)
            .where("examiner_id", "==", uid)
            .where("status", "==", "accepted")
            .limit(1)
            .stream()
        )
        is_examiner = len(inv_docs) > 0

    if not is_owner and not is_examiner:
        return None, (jsonify({"error": "Forbidden"}), 403)

    return {"uid": uid, "proj_data": proj_data}, None


def _predict_with_proba(text: str, previous_text: str = "", threshold: float = 0.5, model_key: str = CONV_LOGREG_KEY):
    p_machine = None
    pred_int = 0

    if model_key == CONV_RNN_KEY:
        seq = conv_rnn_tokenizer.texts_to_sequences([text or ""])
        x = pad_sequences(seq, maxlen=CONV_RNN_MAX_LEN, padding="post", truncating="post")
        raw_pred = conv_rnn_model.predict(x, verbose=0)
        arr = np.asarray(raw_pred)
        p_pos = float(arr[0, 1]) if (arr.ndim == 2 and arr.shape[1] == 2) else float(arr.reshape(-1)[0])
        p_machine = p_pos if CONV_RNN_AI_CLASS_IS_ONE else (1.0 - p_pos)
        p_machine = float(np.clip(p_machine, 0.0, 1.0))
        pred_int = 1 if p_machine >= float(threshold) else 0
        confidence, uncertainty = _confidence_uncertainty_from_prob(p_machine)
        return pred_int, p_machine, confidence, uncertainty

    df_in = pd.DataFrame([{"text": text or "", "prev_text": previous_text or ""}])
    try:
        probs = conv_logreg_model.predict_proba(df_in)[0]
        p_machine = _machine_probability_from_proba(conv_logreg_model, probs)
        if p_machine is not None:
            p_machine = float(np.clip(p_machine, 0.0, 1.0))
            pred_int = 1 if p_machine >= float(threshold) else 0
            confidence, uncertainty = _confidence_uncertainty_from_prob(p_machine)
            return pred_int, p_machine, confidence, uncertainty
    except Exception as e:
        app.logger.warning("Conversation predict_proba failed, falling back to label only: %s", e)

    out = predict_one_turn(text, previous_text)

    if isinstance(out, dict):
        if "pred_int" in out:
            pred_int = int(out.get("pred_int") or 0)
        elif "prediction_int" in out:
            pred_int = int(out.get("prediction_int") or 0)
        elif "pred" in out:
            pred_int = int(out.get("pred") or 0)
        elif "prediction" in out:
            pred_int = int(out.get("prediction") or 0)

        for k in ("p_machine", "proba", "prob", "score", "machine_proba"):
            if k in out and out.get(k) is not None:
                try:
                    p_machine = float(out.get(k))
                except Exception:
                    p_machine = None
                break

    elif isinstance(out, (tuple, list)) and len(out) >= 2:
        a, b = out[0], out[1]

        def _is_prob(x):
            try:
                fx = float(x)
                return 0.0 <= fx <= 1.0
            except Exception:
                return False

        if _is_prob(a) and not _is_prob(b):
            p_machine = float(a)
            pred_int = int(b)
        elif _is_prob(b) and not _is_prob(a):
            p_machine = float(b)
            pred_int = int(a)
        else:
            try:
                pred_int = int(a)
            except Exception:
                pred_int = 0
            p_machine = float(b) if _is_prob(b) else None

    elif isinstance(out, (float, int)) and not isinstance(out, bool):
        try:
            fx = float(out)
            if 0.0 <= fx <= 1.0 and fx not in (0.0, 1.0):
                p_machine = fx
            else:
                pred_int = int(fx)
        except Exception:
            pred_int = 0

    if p_machine is not None:
        pred_int = 1 if p_machine >= float(threshold) else 0
        confidence, uncertainty = _confidence_uncertainty_from_prob(p_machine)
    else:
        confidence = None
        uncertainty = None
        pred_int = 1 if int(pred_int) == 1 else 0

    return pred_int, p_machine, confidence, uncertainty


@app.route("/api/project/<project_id>/conversation_dataset", methods=["GET"])
def get_project_conversation_dataset(project_id):
    ctx, err = _ensure_project_access(project_id)
    if err:
        return err

    proj_data = ctx["proj_data"]
    dataset_id = proj_data.get("dataset_id")
    if not dataset_id:
        return jsonify({"error": "No dataset found for this project"}), 404

    try:
        ref = rtdb.reference(f"datasets/uploaded_conversations/{dataset_id}")
        snapshot = ref.get()

        if not snapshot:
            return jsonify({"dialogues": [], "total_rows": 0, "dataset_id": dataset_id}), 200

        temp_rows = []
        any_dialogue_id = False

        for row_order, (push_id, row_data) in enumerate(snapshot.items()):
            if not isinstance(row_data, dict):
                continue

            payload = row_data.get("payload", {}) or {}
            if not isinstance(payload, dict):
                continue

            dialogue_id, turn_index, text, gt, sender, previous_text = _extract_conversation_fields(payload)
            if not _safe_str(text):
                continue

            if _safe_str(dialogue_id):
                any_dialogue_id = True

            temp_rows.append({
                "id": push_id,
                "row_order": row_order,
                "dialogue_id": dialogue_id,
                "turn_index": turn_index,
                "text": text,
                "sender": sender,
                "previous_text": previous_text,
                "ground_truth": gt,
                "raw_payload": payload,
            })

        if not temp_rows:
            return jsonify({"dialogues": [], "total_rows": 0, "dataset_id": dataset_id}), 200

        rows = []
        for r in temp_rows:
            d_id = _safe_str(r.get("dialogue_id"))

            if not any_dialogue_id and not d_id:
                d_id = f"auto_dialogue_all_{dataset_id}"
            elif any_dialogue_id and not d_id:
                d_id = "unknown_dialogue"

            r["dialogue_id"] = d_id
            rows.append(r)

        dialogues_map = {}
        for r in rows:
            dialogues_map.setdefault(r["dialogue_id"], []).append(r)

        dialogues = []
        for d_id, turns in dialogues_map.items():
            turns_sorted = sorted(
                turns,
                key=lambda x: (x.get("turn_index") is None, x.get("turn_index") if x.get("turn_index") is not None else x.get("row_order", 0), x.get("row_order", 0))
            )
            for i, t in enumerate(turns_sorted):
                if t.get("turn_index") is None:
                    t["turn_index"] = i

            turns_sorted.sort(key=lambda x: (x.get("turn_index", 0), x.get("row_order", 0)))

            dialogues.append({
                "dialogue_id": d_id,
                "num_turns": len(turns_sorted),
                "turns": [
                    {
                        "turn_index": t.get("turn_index", 0),
                        "text_preview": (t.get("text", "")[:160] + ("..." if len(t.get("text", "")) > 160 else "")),
                        "sender": t.get("sender", ""),
                        "ground_truth": t.get("ground_truth"),
                        "row_id": t.get("id"),
                    }
                    for t in turns_sorted
                ]
            })

        dialogues.sort(key=lambda d: d.get("num_turns", 0), reverse=True)

        return jsonify({
            "dialogues": dialogues,
            "total_rows": len(rows),
            "total_dialogues": len(dialogues),
            "dataset_id": dataset_id,
            "note": ("No dialogue_id found in upload. Grouped all rows into one dialogue."
                     if not any_dialogue_id else "")
        }), 200

    except Exception as e:
        app.logger.exception("Failed to fetch conversation dataset: %s", e)
        return jsonify({"error": "Failed to fetch dataset"}), 500


@app.route("/api/project/<project_id>/analyze_conversations", methods=["POST"])
def analyze_all_conversations(project_id):
    ctx, err = _ensure_project_access(project_id)
    if err:
        return err

    uid = ctx["uid"]
    proj_data = ctx["proj_data"]

    req = request.get_json(silent=True) or {}
    task_id = _safe_str(req.get("task_id") or req.get("taskId") or request.args.get("task_id") or request.args.get("taskId"))
    task_ref, task_data, guard_error = _model_selection_task_guard(task_id, project_id=project_id, reject_completed=False)
    if guard_error:
        return guard_error
    task_finalized = bool(task_data.get("selected_model") or _safe_str(task_data.get("status")).lower() == "completed")

    dataset_id = proj_data.get("dataset_id")
    if not dataset_id:
        return jsonify({"error": "No dataset found"}), 404

    model_key = _normalize_conversation_model_key(req.get("model_key"), for_results=True)

    try:
        threshold = float(req.get("threshold", 0.5))
        if threshold < 0.0:
            threshold = 0.0
        if threshold > 1.0:
            threshold = 1.0
    except Exception:
        threshold = 0.5

    run_id = uuid.uuid4().hex[:10]
    analyzed_at = _now_utc_iso()

    try:
        ref = rtdb.reference(f"datasets/uploaded_conversations/{dataset_id}")
        snapshot = ref.get()
        if not snapshot:
            return jsonify({"error": "Dataset is empty"}), 404

        temp_rows = []
        any_dialogue_id = False

        for row_order, (push_id, row_data) in enumerate(snapshot.items()):
            if not isinstance(row_data, dict):
                continue

            payload = row_data.get("payload", {}) or {}
            if not isinstance(payload, dict):
                continue

            dialogue_id, turn_index, text, gt, sender, previous_text = _extract_conversation_fields(payload)
            text = _safe_str(text)
            if not text:
                continue

            if _safe_str(dialogue_id):
                any_dialogue_id = True

            temp_rows.append({
                "row_id": push_id,
                "row_order": row_order,
                "dialogue_id": dialogue_id,
                "turn_index": turn_index,
                "text": text,
                "sender": sender,
                "previous_text": previous_text,
                "gt": gt,
            })

        if not temp_rows:
            return jsonify({"error": "Dataset rows have no valid text"}), 404

        rows = []
        for r in temp_rows:
            d_id = _safe_str(r.get("dialogue_id"))

            if not any_dialogue_id and not d_id:
                d_id = f"auto_dialogue_all_{dataset_id}"
            elif any_dialogue_id and not d_id:
                d_id = "unknown_dialogue"

            r["dialogue_id"] = d_id
            rows.append(r)

        dialogues_map = {}
        for r in rows:
            dialogues_map.setdefault(r["dialogue_id"], []).append(r)

        dialogue_details = []
        dialogue_turns_map = {}

        human_turns = 0
        machine_turns = 0

        y_true = []
        y_pred = []
        has_any_gt = False
        has_all_gt = True

        for d_id, turns in dialogues_map.items():
            turns_sorted = sorted(
                turns,
                key=lambda x: (x.get("turn_index") is None, x.get("turn_index") if x.get("turn_index") is not None else x.get("row_order", 0), x.get("row_order", 0))
            )
            for i, t in enumerate(turns_sorted):
                if t.get("turn_index") is None:
                    t["turn_index"] = i

            turns_sorted.sort(key=lambda x: (x.get("turn_index", 0), x.get("row_order", 0)))

            d_h = 0
            d_m = 0
            per_turn = []

            for i, t in enumerate(turns_sorted):
                previous_text = _safe_str(t.get("previous_text")) or _build_prev_text_for_dialogue(turns_sorted, i)

                pred, p_machine, confidence, uncertainty = _predict_with_proba(
                    text=t["text"],
                    previous_text=previous_text,
                    threshold=threshold,
                    model_key=model_key
                )

                if pred == 0:
                    human_turns += 1
                    d_h += 1
                else:
                    machine_turns += 1
                    d_m += 1

                gt = t.get("gt")
                if gt is None:
                    has_all_gt = False
                else:
                    has_any_gt = True
                    y_true.append(int(gt))
                    y_pred.append(int(pred))

                per_turn.append({
                    "row_id": t["row_id"],
                    "dialogue_id": d_id,
                    "turn_index": int(t.get("turn_index", 0) or 0),
                    "sender": t.get("sender", ""),
                    "text": t.get("text", ""),
                    "text_preview": (t["text"][:180] + ("..." if len(t["text"]) > 180 else "")),
                    "previous_text": previous_text,
                    "prediction": "Machine-generated" if pred == 1 else "Human",
                    "prediction_int": int(pred),
                    "p_machine": p_machine,
                    "confidence": confidence,
                    "uncertainty": uncertainty,
                    "ground_truth": gt,
                })

            dialogue_turns_map[d_id] = per_turn

            total_d = max(1, (d_h + d_m))
            d_ai_pct = round((d_m / total_d) * 100.0, 2)
            d_h_pct = round((d_h / total_d) * 100.0, 2)
            final_label = _final_label_rule(d_ai_pct)

            dialogue_details.append({
                "dialogue_id": d_id,
                "turns": total_d,
                "human_turns": d_h,
                "ai_turns": d_m,
                "human_percentage": d_h_pct,
                "ai_percentage": d_ai_pct,
                "final_label": final_label,
            })

        total_turns = human_turns + machine_turns
        if total_turns == 0:
            return jsonify({"error": "No turns analyzed"}), 500

        summary = {
            "project_id": project_id,
            "dataset_id": dataset_id,
            "model_key": model_key,
            "run_id": run_id,
            "analyzed_at": analyzed_at,
            "analyzed_by": uid,
            "threshold": threshold,
            "total_dialogues": len(dialogues_map),
            "total_turns": total_turns,
            "human_turns": human_turns,
            "machine_turns": machine_turns,
            "human_percentage": round((human_turns / total_turns) * 100.0, 2),
            "machine_percentage": round((machine_turns / total_turns) * 100.0, 2),
            "has_any_ground_truth": has_any_gt,
            "has_all_ground_truth": has_all_gt,
            "show_confusion_matrix": bool(has_any_gt),
            "show_classification_report": bool(has_any_gt),
            "note": ("No dialogue_id found in upload. Grouped all rows into one dialogue."
                     if not any_dialogue_id else "")
        }

        standard_metrics = _compute_standard_metrics(y_true, y_pred)
        metrics_block = {
            "metrics": standard_metrics,
            "confusion_matrix": standard_metrics.get("confusion_matrix") if standard_metrics.get("available") else None,
            "macro_metrics": standard_metrics if standard_metrics.get("available") else None
        }
        if has_any_gt and len(y_true) == len(y_pred) and len(y_true) > 0:
            try:
                from sklearn.metrics import (
                    classification_report
                )

                report_dict = classification_report(
                    y_true, y_pred, labels=[0, 1],
                    target_names=["Human", "Machine-generated"],
                    output_dict=True,
                    zero_division=0
                )

                metrics_block["classification_report"] = report_dict
            except Exception as e:
                app.logger.warning("Metrics skipped: %s", e)
                metrics_block["classification_report"] = None
                metrics_block["metrics_error"] = "Could not compute sklearn classification report on server."

        analysis_doc = {**summary, **metrics_block}

        try:
            db.collection("project_analysis_conversations").document(project_id).collection("runs").document(run_id).set(analysis_doc)
            if not task_finalized:
                db.collection("project_analysis_conversations").document(project_id).set({
                    "latest_run_id": run_id,
                    "latest_model_key": model_key,
                    "latest_analyzed_at": analyzed_at
                }, merge=True)
        except Exception as e:
            app.logger.warning("Firestore save skipped/failed: %s", e)

        base_ref = rtdb.reference(f"analysis_results/conversations/{model_key}/{project_id}")
        if not task_finalized:
            base_ref.child("latest_run_id").set(run_id)

        run_ref = base_ref.child("runs").child(run_id)

        run_ref.child("summary").set(analysis_doc)
        run_ref.child("dialogues").set(dialogue_details)

        turns_ref = run_ref.child("dialogue_turns")
        dialogue_key_map = {}

        for d_id, per_turn in dialogue_turns_map.items():
            safe_key = _rtdb_safe_key(d_id)
            dialogue_key_map[d_id] = safe_key
            turns_ref.child(safe_key).set(per_turn)

        run_ref.child("dialogue_key_map").set(dialogue_key_map)

        return jsonify({
            "message": "Conversation analysis complete",
            "summary": analysis_doc,
            "total_dialogues": summary["total_dialogues"],
            "total_turns": summary["total_turns"],
            "model_key": model_key,
            "run_id": run_id,
            "preview_only": task_finalized
        }), 200

    except Exception as e:
        app.logger.exception("Conversation batch analysis failed: %s", e)
        return jsonify({"error": "Analysis failed"}), 500


@app.route("/api/project/<project_id>/conversation_dialogue/<dialogue_id>", methods=["GET"])
def get_one_dialogue_details(project_id, dialogue_id):
    ctx, err = _ensure_project_access(project_id)
    if err:
        return err

    model_key = _normalize_conversation_model_key(request.args.get("model_key"), for_results=True)
    run_id = _safe_str(request.args.get("run_id"))

    base_ref = rtdb.reference(f"analysis_results/conversations/{model_key}/{project_id}")

    if not run_id:
        run_id = _safe_str(base_ref.child("latest_run_id").get())

    if not run_id:
        return jsonify({"error": "No run found (analyze first)"}), 404

    run_ref = base_ref.child("runs").child(run_id)

    summary = run_ref.child("summary").get()
    if not summary:
        return jsonify({"error": "No results for this run"}), 404

    key_map = run_ref.child("dialogue_key_map").get() or {}
    safe_key = key_map.get(dialogue_id) or _rtdb_safe_key(dialogue_id)

    turns = run_ref.child("dialogue_turns").child(safe_key).get() or []

    h = sum(1 for t in turns if int(t.get("prediction_int", 0)) == 0)
    a = sum(1 for t in turns if int(t.get("prediction_int", 0)) == 1)
    total = max(1, h + a)
    ai_pct = round((a / total) * 100.0, 2)

    header = {
        "dialogue_id": dialogue_id,
        "total_turns": h + a,
        "human_turns": h,
        "ai_turns": a,
        "human_percentage": round((h / total) * 100.0, 2),
        "ai_percentage": ai_pct,
        "final_label": _final_label_rule(ai_pct),
        "model_key": model_key,
        "run_id": run_id,
    }

    return jsonify({
        "header": header,
        "turns": turns
    }), 200


@app.route("/api/project/<project_id>/conversation_analysis_results", methods=["GET"])
def get_conversation_analysis_results(project_id):
    ctx, err = _ensure_project_access(project_id)
    if err:
        return err

    model_key = _normalize_conversation_model_key(request.args.get("model_key"), for_results=True)
    run_id = _safe_str(request.args.get("run_id"))

    base_ref = rtdb.reference(f"analysis_results/conversations/{model_key}/{project_id}")

    if not run_id:
        run_id = _safe_str(base_ref.child("latest_run_id").get())

    if not run_id:
        return jsonify({"error": "No conversation analysis results found"}), 404

    run_ref = base_ref.child("runs").child(run_id)

    summary = run_ref.child("summary").get()
    dialogues = run_ref.child("dialogues").get()

    if not summary or dialogues is None:
        return jsonify({"error": "No conversation analysis results found"}), 404

    return jsonify({
        "summary": summary,
        "dialogues": dialogues
    }), 200


@app.route("/api/project/<project_id>/conversation_export", methods=["GET"])
def export_conversation_enriched_dataset(project_id):
    ctx, err = _ensure_project_access(project_id)
    if err:
        return err

    model_key = _normalize_conversation_model_key(request.args.get("model_key"), for_results=True)
    run_id = _safe_str(request.args.get("run_id"))

    base_ref = rtdb.reference(f"analysis_results/conversations/{model_key}/{project_id}")
    if not run_id:
        run_id = _safe_str(base_ref.child("latest_run_id").get())

    if not run_id:
        return jsonify({"error": "No run found (analyze first)"}), 404

    run_ref = base_ref.child("runs").child(run_id)

    dialogue_turns = run_ref.child("dialogue_turns").get() or {}
    key_map = run_ref.child("dialogue_key_map").get() or {}

    rev_map = {v: k for k, v in (key_map.items() if isinstance(key_map, dict) else [])}

    flat = []
    for safe_key, turns in (dialogue_turns.items() if isinstance(dialogue_turns, dict) else []):
        turns = _turns_to_list(turns)
        if not turns:
            continue

        original_id = rev_map.get(safe_key, safe_key)

        for t in turns:
            flat.append({
                "dialogue_id": original_id,
                "turn_index": t.get("turn_index", 0),
                "sender": t.get("sender", ""),
                "text": t.get("text", ""),
                "previous_text": t.get("previous_text", ""),
                "prediction": t.get("prediction", ""),
                "prediction_int": t.get("prediction_int", None),
                "p_machine": t.get("p_machine", None),
                "confidence": t.get("confidence", None),
                "uncertainty": t.get("uncertainty", None),
                "ground_truth": t.get("ground_truth", None),
                "source_row_id": t.get("row_id", ""),
            })

    flat.sort(key=lambda x: (str(x.get("dialogue_id", "")), int(x.get("turn_index", 0) or 0)))

    return jsonify({
        "project_id": project_id,
        "model_key": model_key,
        "run_id": run_id,
        "exported_at": _now_utc_iso(),
        "rows": flat,
        "total_rows": len(flat)
    }), 200


if __name__ == "__main__":
 app.run(debug=True)
