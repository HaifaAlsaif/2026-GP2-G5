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
    return normalized in ("logistic", "logreg", CONV_LOGREG_KEY)


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

    data = request.form

    name = data.get("project_name", "").strip()
    desc = data.get("description", "").strip()
    category = data.get("category")
    domains = data.getlist("domain")

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
            title = (payload.get("title") or 
                    payload.get("Title") or 
                    payload.get("headline") or "")
            
            content = (payload.get("Article") or 
                      payload.get("article") or 
                      payload.get("content") or 
                      payload.get("text") or "")
            
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

        # نحلل كل مقالة
        for push_id, article_data in snapshot.items():
            if not isinstance(article_data, dict):
                continue

            payload = article_data.get("payload", {})
            
            title = (payload.get("title") or 
                    payload.get("Title") or 
                    payload.get("headline") or "")
            
            content = (payload.get("Article") or 
                      payload.get("article") or 
                      payload.get("content") or "")
            
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
                human_scores.append(float(probabilities[0]))
                ai_scores.append(float(probabilities[1]))


            
            # المتوسط
            final_human = sum(human_scores) / len(human_scores)
            final_ai = sum(ai_scores) / len(ai_scores)
          
            #4 النتيجة النهائية
            prediction = "AI" if final_ai > final_human else "Human"
            confidence, uncertainty = _confidence_uncertainty_from_prob(final_ai)
            
            if prediction == "Human":
                human_count += 1
            else:
                ai_count += 1

            results.append({
                "confidence": _percent_or_none(confidence),
                "uncertainty": _percent_or_none(uncertainty),
    "article_id": push_id,
    "title": title[:100] if title else "",
    "content": full_text[:500] if full_text else "",
    "prediction": prediction,
    "human_percentage": round(final_human * 100, 2),
    "ai_percentage": round(final_ai * 100, 2),
    "chunks": [
        {
            "label": f"F{i+1}",
            "human": round(float(human_scores[i]) * 100, 2),
            "ai": round(float(ai_scores[i]) * 100, 2)
        }
        for i in range(len(chunks))
    ]
})

        # نحفظ النتائج في Firestore
        analysis_doc = {
            "project_id": project_id,
            "dataset_id": dataset_id,
            "model_type": selected_model,
            "total_articles": len(results),
            "human_count": human_count,
            "ai_count": ai_count,
            "human_percentage": round((human_count / len(results)) * 100, 2),
            "ai_percentage": round((ai_count / len(results)) * 100, 2),
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
    if uid not in task_data.get("examiner_ids", []):
        abort(403)
    
    # نجيب اسم المستخدم
    user_doc = db.collection("users").document(uid).get()
    first_name = user_doc.to_dict().get("profile", {}).get("firstName", "")
    last_name = user_doc.to_dict().get("profile", {}).get("lastName", "")
    user_name = f"{first_name} {last_name}".strip() or "User"
    
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
    
    # نجيب Task
    task_doc = db.collection("tasks").document(task_id).get()
    if not task_doc.exists:
        return jsonify({"error": "Task not found"}), 404
    
    task_data = task_doc.to_dict()
    
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
        
        # نحلل كل مقالة
        for push_id, article_data in snapshot.items():
            if not isinstance(article_data, dict):
                continue
            
            payload = article_data.get("payload", {})
            
            title = (payload.get("title") or 
                    payload.get("Title") or "")
            
            content = (payload.get("Article") or 
                      payload.get("article") or "")
            
            full_text = f"{title}. {content}" if title else content
            
            chunks = split_into_3_chunks(full_text)
            
            human_scores = []
            ai_scores = []
            chunk_details = []
            
            for i, chunk in enumerate(chunks):
                if model_type == "rnn":
                    probabilities = rnn_predict_proba([chunk])[0]
                else:
                    probabilities = news_pipeline.predict_proba([chunk])[0]
                
                h_score = float(probabilities[0])
                a_score = float(probabilities[1])
                
                human_scores.append(h_score)
                ai_scores.append(a_score)
                
                chunk_details.append({
                    "label": f"F{i+1}",
                    "human": round(h_score * 100, 2),
                    "ai": round(a_score * 100, 2)
                })
            
            final_human = sum(human_scores) / len(human_scores)
            final_ai = sum(ai_scores) / len(ai_scores)
            
            prediction = "AI" if final_ai > final_human else "Human"
            confidence, uncertainty = _confidence_uncertainty_from_prob(final_ai)
            
            if prediction == "Human":
                human_count += 1
            else:
                ai_count += 1
            
            results.append({
                "confidence": _percent_or_none(confidence),
                "uncertainty": _percent_or_none(uncertainty),
                "article_id": push_id,
                "title": title[:100] if title else "Untitled",
                "content": content[:500] if content else "",  # ✅ أول 500 حرف
                "prediction": prediction,
                "human_percentage": round(final_human * 100, 2),
                "ai_percentage": round(final_ai * 100, 2),
                "chunks": chunk_details  # ✅ تفاصيل الـ Chunks
            })
        
        # ملخص النتائج
        summary = {
            "model_type": model_type,
            "total_articles": len(results),
            "human_count": human_count,
            "ai_count": ai_count,
            "human_percentage": round((human_count / len(results)) * 100, 2) if results else 0,
            "ai_percentage": round((ai_count / len(results)) * 100, 2) if results else 0
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

    task_ref = db.collection("tasks").document(task_id)
    task_doc = task_ref.get()
    if not task_doc.exists:
        return jsonify({"error": "Task not found"}), 404

    task_data = task_doc.to_dict()

    if uid not in task_data.get("examiner_ids", []):
        return jsonify({"error": "Forbidden"}), 403

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

            for push_id, article_data in snapshot.items():
                if not isinstance(article_data, dict):
                    continue

                payload = article_data.get("payload", {})
                title = payload.get("title") or payload.get("Title") or ""
                content = payload.get("Article") or payload.get("article") or ""
                full_text = f"{title}. {content}" if title else content

                chunks = split_into_3_chunks(full_text)
                human_scores = []
                ai_scores = []
                chunk_details = []

                for i, chunk in enumerate(chunks):
                    if selected_model == "rnn":
                        probabilities = rnn_predict_proba([chunk])[0]
                    else:
                        probabilities = news_pipeline.predict_proba([chunk])[0]

                    h = float(probabilities[0])
                    a = float(probabilities[1])
                    human_scores.append(h)
                    ai_scores.append(a)
                    chunk_details.append({
                        "label": f"F{i+1}",
                        "human": round(h * 100, 2),
                        "ai": round(a * 100, 2)
                    })

                final_human = sum(human_scores) / len(human_scores)
                final_ai = sum(ai_scores) / len(ai_scores)
                prediction = "AI" if final_ai > final_human else "Human"
                confidence, uncertainty = _confidence_uncertainty_from_prob(final_ai)

                if prediction == "Human":
                    human_count += 1
                else:
                    ai_count += 1

                results.append({
                    "confidence": _percent_or_none(confidence),
                    "uncertainty": _percent_or_none(uncertainty),
                    "article_id": push_id,
                    "title": title[:100],
                    "content": content[:500],
                    "prediction": prediction,
                    "human_percentage": round(final_human * 100, 2),
                    "ai_percentage": round(final_ai * 100, 2),
                    "chunks": chunk_details
                })

            # نحفظ في RTDB - نظف project_id من الأحرف الممنوعة
            safe_pid = project_id.replace(".", "_").replace("#", "_").replace("$", "_").replace("[", "_").replace("]", "_")
            results_ref = rtdb.reference(f"analysis_results/{safe_pid}/{selected_model}")
            results_ref.set({
                "summary": {
                    "model_type": selected_model,
                    "total_articles": len(results),
                    "human_count": human_count,
                    "ai_count": ai_count,
                },
                "details": results
            })
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
        first_name = user_doc.to_dict().get('profile', {}).get('firstName', '')
        last_name = user_doc.to_dict().get('profile', {}).get('lastName', '')
        examiner_name = f"{first_name} {last_name}".strip() or "Unknown"
        
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
    user_doc = db.collection("users").document(uid).get()
    first_name = user_doc.to_dict().get("profile", {}).get("firstName", "")
    last_name = user_doc.to_dict().get("profile", {}).get("lastName", "")
    user_name = f"{first_name} {last_name}".strip() or "User"
    
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
        articles_with_feedback = []
        for article in details:
            article_id = article.get("article_id")
            feedback = None

            try:
                feedback_ref = rtdb.reference(f"datasets/uploaded_news/{dataset_id}/{article_id}/feedback")
                feedback = feedback_ref.get()
            except Exception:
                pass

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

        active_learning_enabled = _is_logistic_model_key(selected_model)
        active_learning_total = len(articles_with_feedback)
        active_learning_limit = _active_learning_limit(active_learning_total) if active_learning_enabled else active_learning_total

        if active_learning_enabled:
            selected_articles = sorted(
                articles_with_feedback,
                key=lambda item: (_active_learning_sort_value(item), item.get("article_id", ""))
            )[:active_learning_limit]
            selected_ids = {item.get("article_id") for item in selected_articles}
            for item in articles_with_feedback:
                item["active_learning_selected"] = item.get("article_id") in selected_ids
            articles_with_feedback = [item for item in articles_with_feedback if item.get("active_learning_selected")]
            articles_with_feedback.sort(key=lambda item: (_active_learning_sort_value(item), item.get("article_id", "")))

        return jsonify({
            "articles": articles_with_feedback,
            "summary": summary,
            "selected_model": selected_model,
            "active_learning": {
                "enabled": active_learning_enabled,
                "percent": ACTIVE_LEARNING_PERCENT,
                "max_samples": ACTIVE_LEARNING_MAX_SAMPLES,
                "selected": len(articles_with_feedback),
                "source_total": active_learning_total
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

    if not dataset_id:
        return jsonify({"error": "Dataset ID is required"}), 400

    if not agreed_with_model:
        label = data.get("label")
        explanation = data.get("explanation", "").strip()
        if not label or label not in ["Human", "AI"]:
            return jsonify({"error": "Invalid label"}), 400
        if not explanation:
            return jsonify({"error": "Explanation is required"}), 400

    try:
        feedback_ref = rtdb.reference(
            f"datasets/uploaded_news/{dataset_id}/{article_id}/feedback"
        )
        if feedback_ref.get():
            return jsonify({"error": "Feedback already exists for this article"}), 400

        user_doc = db.collection("users").document(uid).get()
        first_name = user_doc.to_dict().get("profile", {}).get("firstName", "")
        last_name  = user_doc.to_dict().get("profile", {}).get("lastName", "")
        examiner_name = f"{first_name} {last_name}".strip() or "Examiner"

        if agreed_with_model:
            feedback_data = {
                "examiner_uid":      uid,
                "examiner_name":     examiner_name,
                "agreed_with_model": True,
                "submitted_at":      datetime.utcnow().isoformat() + "Z"
            }
        else:
            feedback_data = {
                "examiner_uid":      uid,
                "examiner_name":     examiner_name,
                "agreed_with_model": False,
                "label":             data.get("label"),
                "explanation":       data.get("explanation", "").strip(),
                "submitted_at":      datetime.utcnow().isoformat() + "Z"
            }

        feedback_ref.set(feedback_data)
        return jsonify({"message": "Feedback submitted successfully"}), 200

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
        selected_model = (request.args.get("model") or "logreg").lower()
        model_key = CONV_RNN_KEY if selected_model == CONV_RNN_KEY else CONV_LOGREG_KEY
        selected_model_name = "RNN" if selected_model == "rnn" else "Logistic Regression"
        task_id_from_query = (request.args.get("task_id") or request.args.get("taskId") or "").strip()

        if not session.get("idToken"):
            return jsonify({"error": "Unauthorized"}), 401
        uid = session.get("uid")



        proj_doc = db.collection("projects").document(project_id).get()
        if not proj_doc.exists:
            return jsonify({"error": "Project not found"}), 404

        tasks = db.collection("tasks").where("project_ID", "==", project_id).stream()
        out_ref = rtdb.reference(f"{ANALYSIS_ROOT}/{model_key}/{project_id}")
        out_ref.delete()

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
                    "gt": _gt_label_from_sender(m.get("sender_type"), conv_type),
                    "sender": _sender_label(m.get("sender_type"), conv_type),
                    "p_machine": p_machine,
                    "confidence": conf,
                    "uncertainty": uncertainty,
                })

        if analyzed_conversations == 0:
            return jsonify({
                "error": "No conversation messages found for this project. Complete a conversation task first."
            }), 400
        if task_id_from_query:
            ms_task_ref = db.collection("tasks").document(task_id_from_query)
            ms_task_doc = ms_task_ref.get()
            if ms_task_doc.exists:
                ms_data = ms_task_doc.to_dict() or {}
                if ms_data.get("project_ID") == project_id and ms_data.get("task_type") == "model_selection":
                    ms_task_ref.update({
                        "selected_model": selected_model,
                        "selected_model_name": selected_model_name,
                        "selected_by": uid,
                        "selected_at": datetime.utcnow().isoformat() + "Z",
                        "status": "completed"
                    })

        return jsonify({
            "success": True,
            "model": selected_model,
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
    selected_model = (request.args.get("model") or "logreg").lower()
    model_key = CONV_RNN_KEY if selected_model == CONV_RNN_KEY else CONV_LOGREG_KEY

    if not session.get("idToken"):
        return jsonify({"error": "Unauthorized"}), 401

    raw = rtdb.reference(f"{ANALYSIS_ROOT}/{model_key}/{project_id}").get() or {}
    results = []

    y_true = []
    y_pred = []

    for task_id, node in raw.items():
        meta = node.get("meta", {})
        turns_raw = node.get("turns", {}) or {}
        turns = list(turns_raw.values()) if isinstance(turns_raw, dict) else turns_raw
        turns.sort(key=lambda x: x.get("turn_index", 0))

        for t in turns:
            gt = str(t.get("gt", "")).strip().lower()
            pr = str(t.get("prediction", "")).strip().lower()
            if gt in ("ai", "human") and pr in ("ai", "human"):
                y_true.append(1 if gt == "ai" else 0)
                y_pred.append(1 if pr == "ai" else 0)

        results.append({
            "task_id": task_id,
            "task_name": meta.get("task_name", "Conversation"),
            "selected_model_name": meta.get("selected_model_name"),
            "turns": turns
        })

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    total = len(y_true)
    accuracy = ((tp + tn) / total) if total else 0.0

    def prf_for_class(c):
        tp_c = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp_c = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn_c = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)

        prec = (tp_c / (tp_c + fp_c)) if (tp_c + fp_c) else 0.0
        rec = (tp_c / (tp_c + fn_c)) if (tp_c + fn_c) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        return prec, rec, f1

    prec_h, rec_h, f1_h = prf_for_class(0)
    prec_ai, rec_ai, f1_ai = prf_for_class(1)

    metrics = {
        "accuracy": accuracy,
        "precision_macro": (prec_h + prec_ai) / 2.0,
        "recall_macro": (rec_h + rec_ai) / 2.0,
        "f1_macro": (f1_h + f1_ai) / 2.0,
    }

    confusion_matrix = {
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }

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
    selected_model = (data.get("model") or "").strip().lower()
    if selected_model == "tfidf_logreg":
        selected_model = "logreg"

    if selected_model not in ("logreg", "rnn"):
        return jsonify({"error": "Invalid model"}), 400
    if not project_id or not task_id:
        return jsonify({"error": "project_id and task_id are required"}), 400

    task_ref = db.collection("tasks").document(task_id)
    task_doc = task_ref.get()
    if not task_doc.exists:
        return jsonify({"error": "Task not found"}), 404

    task_data = task_doc.to_dict() or {}
    if task_data.get("project_ID") != project_id:
        return jsonify({"error": "Task does not belong to this project"}), 400
    if task_data.get("task_type") != "model_selection":
        return jsonify({"error": "Task is not model_selection"}), 400
    if uid not in (task_data.get("examiner_ids") or []):
        return jsonify({"error": "Forbidden"}), 403

    now_iso = datetime.utcnow().isoformat() + "Z"
    model_name = "RNN" if selected_model == "rnn" else "Logistic Regression"

    task_ref.update({
        "selected_model": selected_model,  # ✅ نحفظ المفتاح نفسه: logreg / rnn
        "selected_model_name": model_name,
        "selected_by": uid,
        "selected_at": now_iso,
        "status": "completed"
    })

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

    model_key = (task_data.get("selected_model") or "").strip().lower()
    model_name = (task_data.get("selected_model_name") or "").strip()

    if model_key in ("rnn", "logreg") or model_name:
        if not model_name:
            model_name = "RNN" if model_key == "rnn" else "Logistic Regression"

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
    root_ref = rtdb.reference(f"{ANALYSIS_ROOT}/{model_key}/{project_id}")
    raw = root_ref.get() or {}
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

    active_learning_enabled = _is_logistic_model_key(model_key)
    items, active_learning_info = _apply_active_learning_turn_selection(items, active_learning_enabled)

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

        project_id = task_data.get("project_ID")
        selected_model, model_key, _ = _pick_conversation_model_for_project(project_id)
        if not selected_model:
            return jsonify({"error": "Model selection is not completed yet"}), 400

        conv_ref = rtdb.reference(f"{ANALYSIS_ROOT}/{model_key}/{project_id}/{conversation_id}")
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
            if not explanation:
                return jsonify({"error": "Explanation is required"}), 400

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
        if isinstance(existing_turn_feedbacks, dict) and len(existing_turn_feedbacks) > 0:
            return jsonify({"error": "Feedback already submitted for this turn"}), 409

        udoc = db.collection("users").document(uid).get()
        if udoc.exists:
            u = udoc.to_dict() or {}
            p = u.get("profile", {})
            examiner_name = f"{p.get('firstName','')} {p.get('lastName','')}".strip() or "Examiner"
        else:
            examiner_name = "Examiner"

        payload = {
            "examiner_uid": uid,
            "examiner_name": examiner_name,
            "agreed_with_model": agree_with_model,  # ✅ جديد
            "label": label,
            "explanation": explanation,
            "submitted_at": datetime.utcnow().isoformat() + "Z"
        }

        conv_ref.child("turn_feedbacks").child(str(turn_index)).child(uid).set(payload)

        return jsonify({
            "message": "Turn feedback saved successfully",
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
        return "Examiner"

    user_data = user_doc.to_dict() or {}
    profile = user_data.get("profile", {}) or {}
    return f"{profile.get('firstName','')} {profile.get('lastName','')}".strip() or "Examiner"


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
    value = _safe_str(label).strip().lower()
    if value in ("ai", "machine", "machine-generated", "1"):
        return 1
    if value in ("human", "0"):
        return 0
    return None


def _project_logistic_task_model(project_id):
    for task in db.collection("tasks").where("project_ID", "==", project_id).stream():
        data = task.to_dict() or {}
        if data.get("task_type") == "model_selection" and _is_logistic_model_key(data.get("selected_model")):
            return data.get("selected_model")
    return None


def _export_news_active_learning_rows(project_id, proj_data):
    selected_model = _project_logistic_task_model(project_id) or "logistic"
    dataset_id = proj_data.get("dataset_id")
    if not dataset_id:
        return []

    safe_pid = project_id.replace(".", "_").replace("#", "_").replace("$", "_").replace("[", "_").replace("]", "_")
    results_data = rtdb.reference(f"analysis_results/{safe_pid}/{selected_model}").get() or {}
    details = results_data.get("details") or []
    if not isinstance(details, list):
        details = list(details.values()) if isinstance(details, dict) else []

    selected = sorted(
        details,
        key=lambda item: (_active_learning_sort_value(item), item.get("article_id", ""))
    )[:_active_learning_limit(len(details))]
    selected_ids = {item.get("article_id") for item in selected}

    dataset_rows = rtdb.reference(f"datasets/uploaded_news/{dataset_id}").get() or {}
    detail_map = {item.get("article_id"): item for item in selected if isinstance(item, dict)}
    rows = []

    for article_id in selected_ids:
        source = dataset_rows.get(article_id) or {}
        payload = source.get("payload", {}) if isinstance(source, dict) else {}
        feedback = source.get("feedback", {}) if isinstance(source, dict) else {}
        detail = detail_map.get(article_id, {})

        if not isinstance(feedback, dict) or not feedback:
            continue

        label = feedback.get("label") if not feedback.get("agreed_with_model") else detail.get("prediction")
        label_int = _label_to_int(label)
        if label_int is None:
            continue

        title = payload.get("title") or payload.get("Title") or detail.get("title", "")
        article = payload.get("Article") or payload.get("article") or payload.get("content") or detail.get("content", "")

        rows.append({
            "project_id": project_id,
            "dataset_id": dataset_id,
            "article_id": article_id,
            "title": title,
            "Article": article,
            "text": f"{title}. {article}" if title else article,
            "MachineGen": label_int,
            "label": label_int,
            "feedback_label": label,
            "agreed_with_model": bool(feedback.get("agreed_with_model", False)),
            "confidence": detail.get("confidence"),
            "uncertainty": detail.get("uncertainty"),
            "examiner_uid": feedback.get("examiner_uid"),
            "submitted_at": feedback.get("submitted_at")
        })

    return rows


def _first_turn_feedback(feedback_root, turn_index):
    if not isinstance(feedback_root, dict):
        return None
    turn_feedbacks = feedback_root.get(str(turn_index), {}) or {}
    if not isinstance(turn_feedbacks, dict) or not turn_feedbacks:
        return None
    return next(iter(turn_feedbacks.values()))


def _export_generated_conversation_active_learning_rows(project_id):
    raw = rtdb.reference(f"{ANALYSIS_ROOT}/{CONV_LOGREG_KEY}/{project_id}").get() or {}
    if not isinstance(raw, dict):
        return []

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
            feedback = _first_turn_feedback(feedback_root, turn_index)
            if not feedback:
                continue
            label_int = _label_to_int(feedback.get("label"))
            if label_int is None:
                continue

            rows.append({
                "project_id": project_id,
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "text": turn.get("text", ""),
                "prev_text": turn.get("prev_text", ""),
                "MachineGen": label_int,
                "label": label_int,
                "feedback_label": feedback.get("label"),
                "agreed_with_model": bool(feedback.get("agreed_with_model", False)),
                "confidence": turn.get("confidence"),
                "uncertainty": turn.get("uncertainty"),
                "examiner_uid": feedback.get("examiner_uid"),
                "submitted_at": feedback.get("submitted_at")
            })

    rows.sort(key=lambda item: (_active_learning_sort_value(item), item.get("conversation_id", ""), item.get("turn_index", 0)))
    return rows


def _export_uploaded_conversation_active_learning_rows(project_id):
    run_id, run_ref, _ = _uploaded_conversation_run(project_id, CONV_LOGREG_KEY)
    if not run_ref:
        return []

    dialogue_turns = run_ref.child("dialogue_turns").get() or {}
    feedback_root = run_ref.child("turn_feedbacks").get() or {}
    key_map = run_ref.child("dialogue_key_map").get() or {}
    reverse_key_map = {v: k for k, v in key_map.items()} if isinstance(key_map, dict) else {}

    rows = []
    for safe_key, turns_raw in (dialogue_turns.items() if isinstance(dialogue_turns, dict) else []):
        turns = list(turns_raw.values()) if isinstance(turns_raw, dict) else (turns_raw if isinstance(turns_raw, list) else [])
        dialogue_id = reverse_key_map.get(safe_key, safe_key)
        dialogue_feedbacks = feedback_root.get(safe_key, {}) if isinstance(feedback_root, dict) else {}

        for idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            turn_index = idx + 1
            feedback = _first_turn_feedback(dialogue_feedbacks, turn_index)
            if not feedback:
                continue
            label_int = _label_to_int(feedback.get("label"))
            if label_int is None:
                continue

            rows.append({
                "project_id": project_id,
                "run_id": run_id,
                "dialogue_id": dialogue_id,
                "turn_index": turn.get("turn_index", turn_index),
                "text": turn.get("text", ""),
                "prev_text": turn.get("previous_text", ""),
                "MachineGen": label_int,
                "label": label_int,
                "feedback_label": feedback.get("label"),
                "agreed_with_model": bool(feedback.get("agreed_with_model", False)),
                "confidence": turn.get("confidence"),
                "uncertainty": turn.get("uncertainty"),
                "examiner_uid": feedback.get("examiner_uid"),
                "submitted_at": feedback.get("submitted_at")
            })

    rows.sort(key=lambda item: (_active_learning_sort_value(item), item.get("dialogue_id", ""), item.get("turn_index", 0)))
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
        "accuracy": None,
        "precision_macro": None,
        "recall_macro": None,
        "f1_macro": None,
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
    label = _safe_str(value).strip().lower()
    return label in ("ai", "machine", "machine-generated", "1")


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
    if not y_true or len(y_true) != len(y_pred):
        return _owner_empty_metrics()

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    def prf(class_value):
        tp_c = sum(1 for t, p in zip(y_true, y_pred) if t == class_value and p == class_value)
        fp_c = sum(1 for t, p in zip(y_true, y_pred) if t != class_value and p == class_value)
        fn_c = sum(1 for t, p in zip(y_true, y_pred) if t == class_value and p != class_value)
        precision = (tp_c / (tp_c + fp_c)) if (tp_c + fp_c) else 0
        recall = (tp_c / (tp_c + fn_c)) if (tp_c + fn_c) else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
        return precision, recall, f1

    ph, rh, f1h = prf(0)
    pa, ra, f1a = prf(1)
    total = len(y_true)

    return {
        "accuracy": round((tp + tn) / total, 4) if total else None,
        "precision_macro": round((ph + pa) / 2, 4),
        "recall_macro": round((rh + ra) / 2, 4),
        "f1_macro": round((f1h + f1a) / 2, 4),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp
        }
    }


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

    selected_ids = set()
    limit = _active_learning_limit(len(details))
    sorted_details = sorted(details, key=lambda item: (_active_learning_sort_value(item), _safe_str((item or {}).get("article_id"))))
    for item in sorted_details[:limit]:
        if isinstance(item, dict) and item.get("article_id"):
            selected_ids.add(item.get("article_id"))

    reviewed_targets = 0
    corrected_labels = 0
    examiners = {}

    for item in details:
        if not isinstance(item, dict):
            continue
        article_id = _safe_str(item.get("article_id") or item.get("id"))
        prediction = item.get("prediction")
        if _owner_prediction_is_ai(prediction):
            ai_count += 1
        else:
            human_count += 1

        confidence = _owner_decimal(item.get("confidence"))
        uncertainty = _owner_decimal(item.get("uncertainty"))
        if confidence is not None:
            confidence_values.append(confidence)
        if uncertainty is not None:
            uncertainty_values.append(uncertainty)

        feedback = {}
        if isinstance(dataset_rows, dict) and article_id in dataset_rows and isinstance(dataset_rows[article_id], dict):
            feedback = dataset_rows[article_id].get("feedback") or {}
        feedbacks = _owner_iter_feedbacks(feedback)
        if article_id in selected_ids and feedbacks:
            reviewed_targets += 1
        for fb in feedbacks:
            _owner_feedback_bucket(examiners, fb)
            if fb.get("agreed_with_model") is False:
                corrected_labels += 1

        normalized.append({
            "id": article_id,
            "article_id": article_id,
            "title": item.get("title", ""),
            "content": item.get("content", ""),
            "prediction": prediction,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "review_target": article_id in selected_ids
        })

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

    return {
        "selected_model": selected_model,
        "summary": summary,
        "metrics": _owner_empty_metrics(),
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

    sorted_turns = sorted(all_turns, key=lambda item: (_active_learning_sort_value(item), item.get("id", "")))
    review_target_keys = {(item.get("safe_key"), item.get("turn_index")) for item in sorted_turns[:_active_learning_limit(len(all_turns))]}
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
            if (safe_key, int(turn_index or 0)) in review_target_keys and feedbacks:
                reviewed_targets += 1
            for fb in feedbacks:
                _owner_feedback_bucket(examiners, fb)
                if fb.get("agreed_with_model") is False:
                    corrected_labels += 1

    macro = summary_raw.get("macro_metrics") or {}
    metrics = {
        "accuracy": macro.get("accuracy"),
        "precision_macro": macro.get("precision_macro"),
        "recall_macro": macro.get("recall_macro"),
        "f1_macro": macro.get("f1_macro"),
        "confusion_matrix": summary_raw.get("confusion_matrix")
    }
    if not any(value is not None for value in metrics.values()):
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

    raw = rtdb.reference(f"{ANALYSIS_ROOT}/{model_key}/{project_id}").get()
    if not raw and model_key != CONV_LOGREG_KEY:
        model_key = CONV_LOGREG_KEY
        raw = rtdb.reference(f"{ANALYSIS_ROOT}/{model_key}/{project_id}").get()
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

            gt_int = _label_to_int(turn.get("gt"))
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
                "confidence": confidence,
                "uncertainty": uncertainty,
                "turn_index": turn_index
            })

            turn_feedbacks = {}
            if isinstance(feedback_root, dict):
                turn_feedbacks = feedback_root.get(str(turn_index), {}) or {}
            for fb in _owner_iter_feedbacks(turn_feedbacks):
                _owner_feedback_bucket(examiners, fb)

    sorted_turns = sorted(all_turns, key=lambda item: (_active_learning_sort_value(item), item.get("id", "")))
    review_target_ids = {item.get("id") for item in sorted_turns[:_active_learning_limit(len(all_turns))]}
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
            if f"{task_id}:{turn_index}" in review_target_ids and feedbacks:
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
        raw = rtdb.reference(f"{ANALYSIS_ROOT}/{mk}/{project_id}").get() or {}
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

    tasks = get_project_tasks(project_id)
    summary_data = normalized.get("summary", _owner_empty_summary())
    counts = _owner_counts_from_summary(result_type, summary_data)
    examiners = build_owner_examiners_summary(project_id, project, tasks, summary_data, result_type)

    if any(examiner.get("labeling_assigned", 0) > 0 for examiner in examiners):
        warnings.append("Feedback targets are shared across assigned labeling examiners; pending count is an estimate.")

    project_public = {
        "project_id": project.get("project_id", ""),
        "project_name": project.get("project_name", ""),
        "category": project.get("category", ""),
        "dataset_id": project.get("dataset_id", ""),
        "generated_from_scratch": bool(project.get("generated_from_scratch", False)),
        "status": project.get("status", "")
    }

    return {
        "ok": True,
        "project": project_public,
        "result_type": result_type,
        "selected_model": normalized.get("selected_model"),
        "summary": summary_data,
        "counts": counts,
        "metrics": normalized.get("metrics", _owner_empty_metrics()),
        "examiners": examiners,
        "tasks": tasks,
        "most_uncertain_samples": normalized.get("most_uncertain_samples", []),
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


# =========================
# Owner Final Dataset Export Helpers
# =========================
# Read-only export helpers for Owner Results. They combine source rows,
# detection output, and optional examiner feedback without writing to Firebase.

NEWS_EXPORT_COLUMNS = [
    "project_id", "dataset_id", "article_id", "title", "text", "prediction",
    "prediction_int", "human_probability", "ai_probability", "confidence",
    "uncertainty", "selected_model", "model_version",
    "active_learning_selected", "original_ground_truth", "source_type",
    "examiner_uid", "examiner_name", "agreed_with_model", "corrected_label",
    "corrected_MachineGen", "feedback_explanation", "submitted_at",
    "used_for_retraining"
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
    "prediction_int", "p_machine", "confidence", "uncertainty",
    "selected_model", "model_version", "active_learning_selected",
    "examiner_uid", "examiner_name", "agreed_with_model", "corrected_label",
    "corrected_MachineGen", "feedback_explanation", "submitted_at",
    "used_for_retraining"
]


def _label_to_machinegen(label):
    value = _safe_str(label).strip().lower()
    if value in ("ai", "ai-generated", "ai generated", "machine", "machine-generated", "machine generated", "1", "true"):
        return 1
    if value in ("human", "human-written", "human written", "0", "false"):
        return 0
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
        writer.writerow({key: _flatten_json_for_csv(row.get(key)) for key in columns})

    return Response(
        "\ufeff" + output.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


def _owner_export_filename(project, stage, fmt):
    name = _safe_str(project.get("project_name") or project.get("project_id") or "project").lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_") or "project"
    return f"trustlens_{name}_{stage}.{fmt}"


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
        "version": selected.get("version") or "v1",
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
            "feedback_explanation": None,
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
        "feedback_explanation": feedback.get("explanation") or feedback.get("feedback_explanation"),
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
    title = payload.get("title") or payload.get("Title") or payload.get("headline") or detail.get("title") or ""
    text = (
        payload.get("Article") or payload.get("article") or payload.get("content")
        or payload.get("text") or payload.get("Text") or detail.get("content") or ""
    )
    return title, text


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

    active_ids = set()
    if _is_logistic_model_key(model_key):
        selected_items = sorted(
            details,
            key=lambda item: (_active_learning_sort_value(item), _safe_str((item or {}).get("article_id") or (item or {}).get("id")))
        )[:_active_learning_limit(len(details))]
        active_ids = {_safe_str(item.get("article_id") or item.get("id")) for item in selected_items if isinstance(item, dict)}

    article_ids = list(dataset_rows.keys()) if isinstance(dataset_rows, dict) else []
    for article_id in detail_map.keys():
        if article_id not in article_ids:
            article_ids.append(article_id)

    rows = []
    for article_id in article_ids:
        source = dataset_rows.get(article_id, {}) if isinstance(dataset_rows, dict) else {}
        payload = _news_source_payload(source)
        detail = detail_map.get(article_id, {})
        title, text = _text_from_news_payload(payload, detail)
        prediction = detail.get("prediction")
        human_probability = _owner_decimal(_first_present(detail.get("human_probability"), detail.get("human_percentage")))
        ai_probability = _owner_decimal(_first_present(detail.get("ai_probability"), detail.get("ai_percentage")))
        if human_probability is None and ai_probability is not None:
            human_probability = round(1 - ai_probability, 6)
        if ai_probability is None and human_probability is not None:
            ai_probability = round(1 - human_probability, 6)

        base_row = {
            "project_id": project_id,
            "dataset_id": dataset_id,
            "article_id": article_id,
            "title": title,
            "text": text,
            "prediction": prediction,
            "prediction_int": _label_to_machinegen(prediction),
            "human_probability": human_probability,
            "ai_probability": ai_probability,
            "confidence": _owner_decimal(detail.get("confidence")),
            "uncertainty": _owner_decimal(detail.get("uncertainty")),
            "selected_model": selected.get("name"),
            "model_version": selected.get("version") or "v1",
            "active_learning_selected": article_id in active_ids,
            "original_ground_truth": _first_present(payload.get("MachineGen"), payload.get("label"), payload.get("target"), payload.get("ground_truth")),
            "source_type": "uploaded_news"
        }

        feedbacks = []
        if isinstance(source, dict):
            feedbacks.extend(_feedback_items(source.get("feedback") or {}))
            feedbacks.extend(_feedback_items(source.get("examiner_feedbacks") or {}))
        rows.extend(_rows_with_feedback(base_row, feedbacks, prediction, stage))

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
    if _is_logistic_model_key(model_key):
        sorted_refs = sorted(all_turn_refs, key=lambda ref: (_active_learning_sort_value(ref[2]), _safe_str(ref[2].get("row_id") or ref[2].get("id"))))
        active_keys = {
            (safe_key, int((turn or {}).get("turn_index", idx + 1) or idx + 1))
            for safe_key, idx, turn in sorted_refs[:_active_learning_limit(len(sorted_refs))]
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
            "ground_truth": _first_present(turn.get("ground_truth"), source_payload.get("ground_truth"), source_payload.get("label"), source_payload.get("target")),
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
    raw = rtdb.reference(f"{ANALYSIS_ROOT}/{model_key}/{project_id}").get()
    if not raw and model_key != CONV_LOGREG_KEY:
        model_key = CONV_LOGREG_KEY
        raw = rtdb.reference(f"{ANALYSIS_ROOT}/{model_key}/{project_id}").get()
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
    if _is_logistic_model_key(model_key):
        sorted_refs = sorted(all_turn_refs, key=lambda ref: (_active_learning_sort_value(ref[2]), f"{ref[0]}:{ref[2].get('turn_index', ref[1])}"))
        active_ids = {
            f"{task_id}:{int((turn or {}).get('turn_index', idx) or idx)}"
            for task_id, idx, turn in sorted_refs[:_active_learning_limit(len(sorted_refs))]
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

    result_type = detect_project_result_type(project)
    try:
        if result_type == "news":
            rows, columns = _export_news_dataset(project_id, project, stage)
        elif result_type == "uploaded_conversation":
            rows, columns = _export_uploaded_conversation_dataset(project_id, project, stage)
        elif result_type == "generated_conversation":
            rows, columns = _export_generated_conversation_dataset(project_id, project, stage)
        else:
            return jsonify({"error": "Unsupported project type"}), 400

        if not rows or not _has_detection_results(rows):
            return jsonify({"ok": False, "error": "No detection results found"}), 404

        if fmt == "csv":
            return _safe_csv_response(rows, _owner_export_filename(project, stage, "csv"), columns)

        payload = {
            "ok": True,
            "project_id": project_id,
            "stage": stage,
            "format": "json",
            "row_count": len(rows),
            "rows": rows
        }
        return _safe_json_export_response(payload, _owner_export_filename(project, stage, "json"))
    except Exception as e:
        app.logger.exception("Final dataset export failed: %s", e)
        return jsonify({"error": "Failed to export final dataset"}), 500


@app.route("/api/project/<project_id>/active_learning_export", methods=["GET"])
def api_project_active_learning_export(project_id):
    ctx, err = _ensure_project_access(project_id)
    if err:
        return err

    proj_data = ctx["proj_data"]
    category = (proj_data.get("category") or "").strip().lower()
    is_conversation = category in ("conversation", "conversations", "chat", "chats")

    if not is_conversation:
        rows = _export_news_active_learning_rows(project_id, proj_data)
        project_type = "news"
    elif bool(proj_data.get("generated_from_scratch", False)):
        rows = _export_generated_conversation_active_learning_rows(project_id)
        project_type = "generated_conversation"
    else:
        rows = _export_uploaded_conversation_active_learning_rows(project_id)
        project_type = "uploaded_conversation"

    return jsonify({
        "project_id": project_id,
        "project_type": project_type,
        "model_key": CONV_LOGREG_KEY if is_conversation else "logistic",
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

    active_learning_enabled = _is_logistic_model_key(model_key)
    items, active_learning_info = _apply_active_learning_turn_selection(items, active_learning_enabled)

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

        project_id = task_data.get("project_ID")
        selected_model, model_key, _ = _pick_conversation_model_for_project(project_id)
        if not selected_model:
            return jsonify({"error": "Model selection is not completed yet"}), 400

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
            if not explanation:
                return jsonify({"error": "Explanation is required"}), 400

        feedback_ref = run_ref.child("turn_feedbacks").child(safe_key).child(str(turn_index))
        existing_feedbacks = feedback_ref.get() or {}
        if isinstance(existing_feedbacks, dict) and existing_feedbacks:
            return jsonify({"error": "Feedback already submitted for this turn"}), 409

        examiner_name = _feedback_examiner_name(uid)
        payload = {
            "examiner_uid": uid,
            "examiner_name": examiner_name,
            "agreed_with_model": agree_with_model,
            "label": label,
            "explanation": explanation,
            "submitted_at": datetime.utcnow().isoformat() + "Z"
        }

        feedback_ref.child(uid).set(payload)

        return jsonify({
            "message": "Turn feedback saved successfully",
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
                        ref = rtdb.reference(f"{ANALYSIS_ROOT}/{mk}/{project_id}")
                        raw = ref.get() or {}

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
    if label is None:
        return None

    s = str(label).strip().lower()
    if s == "":
        return None

    if s in ("0", "human", "h", "real", "genuine"):
        return 0
    if s in ("1", "ai", "machine", "machine-generated", "synthetic", "bot", "llm"):
        return 1

    if "human" in s:
        return 0
    if "ai" in s or "machine" in s or "llm" in s:
        return 1

    return None


def _extract_conversation_fields(payload):
    dialogue_id = (
        payload.get("dialogue_id") or payload.get("dialogueId") or payload.get("Dialogue_ID")
        or payload.get("conversation_id") or payload.get("conversationId") or payload.get("Conversation_ID")
        or payload.get("chat_id") or payload.get("Chat_ID")
        or payload.get("thread_id") or payload.get("Thread_ID")
        or payload.get("session_id") or payload.get("Session_ID")
        or payload.get("id") or payload.get("ID")
    )
    dialogue_id = _safe_str(dialogue_id)

    turn_index = (
        payload.get("turn_index") or payload.get("turnIndex") or payload.get("Turn_Index")
        or payload.get("turn_id") or payload.get("turnId") or payload.get("Turn_ID")
        or payload.get("turn_number") or payload.get("turnNumber") or payload.get("Turn_Number")
        or payload.get("utterance_id") or payload.get("Utterance_ID")
        or payload.get("message_index") or payload.get("messageIndex")
    )
    turn_index = None if turn_index is None or str(turn_index).strip() == "" else _as_int(turn_index, default=0)

    text = (
        payload.get("text") or payload.get("Text")
        or payload.get("utterance") or payload.get("Utterance")
        or payload.get("message") or payload.get("Message")
        or payload.get("turn") or payload.get("Turn")
        or payload.get("content") or payload.get("Content")
        or payload.get("reply") or payload.get("Reply")
        or payload.get("msg") or payload.get("Msg")
    )
    text = _safe_str(text)

    sender = (
        payload.get("sender") or payload.get("Sender")
        or payload.get("role") or payload.get("Role")
        or payload.get("author") or payload.get("Author")
        or payload.get("from") or payload.get("From")
    )
    sender = _safe_str(sender)

    gt = (
        payload.get("ground_truth") or payload.get("Ground_Truth") or payload.get("groundTruth")
        or payload.get("label") or payload.get("Label")
        or payload.get("target") or payload.get("Target")
        or payload.get("is_ai") or payload.get("isAI")
        or payload.get("class") or payload.get("Class")
        or payload.get("y") or payload.get("Y") or payload.get("MachineGen")
    )
    gt_norm = _normalize_gt(gt)

    return dialogue_id, turn_index, text, gt_norm, sender


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

        for push_id, row_data in snapshot.items():
            if not isinstance(row_data, dict):
                continue

            payload = row_data.get("payload", {}) or {}
            if not isinstance(payload, dict):
                continue

            dialogue_id, turn_index, text, gt, sender = _extract_conversation_fields(payload)
            if not _safe_str(text):
                continue

            if _safe_str(dialogue_id):
                any_dialogue_id = True

            temp_rows.append({
                "id": push_id,
                "dialogue_id": dialogue_id,
                "turn_index": turn_index,
                "text": text,
                "sender": sender,
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
                key=lambda x: (x.get("turn_index") is None, x.get("turn_index") or 0, x.get("id", ""))
            )
            for i, t in enumerate(turns_sorted):
                if t.get("turn_index") is None:
                    t["turn_index"] = i

            turns_sorted.sort(key=lambda x: (x.get("turn_index", 0), x.get("id", "")))

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

    dataset_id = proj_data.get("dataset_id")
    if not dataset_id:
        return jsonify({"error": "No dataset found"}), 404

    req = request.get_json(silent=True) or {}
    model_key = _safe_str(req.get("model_key")) or "tfidf_logreg"
    if model_key in ("baseline_rnn", "conv_rnn"):
        model_key = CONV_RNN_KEY
    elif model_key not in (CONV_LOGREG_KEY, CONV_RNN_KEY):
        model_key = CONV_LOGREG_KEY

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

        for push_id, row_data in snapshot.items():
            if not isinstance(row_data, dict):
                continue

            payload = row_data.get("payload", {}) or {}
            if not isinstance(payload, dict):
                continue

            dialogue_id, turn_index, text, gt, sender = _extract_conversation_fields(payload)
            text = _safe_str(text)
            if not text:
                continue

            if _safe_str(dialogue_id):
                any_dialogue_id = True

            temp_rows.append({
                "row_id": push_id,
                "dialogue_id": dialogue_id,
                "turn_index": turn_index,
                "text": text,
                "sender": sender,
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
                key=lambda x: (x.get("turn_index") is None, x.get("turn_index") or 0, x.get("row_id", ""))
            )
            for i, t in enumerate(turns_sorted):
                if t.get("turn_index") is None:
                    t["turn_index"] = i

            turns_sorted.sort(key=lambda x: (x.get("turn_index", 0), x.get("row_id", "")))

            d_h = 0
            d_m = 0
            per_turn = []

            for i, t in enumerate(turns_sorted):
                previous_text = _build_prev_text_for_dialogue(turns_sorted, i)

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

        metrics_block = {}
        if has_any_gt and len(y_true) == len(y_pred) and len(y_true) > 0:
            try:
                from sklearn.metrics import (
                    confusion_matrix, classification_report,
                    f1_score, precision_score, recall_score, accuracy_score
                )

                cm_2d = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
                if not (isinstance(cm_2d, list) and len(cm_2d) == 2 and len(cm_2d[0]) == 2 and len(cm_2d[1]) == 2):
                    cm_2d = [[0, 0], [0, 0]]

                report_dict = classification_report(
                    y_true, y_pred, labels=[0, 1],
                    target_names=["Human", "Machine-generated"],
                    output_dict=True,
                    zero_division=0
                )

                macro = {
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                    "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                    "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                }

                metrics_block = {
                    "confusion_matrix": cm_2d,
                    "classification_report": report_dict,
                    "macro_metrics": macro
                }
            except Exception as e:
                app.logger.warning("Metrics skipped: %s", e)
                metrics_block = {
                    "confusion_matrix": None,
                    "classification_report": None,
                    "macro_metrics": None,
                    "metrics_error": "Could not compute sklearn metrics on server."
                }

        analysis_doc = {**summary, **metrics_block}

        try:
            db.collection("project_analysis_conversations").document(project_id).collection("runs").document(run_id).set(analysis_doc)
            db.collection("project_analysis_conversations").document(project_id).set({
                "latest_run_id": run_id,
                "latest_model_key": model_key,
                "latest_analyzed_at": analyzed_at
            }, merge=True)
        except Exception as e:
            app.logger.warning("Firestore save skipped/failed: %s", e)

        base_ref = rtdb.reference(f"analysis_results/conversations/{model_key}/{project_id}")
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
            "run_id": run_id
        }), 200

    except Exception as e:
        app.logger.exception("Conversation batch analysis failed: %s", e)
        return jsonify({"error": "Analysis failed"}), 500


@app.route("/api/project/<project_id>/conversation_dialogue/<dialogue_id>", methods=["GET"])
def get_one_dialogue_details(project_id, dialogue_id):
    ctx, err = _ensure_project_access(project_id)
    if err:
        return err

    model_key = _safe_str(request.args.get("model_key")) or "tfidf_logreg"
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

    model_key = _safe_str(request.args.get("model_key")) or "tfidf_logreg"
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

    model_key = _safe_str(request.args.get("model_key")) or "tfidf_logreg"
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
        if not isinstance(turns, list):
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
