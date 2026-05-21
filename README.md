# 🔎 TrustLens

TrustLens is a web-based platform designed to **detect text authenticity** by distinguishing between **human-written** and **AI-generated** content.  
It empowers researchers, platform owners, and examiners to collaborate on evaluating datasets, creating tasks, and reviewing results with authenticity scoring and explainability insights.

## 🌐 Overview

In today’s digital world, AI-generated content is increasingly common, raising challenges in **trust, authenticity, and reliability**.  
TrustLens provides a solution by acting as a **collaborative environment** where:

- **Project Owners** create and manage projects, upload or generate datasets, and track evaluations.  
- **Examiners** participate in structured tasks including , human–human and human–AI conversations, and detection tasks where they select the most suitable model and verify its result, in addition to providing feedback and contributing to dataset labeling.
- **Admins** initialize the system with seed datasets, AI models, and manging users.

This interaction builds a **crowdsourced, evolving dataset**, enabling better AI evaluation and more trustworthy results.

## ⚙️ Core Functionality

TrustLens supports a complete workflow including:

- **Role-Based Dashboards** for Admins, Project Owners, and Examiners  
- **Project Lifecycle Management** (create, edit, delete)  
- **Dataset Upload & Generation** (CSV + conversation generation)  
- **Task Assignment & Tracking**  Define and assign tasks (News or Conversations).
- **AI Evaluation with Explainability** (Human vs AI with confidence score)  
- **Feedback Integration** from examiners Feedback Integration → Examiners enrich datasets by submitting Label&Explainbilty 
- **System Administration** (models, datasets)

## 🏗️ System Architecture

- **Frontend:** HTML, CSS, JavaScript (Bilingual: Arabic & English)  
- **Backend:** Flask (Python)  
- **Database / Authentication:** Firebase Authentication, Firestore, Realtime DB  
- **AI Integration:** Hugging Face API (baseline & fine-tuned LLMs)

---

## 🛠️ Installation & Setup

Follow these steps to install and run the project locally:

### 1. Clone the Repository
```bash
git clone https://github.com/HaifaAlsaif/2026-GP2-G5.git
cd 2026-GP2-G5
```
### 2.Open in VS Code
code .

### 3. Create a Virtual Environment
Windows:
python -m venv venv
venv\Scripts\activate

macOS / Linux:
python3 -m venv venv
source venv/bin/activate
### 4. Install Dependencies
pip install -r requirements.txt
### 5. Run the Flask App
python app.py
### 6. Open in Browser
Default: http://127.0.0.1:5000/


---
## 👥 Project Team (G5)
#### Haifa Alsaif – 443202006
#### Amira Aljeraisy – 443200950
#### Nouf Al-Muhanna – 444201063
#### Afnan Alzakary – 444201013

## 🎓 Supervisor
Dr. Abeer Aldayel

## 📌 Project Resources  
- **GitHub Repository:**
  (https://github.com/HaifaAlsaif/2025-GP1-G5.git)  
- **Jira Board:** [[TrustLens-Jira]([https://afnanalzakary0-1764491560304.atlassian.net/jira/software/projects/GP/pages?atlOrigin=eyJpIjoiMmUxYmVkYzgzMzdmNDI4NGJiNzNkNjAzNzg4NWRhODIiLCJwIjoiaiJ9])]  
- **University:** King Saud University – IT497 Graduation Project
- Department of Information Technology
- Semester: 2, 1447H (Spring 2026)
