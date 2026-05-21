# TrustLens Website Setup Guide

## Overview

TrustLens is a Flask-based web platform for detecting text authenticity. It supports news article analysis, conversation analysis, model selection, examiner feedback, Active Learning review targets, and owner result exports.

This guide explains how to set up and run the website locally after cloning the project from GitHub.

---

## Requirements

Before running the project, make sure the following tools are installed:

- Python 3.10 or newer
- Git
- Visual Studio Code or any code editor
- A terminal or command prompt

The project uses:

- Flask for the backend
- HTML, CSS, and JavaScript for the frontend
- Firebase for authentication and database services
- Hugging Face integration for LLM-related features
- Machine learning libraries listed in `requirements.txt`

---

## 1. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/HaifaAlsaif/2026-GP2-G5.git
```

Then move into the project folder:

```bash
cd 2026-GP2-G5
```

If you are using a different repository URL, replace the link above with the correct GitHub link.

---

## 2. Open the Project

If you use Visual Studio Code, run:

```bash
code .
```

Or open the folder manually from your editor.

---

## 3. Create a Virtual Environment

It is recommended to use a virtual environment so the project dependencies do not conflict with other Python projects.

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

After activation, the terminal should show the virtual environment name, usually `(venv)`.

---

## 4. Install Dependencies

Install all required Python libraries:

```bash
pip install -r requirements.txt
```

If installation fails, make sure your virtual environment is activated and your Python version is compatible.

---

## 5. Environment Variables

The project needs a `.env` file in the root folder.

Create a file named:

```text
.env
```

Add the required environment variables. For example:

```env
HF_TOKEN=your_hugging_face_token_here
```

Do not share your real token publicly, and do not upload private tokens to GitHub.

---

## 6. Firebase Service Account

The project uses Firebase services. Make sure the Firebase service account file exists in the project root.

Expected file:

```text
service-account.json
```

This file is required for Firebase Admin SDK access.

Important notes:

- Do not share this file publicly.
- Do not upload private Firebase credentials to a public repository.
- If the file is missing, request it from the project owner or Firebase administrator.

---

## 7. Run the Flask Website

From the project root, run:

```bash
python app.py
```

If the application starts successfully, Flask will show a local URL similar to:

```text
http://127.0.0.1:5000/
```

---

## 8. Open the Website

Open your browser and go to:

```text
http://127.0.0.1:5000/
```

The TrustLens website should now be running locally.

---

## 9. Main Project Structure

The project contains the following main files and folders:

```text
2026-GP2-G5/
│
├── app.py
├── requirements.txt
├── .env
├── service-account.json
│
├── templates/
│   ├── HTML pages used by Flask
│
├── static/
│   ├── CSS, JavaScript, images, and frontend assets
│
├── models/
│   ├── Local model-related files
│
├── docs/
│   ├── Project documentation
│
└── For_Report/
    ├── Report-related README files
```

---

## 10. Common Issues and Fixes

### Issue: `ModuleNotFoundError`

This usually means the dependencies were not installed.

Fix:

```bash
pip install -r requirements.txt
```

Make sure the virtual environment is activated before running the command.

---

### Issue: Missing Hugging Face Token

If the app shows an error related to `HF_TOKEN`, check the `.env` file.

Fix:

```env
HF_TOKEN=your_hugging_face_token_here
```

Then restart the Flask app.

---

### Issue: Missing Firebase Credentials

If the app shows an error related to Firebase Admin SDK or service account credentials, check that this file exists:

```text
service-account.json
```

If it is missing, request it from the project owner.

---

### Issue: Port 5000 Already in Use

If another application is already using port `5000`, stop the other application or run Flask on another port.

Example:

```bash
flask run --port 5001
```

Then open:

```text
http://127.0.0.1:5001/
```

---

### Issue: Hugging Face Model Takes Time to Load

Some model-related features may take time to load when the app starts. Wait until the terminal shows that Flask is running before opening the website.

---

## 11. Recommended Run Flow

Use this order every time you want to run the project:

```bash
cd 2026-GP2-G5
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

For macOS or Linux:

```bash
cd 2026-GP2-G5
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

---

## 12. Notes for New Developers

- Do not commit `.env`.
- Do not commit private Firebase credentials.
- Do not commit local cache folders.
- Keep the existing project structure unchanged.
- Add new templates inside `templates/`.
- Add static assets inside `static/`.
- Run the website locally before pushing changes to GitHub.

