# TrustLens Project Documentation

TrustLens is a Flask web application for detecting whether text content is human-written or AI-generated. The project now supports three project paths:

- News/articles uploaded as CSV datasets.
- Generated conversation projects, where examiners create Human-Human or Human-AI conversations inside the system.
- Uploaded conversation dataset projects, where the owner uploads a CSV conversation dataset and examiners run conversation detection directly on the uploaded data.

The system is designed around collaboration between project owners and examiners. Owners create projects, invite examiners, create tasks, run model analysis, and review progress. Examiners accept invitations, complete assigned tasks, select models, participate in conversations, and submit feedback on model predictions.

This README is written as a handover document for a new technical team. It explains the project structure, backend routes, frontend pages, database usage, machine learning models, and the expected user flows.

---

## 1. High-Level Purpose

The platform helps evaluate content authenticity by combining:

- User-managed projects.
- Examiner participation.
- Machine learning predictions.
- Feedback from human examiners.
- Stored analysis results for later review.

The project currently separates the work into three project flows:

1. Article/news detection:
   - A project owner uploads a CSV dataset.
   - The system stores dataset rows in Firebase Realtime Database.
   - An examiner runs available models.
   - The examiner selects the best model.
   - Examiners can later provide feedback on articles.

2. Generated conversation detection:
   - A project owner creates a conversation project and chooses Generate from scratch.
   - The owner creates tasks for Human-AI or Human-Human conversations.
   - Examiners complete conversation turns.
   - A model-selection task chooses the best model.
   - A labeling task allows examiners to review turn-level predictions.

3. Uploaded conversation dataset detection:
   - A project owner creates a conversation project and uploads a CSV dataset instead of generating from scratch.
   - The system stores rows in `datasets/uploaded_conversations/{dataset_id}`.
   - The owner creates `model_selection` and later `labeling` tasks.
   - The examiner opens the uploaded conversation detection page and runs the baseline conversation model on the uploaded dataset.

---

## 2. Technology Stack

### Backend

- Python
- Flask
- Firebase Admin SDK
- Firebase Authentication through Firebase REST API
- Firestore
- Firebase Realtime Database
- TensorFlow / Keras
- scikit-learn compatible pipelines
- joblib
- pandas
- Hugging Face Hub
- ctransformers

### Frontend

- HTML templates rendered by Flask/Jinja
- Inline CSS and JavaScript inside templates
- Static JavaScript files for translation support
- Static image and SVG assets under `static/images`
- Chart.js loaded through CDN in dashboard pages

### Machine Learning

Article/news models:

- `news_baseline_pipeline.pkl`
  - Classical baseline model loaded with `joblib`.
  - Used for article/news classification.

- `news_rnn_baseline.keras`
  - Keras RNN model.
  - Uses `news_rnn_tokenizer.pkl`.
  - Maximum input length is `600`.

Conversation models:

- `models/conversation_logistic_regression.joblib`
  - Logistic Regression based conversation model.
  - Used through key `tfidf_logreg`.
  - Used by generated conversation analysis and uploaded conversation dataset detection.

- `Model-Gen-Con/rnn_v2_model.keras`
  - RNN conversation model.
  - Uses `Model-Gen-Con/rnn_v2_tokenizer.pkl`.
  - Configuration exists in `Model-Gen-Con/rnn_v2_config.json`.
  - Kept in `Model-Gen-Con` and shared by conversation detection flows when RNN is selected.

---

## 3. Project Structure

```text
2025-GP1-G5/
  app.py
  auth_rest.py
  firebase_admin_setup.py
  llm_service.py
  ml_runner.py
  conversation_baseline_model.py
  requirements.txt
  README.md
  AUTHORS
  .gitignore

  docs/
    GP1_G5_TrustLens release 1.pdf

  models/
    conversation_logistic_regression.joblib

  Model-Gen-Con/
    rnn_v2_model.keras
    rnn_v2_tokenizer.pkl
    rnn_v2_config.json

  static/
    js/
      i18n.js
      translations.js
    images/
      ... UI images and SVG icons ...

  templates/
    HomePage.html
    Login.html
    signup.html
    CheckEmail.html
    Verified.html
    ForgotPassword.html
    Profile.html
    Ownerdashboard.html
    Examinerdashboard.html
    CreateProject.html
    myprojectowner.html
    myprojectexaminer.html
    ProjectDetailsOwner.html
    ProjectDetailsExaminer.html
    CreateTask.html
    invitation.html
    ConversationH-AI.html
    ConversationH-H.html
    ConversationAnalysisResults.html
    ModelSelectionTask.html
    feedback.html
    feedbacktask.html
    results.con.html
    index.html
```

### Important note about `docs/`

The repository includes:

```text
docs/GP1_G5_TrustLens release 1.pdf
```

This appears to be the attached project release document. The current local environment does not include a PDF text extraction package or `pdftotext`, so the code-level documentation below is based on direct source-code review.

---

## 4. Main Files Explained

### `app.py`

This is the main Flask application and the central backend file. It contains:

- Flask app initialization.
- Firebase Firestore and Realtime Database usage.
- Loading article/news ML models.
- Loading generated-conversation ML models.
- Page routes.
- Authentication endpoints.
- Project APIs.
- Invitation APIs.
- Task APIs.
- Conversation APIs.
- Article analysis APIs.
- Conversation analysis APIs.
- Feedback APIs.
- Examiner progress and rating APIs.

Most business logic currently lives in this file.

### `firebase_admin_setup.py`

Initializes Firebase Admin SDK.

It expects:

```text
service-account.json
```

to exist in the project root.

It initializes:

- Firebase project ID: `trustlens-e9038`
- Realtime Database URL: `https://trustlens-e9038-default-rtdb.firebaseio.com`
- Firestore client exported as `db`

### `auth_rest.py`

Wraps Firebase Authentication REST API calls:

- `signup(email, password)`
- `signin(email, password)`
- `send_password_reset(email)`
- `update_password(id_token, new_password)`
- `refresh_id_token(refresh_token)`

It reads:

```text
FIREBASE_WEB_API_KEY
```

from `.env`.

### `llm_service.py`

Loads a local Hugging Face compatible Llama model using `ctransformers`.

It reads:

```text
HF_TOKEN
```

from `.env`.

It loads:

```text
TheBloke/Llama-2-7B-chat-GGML
```

and exposes:

```python
generate_reply(user_message: str) -> str
```

This function is used by `/api/ai_reply` to generate the AI response in Human-AI conversation tasks.

### `ml_runner.py`

A small standalone runner for the article/news RNN model.

It loads:

- `news_rnn_baseline.keras`
- `news_rnn_tokenizer.pkl`

and exposes:

```python
predict(text)
```

which returns:

```text
(human_probability, ai_probability)
```

### `conversation_baseline_model.py`

Helper module for the uploaded conversation dataset detection flow.

It loads:

```text
models/conversation_logistic_regression.joblib
```

and exposes:

```python
predict_one_turn(text, prev_text="")
```

The page `ConversationAnalysisResults.html` uses backend APIs that call this helper to classify each uploaded conversation turn as Human or Machine-generated.

### `requirements.txt`

Contains the Python dependencies needed for the Flask application, Firebase, requests, Hugging Face integration, and other utilities.

Important note:

The current file lists TensorFlow, pandas, scikit-learn, and joblib usage in the code, but not all of those packages are explicitly pinned in `requirements.txt`. A new team should verify the environment and add missing packages if installation fails.

---

## 5. Required Runtime Files and Environment Variables

### Required local files

The app expects these model and credential files at runtime:

```text
service-account.json
news_baseline_pipeline.pkl
news_rnn_baseline.keras
news_rnn_tokenizer.pkl
models/conversation_logistic_regression.joblib
Model-Gen-Con/rnn_v2_model.keras
Model-Gen-Con/rnn_v2_tokenizer.pkl
Model-Gen-Con/rnn_v2_config.json
```

### Required `.env` variables

```text
FIREBASE_WEB_API_KEY=...
HF_TOKEN=...
```

`FIREBASE_WEB_API_KEY` is needed by `auth_rest.py`.

`HF_TOKEN` is needed by `llm_service.py`.

### Flask secret key

`app.py` currently contains:

```python
app.secret_key = "CHANGE_THIS_SECRET_IN_ENV_OR_CONFIG"
```

For production, move this value to an environment variable.

---

## 6. Installation and Running Locally

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate it

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If the app fails because of missing ML packages, install the missing packages used by the source code:

```bash
pip install tensorflow pandas scikit-learn joblib
```

### 4. Add Firebase and environment files

Place `service-account.json` in the project root.

Create `.env`:

```text
FIREBASE_WEB_API_KEY=your_firebase_web_api_key
HF_TOKEN=your_huggingface_token
```

### 5. Run the app

```bash
python app.py
```

Default local URL:

```text
http://127.0.0.1:5000/
```

---

## 7. User Roles

The app mainly uses two roles:

### Owner

The owner can:

- Sign up and log in.
- Create projects.
- Upload article datasets or create generated-conversation projects.
- Invite examiners.
- View owned projects.
- Open project details.
- Add or remove examiners.
- Create tasks.
- Edit/delete tasks.
- Run analysis.
- View examiner feedback count.
- Rate examiners.

### Examiner

The examiner can:

- Sign up and log in.
- Mark profile as volunteer.
- Accept or reject invitations.
- View accepted projects.
- Open assigned project details.
- Complete conversation tasks.
- Run or select a model for model-selection tasks.
- Submit article feedback.
- Submit conversation turn feedback.
- View rating from the owner.

---

## 8. Authentication Flow

### Signup

Page:

```text
/signup
```

Template:

```text
templates/signup.html
```

Backend endpoint:

```text
POST /api/signup
```

Flow:

1. User submits signup form.
2. Backend calls Firebase Auth REST signup.
3. Backend creates a Firestore `users/{uid}` document.
4. Backend sends email verification.
5. User is sent to `CheckEmail.html`.
6. After verification, `/auto-login` signs the user in and redirects based on role.

### Login

Page:

```text
/login
```

Template:

```text
templates/Login.html
```

Backend endpoint:

```text
POST /api/signin
```

Flow:

1. User submits email/username and password.
2. Backend resolves username if needed.
3. Firebase Auth signs the user in.
4. Backend checks email verification.
5. Flask session stores:
   - `idToken`
   - `uid`
6. User is redirected:
   - Owner -> `/ownerdashboard`
   - Examiner -> `/examinerdashboard`
   - Unknown role -> `/profile`

### Logout

Endpoint:

```text
/logout
```

It clears the Flask session and redirects to `/`.

### Password reset

Pages/endpoints:

```text
/forgot
POST /forgot
POST /api/reset
```

Firebase password reset email is sent through the REST API wrapper.

---

## 9. Firebase Data Model

The project uses both Firestore and Firebase Realtime Database.

### Firestore collections

#### `users`

Stores user profile and role information.

Common fields used in the code:

```text
uid
email
username
first_name
last_name
role
is_volunteer
created_at
updated_at
specialization
linkedin
github
bio
```

The exact fields depend on signup/profile form data.

#### `projects`

Created by project owners.

Important fields:

```text
project_id
project_name
description
category
owner_id
owner_name
dataset_id
dataset_url
status
created_at
updated_at
```

`category` controls the project behavior. The code treats article/news projects differently from conversation/generated projects.

#### `invitations`

Used to invite examiners to projects.

Important fields:

```text
project_id
project_name
owner_id
owner_name
examiner_id
examiner_email
examiner_name
status
created_at
updated_at
```

Typical statuses:

```text
pending
accepted
rejected
removed
```

#### `tasks`

Created inside projects.

Common fields:

```text
task_ID
project_ID
task_name
task_description
owner_id
examiner_ids
status
created_at
updated_at
```

Article task fields:

```text
task_type
selected_model
selected_model_name
selected_at
```

Conversation task fields:

```text
conversation_type
number_of_turns
task_type
selected_model
selected_model_name
selected_at
```

Important task types:

```text
model_selection
labeling
```

Important conversation types:

```text
human-ai
human-human
```

#### `project_analysis`

Used for article/news project analysis summaries.

Document ID:

```text
project_id
```

Stores analysis summary such as:

```text
dataset_id
model_type
total
human_count
ai_count
avg_confidence
created_at
```

#### `projects/{project_id}/assigned_examiners`

Subcollection used by owner rating APIs.

Document ID:

```text
examiner_id
```

Fields include:

```text
rating
rated_at
```

### Realtime Database paths

#### Uploaded datasets

Article/news datasets:

```text
datasets/uploaded_news/{dataset_id}/{article_id}
```

Conversation datasets, if ingested:

```text
datasets/uploaded_conversations/{dataset_id}/{row_id}
```

Rows include:

```text
dataset_id
owner_id
project_id
title
text
label
source
```

The exact row fields depend on the uploaded CSV columns.

#### Human-AI messages

```text
llm_conversations/{task_id}/messages/{message_id}
```

Used by `/conversation-ai`.

Messages include:

```text
sender_id
sender_name
sender_type
message
timestamp
```

For AI replies, `sender_type` is typically AI-related.

#### Human-Human messages

```text
hh_conversations/{task_id}/messages/{message_id}
```

Used by `/conversation-hh`.

Messages include:

```text
sender_id
sender_name
sender_type
message
timestamp
```

#### Article/news analysis results

```text
analysis_results/{project_id}/{selected_model}
```

Examples:

```text
analysis_results/{project_id}/logistic
analysis_results/{project_id}/rnn
```

Stores:

```text
summary
results
```

Each result includes prediction, confidence, and explanation-style fields.

#### Generated conversation analysis results

```text
analysis_results/conversation_gen/{model_key}/{project_id}/{task_id}
```

Model keys:

```text
tfidf_logreg
rnn
```

Each analyzed conversation has:

```text
meta
turns
turn_feedbacks
```

Each turn contains:

```text
turn
text
prediction
gt
sender
confidence
```

Feedback is saved under:

```text
analysis_results/conversation_gen/{model_key}/{project_id}/{conversation_id}/turn_feedbacks/{turn_index}/{examiner_uid}
```

#### Uploaded conversation dataset analysis results

```text
analysis_results/conversations/{model_key}/{project_id}/runs/{run_id}
```

This path is separate from generated conversation results. It is used by `ConversationAnalysisResults.html` for uploaded conversation CSV projects.

The latest run ID is stored at:

```text
analysis_results/conversations/{model_key}/{project_id}/latest_run_id
```

Each run stores:

```text
summary
dialogues
dialogue_turns
dialogue_key_map
```

The uploaded conversation dataset flow reads source rows from:

```text
datasets/uploaded_conversations/{dataset_id}
```

and writes analysis output to:

```text
analysis_results/conversations/{model_key}/{project_id}/runs/{run_id}
```

---

## 10. Page-by-Page Frontend Documentation

### `templates/HomePage.html`

Route:

```text
/
```

Purpose:

- Public landing page for TrustLens.
- Explains project idea, features, target users, team, and contact section.
- Links to login and signup.
- Includes theme and language switching logic.

Main user actions:

- Go to `/login`.
- Go to `/signup`.
- Navigate informational sections.

### `templates/Login.html`

Route:

```text
/login
```

Purpose:

- Login form.
- Submits to `/api/signin`.
- Supports theme and language switching.
- Links to signup and forgot password.

Main user actions:

- Enter email/username.
- Enter password.
- Submit login.
- Navigate to `/forgot`.
- Navigate to `/signup`.

### `templates/signup.html`

Route:

```text
/signup
```

Purpose:

- Account creation form.
- Collects user profile details.
- Allows user to choose role and volunteer-related information.
- Submits to `/api/signup`.

Main user actions:

- Create an owner or examiner account.
- Add optional profile details.
- Mark volunteer availability.

### `templates/CheckEmail.html`

Rendered after signup.

Purpose:

- Tells the user to check email verification.
- Used before final login.

### `templates/Verified.html`

Route:

```text
/verified
```

Purpose:

- Email verified confirmation page.
- Redirects user to `/auto-login`.

### `templates/ForgotPassword.html`

Route:

```text
/forgot
```

Purpose:

- Password reset page.
- Posts reset request to backend.
- Links back to login/signup.

### `templates/Profile.html`

Route:

```text
/profile
```

Purpose:

- Shows and updates user profile data.
- Calls `/api/update-profile`.

Main user actions:

- Update first name, last name, username, specialization, links, and profile details.

### `templates/Ownerdashboard.html`

Route:

```text
/ownerdashboard
```

Purpose:

- Owner dashboard.
- Provides navigation to create projects and view projects.
- Includes dashboard cards and Chart.js visualization.

Important links:

```text
/createproject
/myprojectowner
/profile
/logout
```

### `templates/Examinerdashboard.html`

Route:

```text
/examinerdashboard
```

Purpose:

- Examiner dashboard.
- Provides navigation to invitations, assigned projects, feedback, and profile.
- Includes dashboard cards and Chart.js visualization.

Important links:

```text
/invitation
/myprojectexaminer
/feedback
/profile
/logout
```

### `templates/CreateProject.html`

Route:

```text
/createproject
```

Optional edit mode:

```text
/createproject?id={project_id}
```

Purpose:

- Owner creates or edits a project.
- Collects project name, description, category, dataset upload, and examiner invitations.
- If editing, loads existing project data from `/api/project_json_owner/{project_id}`.
- Loads volunteer examiner suggestions from `/api/volunteers`.
- Submits create request to `/api/create_project`.
- Submits update request to `/api/update_project/{project_id}`.

Important behavior:

- Article/news projects require a dataset upload.
- Conversation projects have two paths:
  - Upload a conversation CSV dataset.
  - Generate from scratch.
- If Conversation is generated from scratch, the project later uses Human-AI and Human-Human tasks before model selection.
- If Conversation uses an uploaded dataset, the project skips Human-AI and Human-Human tasks and goes directly to model-selection/feedback style tasks for the uploaded dataset.
- Selected examiners are passed to backend as `invited_examiners`.

### `templates/myprojectowner.html`

Route:

```text
/myprojectowner
```

Purpose:

- Owner list of owned projects.
- Loads projects from `/api/my_projects`.
- Allows view, edit, and delete.

Main actions:

- View project details:

```text
/projectdetailsowner/{project_id}
```

- Edit project:

```text
/createproject?id={project_id}&basic=1
```

- Delete project:

```text
DELETE /api/delete_project/{project_id}
```

### `templates/ProjectDetailsOwner.html`

Route:

```text
/projectdetailsowner/{project_id}
```

Purpose:

- Owner view for a single project.
- Shows project details.
- Shows examiners.
- Shows tasks.
- Allows examiner management.
- Allows creating/editing/deleting tasks.
- Shows examiner feedback count.
- Allows rating examiners.

Important APIs:

```text
GET  /api/project_json_owner/{project_id}
GET  /api/project_examiners_owner/{project_id}
GET  /api/project_tasks/{project_id}
GET  /api/volunteers
POST /api/add_examiner_to_project
POST /api/remove_examiner
POST /api/tasks/{task_id}/delete
DELETE /api/delete_project/{project_id}
GET  /api/project/{project_id}/examiner_feedback_count
POST /api/project/{project_id}/rate_examiner
```

### `templates/CreateTask.html`

Route:

```text
/projects/{project_id}/tasks/create
```

Optional edit mode:

```text
/projects/{project_id}/tasks/create?task_id={task_id}
```

Purpose:

- Owner creates or edits a task inside a project.
- Loads assigned project examiners from `/api/project_examiners_for_task/{project_id}`.
- For edit mode, loads task details from `/api/tasks/{task_id}`.

Task behavior:

- Article projects:
  - `model_selection` task requires exactly one examiner.
  - `labeling` task requires at least one examiner.

- Generated conversation projects:
  - `human-ai` requires exactly one examiner.
  - `human-human` requires exactly two examiners.
  - `model_selection` and `labeling` can be used after conversations are completed.

- Uploaded conversation dataset projects:
  - `model_selection` task requires exactly one examiner.
  - `labeling` task requires at least one examiner.
  - `human-ai` and `human-human` tasks are not used because the conversations already come from the uploaded dataset.

Submits to:

```text
POST  /api/create_task
PATCH /api/update_task/{task_id}
```

### `templates/myprojectexaminer.html`

Route:

```text
/myprojectexaminer
```

Purpose:

- Examiner list of accepted projects.
- Loads data from `/api/accepted_projects`.
- Opens project details through `/projectdetailsexaminer/{project_id}`.

### `templates/invitation.html`

Route:

```text
/invitation
```

Purpose:

- Examiner invitation inbox.
- Loads invitations from `/api/invitations`.
- Updates invitation status through `/api/invitations/{invitation_id}`.

Main statuses:

```text
accepted
rejected
pending
```

### `templates/ProjectDetailsExaminer.html`

Route:

```text
/projectdetailsexaminer/{project_id}
```

Purpose:

- Examiner view for a single accepted project.
- Loads project details.
- Loads assigned examiners.
- Loads tasks assigned to current examiner.
- Shows owner rating for current examiner.

Important APIs:

```text
GET /api/project_json/{project_id}
GET /api/project_examiners/{project_id}
GET /api/examiner_tasks/{project_id}
GET /api/project/{project_id}/my_rating
```

Main navigation:

- Model selection task:

```text
/task/{task_id}/model-selection
```

- Article feedback task:

```text
/task/{task_id}/feedback
```

- Human-AI conversation:

```text
/conversation-ai?taskId={task_id}&projectId={project_id}
```

- Human-Human conversation:

```text
/conversation-hh?taskId={task_id}&projectId={project_id}
```

- Conversation results/feedback:

```text
/results_con?projectId={project_id}&taskId={task_id}
```

### `templates/ConversationH-AI.html`

Route:

```text
/conversation-ai
```

Query parameters:

```text
taskId
projectId
```

Purpose:

- Allows one examiner to chat with the AI assistant.
- Sends user message to `/api/ai_reply`.
- Loads existing messages from `/api/ai/messages`.
- The backend writes messages under `llm_conversations/{task_id}/messages`.
- Once the required number of turns is reached, the task status becomes completed.

### `templates/ConversationH-H.html`

Route:

```text
/conversation-hh
```

Query parameters:

```text
taskId
projectId
```

Purpose:

- Allows two assigned examiners to chat with each other.
- Loads messages from `/api/hh/messages`.
- Sends messages to `/api/hh/send`.
- Backend enforces turn-taking so the same examiner cannot keep sending messages without the other examiner replying.
- The backend writes messages under `hh_conversations/{task_id}/messages`.
- Once both examiners complete the configured turns, the task becomes completed.

### `templates/ModelSelectionTask.html`

Route:

```text
/task/{task_id}/model-selection
```

Purpose:

- Used by examiner for article/news model-selection tasks.
- Runs both article models.
- Allows examiner to select the better model.

Important APIs:

```text
GET  /api/tasks/{task_id}
POST /api/task/{task_id}/run_model
POST /api/task/{task_id}/select_model
```

After model selection, the task is marked completed.

### `templates/feedback.html`

Route:

```text
/feedback
```

Purpose:

- Examiner page listing projects that are ready for feedback.
- Loads accepted projects from `/api/accepted_projects`.
- Links to project details or feedback-related pages.

### `templates/feedbacktask.html`

Route:

```text
/task/{task_id}/feedback
```

Purpose:

- Article/news feedback page.
- Loads articles assigned to the task from `/api/task/{task_id}/articles`.
- Sends feedback through `/api/article/{article_id}/submit_feedback`.

Feedback types:

- Agree with model.
- Choose Human/AI label with explanation.

### `templates/results.con.html`

Route:

```text
/results_con
```

Query parameters:

```text
projectId
taskId
mode
model
```

Purpose:

- Main results page for generated conversation analysis.
- In model-selection mode, it lets examiner run conversation analysis and submit selected model.
- In feedback mode, it shows conversation turns and lets examiners submit turn-level feedback.

Important APIs:

```text
GET  /api/analysis_project/{project_id}
POST /api/run_analysis_project/{project_id}
POST /api/conversation/select_model_task
GET  /api/conversation/selected_model_task
GET  /api/task/{task_id}/conversation_feedback_list
POST /api/task/{task_id}/conversation_feedback/{conversation_id}/turn/{turn_index}/submit
```

### `templates/ConversationAnalysisResults.html`

Route:

```text
/projectdetailsexaminer/{project_id}/conversation-analysis
```

Purpose:

- Uploaded conversation dataset detection page.
- Used when a project category is Conversation and `generated_from_scratch` is false.
- Lets the examiner run conversation detection on rows stored under `datasets/uploaded_conversations/{dataset_id}`.
- Uses the conversation logistic baseline through `conversation_baseline_model.py`.
- Stores versioned analysis runs under `analysis_results/conversations/{model_key}/{project_id}`.
- Supports viewing dialogue details and exporting enriched detection results.

Important APIs:

```text
GET  /api/project/{project_id}/conversation_dataset
POST /api/project/{project_id}/analyze_conversations
GET  /api/project/{project_id}/conversation_analysis_results
GET  /api/project/{project_id}/conversation_dialogue/{dialogue_id}
GET  /api/project/{project_id}/conversation_export
```

### `templates/index.html`

This appears to be an older/generated index page that links to static page names under `./pages/`. The active app route `/` renders `HomePage.html`, so this file is not part of the main active flow.

---

## 11. Backend Route Map

### Public pages

| Method | Route | Template / Purpose |
|---|---|---|
| GET | `/` | `HomePage.html` |
| GET | `/login` | `Login.html` |
| GET | `/signup` | `signup.html` |
| GET | `/verified` | `Verified.html` |
| GET/POST | `/forgot` | `ForgotPassword.html` |
| GET | `/health` | Health check JSON |

### Auth APIs

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/signup` | Create Firebase Auth user and Firestore user document |
| GET | `/auto-login` | Sign user in after email verification |
| POST | `/api/signin` | Login and set Flask session |
| GET | `/logout` | Clear session |
| POST | `/api/reset` | Send password reset email |
| POST | `/api/update-profile` | Update current user profile |

### Owner and examiner pages

| Method | Route | Purpose |
|---|---|---|
| GET | `/profile` | Current user profile |
| GET | `/ownerdashboard` | Owner dashboard |
| GET | `/examinerdashboard` | Examiner dashboard |
| GET | `/createproject` | Create/edit project page |
| GET | `/myprojectowner` | Owner project list |
| GET | `/myprojectexaminer` | Examiner accepted projects |
| GET | `/projectdetailsowner/{project_id}` | Owner project details |
| GET | `/projectdetailsexaminer/{project_id}` | Examiner project details |
| GET | `/projectdetailsexaminer/{project_id}/conversation-analysis` | Uploaded conversation dataset analysis page |
| GET | `/invitation` | Examiner invitation page |
| GET | `/feedback` | Examiner feedback overview |

### Project APIs

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/my_projects` | Projects owned by current owner |
| POST | `/api/create_project` | Create project and optional invitations |
| POST | `/api/update_project/{project_id}` | Update owner project |
| DELETE | `/api/delete_project/{project_id}` | Delete owner project and invitations |
| GET | `/api/project_json_owner/{project_id}` | Owner-safe project JSON |
| GET | `/api/project_json/{project_id}` | Examiner-safe project JSON |
| GET | `/api/project/{project_id}/dataset` | Get project dataset rows |

### Invitation and examiner APIs

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/volunteers` | List volunteer examiners |
| POST | `/api/send_invitation` | Send invitation by email |
| POST | `/api/add_examiner_to_project` | Add/invite examiner to project |
| POST | `/api/remove_examiner` | Remove examiner invitation and task assignment |
| GET | `/api/invitations` | Current examiner invitations |
| PATCH | `/api/invitations/{invitation_id}` | Accept/reject invitation |
| GET | `/api/accepted_projects` | Examiner accepted projects |
| GET | `/api/project_examiners_owner/{project_id}` | Owner view of project examiners |
| GET | `/api/project_examiners/{project_id}` | Examiner view of project examiners |
| GET | `/api/project_examiners_for_task/{project_id}` | Examiners available for task assignment |

### Task APIs

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/create_task` | Create project task |
| GET | `/projects/{project_id}/tasks/create` | Task creation page |
| GET | `/api/project_tasks/{project_id}` | Owner task list |
| GET | `/api/examiner_tasks/{project_id}` | Examiner assigned task list |
| GET | `/api/tasks/{task_id}` | Get task details |
| POST | `/api/tasks/{task_id}/delete` | Delete task |
| PATCH | `/api/update_task/{task_id}` | Update task |

### Conversation APIs

| Method | Route | Purpose |
|---|---|---|
| GET | `/conversation-ai` | Human-AI conversation page |
| GET | `/conversation-hh` | Human-Human conversation page |
| POST | `/api/ai_reply` | Save examiner message and generated AI reply |
| GET | `/api/ai/messages` | Get Human-AI messages |
| GET | `/api/hh/messages` | Get Human-Human messages |
| POST | `/api/hh/send` | Send Human-Human message |
| GET | `/api/hh/messages_owner` | Owner view of HH messages |
| GET | `/api/llm/messages_owner` | Owner view of Human-AI messages |

### Article/news analysis APIs

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/project/{project_id}/analyze_all` | Run batch article analysis |
| GET | `/task/{task_id}/model-selection` | Article model-selection page |
| POST | `/api/task/{task_id}/run_model` | Run article model for task |
| POST | `/api/task/{task_id}/select_model` | Save selected article model |
| GET | `/api/task/{task_id}/articles` | Load articles for feedback |
| POST | `/api/article/{article_id}/submit_feedback` | Submit article feedback |
| POST | `/api/article/{article_id}/feedback` | Alternative article feedback endpoint |
| GET | `/api/article/{article_id}/feedbacks` | Get article feedbacks |

### Conversation analysis and feedback APIs

| Method | Route | Purpose |
|---|---|---|
| GET | `/results_con` | Conversation results and feedback page |
| POST | `/api/run_analysis_project/{project_id}` | Run generated-conversation analysis |
| GET | `/api/analysis_project/{project_id}` | Get analysis summary |
| POST | `/api/conversation/select_model_task` | Save selected conversation model |
| GET | `/api/conversation/selected_model_task` | Check saved selected model |
| GET | `/api/task/{task_id}/conversation_feedback_list` | Load conversation feedback list |
| POST | `/api/task/{task_id}/conversation_feedback/{conversation_id}/turn/{turn_index}/submit` | Submit turn feedback |
| GET | `/project/{project_id}/analysis/examiner` | Redirect to results page |

### Uploaded conversation dataset analysis APIs

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/project/{project_id}/conversation_dataset` | Preview uploaded conversation dataset grouped by dialogue |
| POST | `/api/project/{project_id}/analyze_conversations` | Run baseline detection on uploaded conversation dataset |
| GET | `/api/project/{project_id}/conversation_analysis_results` | Get latest uploaded conversation analysis results |
| GET | `/api/project/{project_id}/conversation_dialogue/{dialogue_id}` | Get one dialogue with turn-level predictions |
| GET | `/api/project/{project_id}/conversation_export` | Export enriched uploaded conversation rows after detection |
| GET | `/api/project/{project_id}/active_learning_export` | Export reviewed Active Learning samples for offline Logistic Regression retraining |

### Examiner progress and rating APIs

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/project/{project_id}/examiner_progress` | Count article feedback by examiner |
| POST | `/api/project/{project_id}/rate_examiner` | Owner rates examiner |
| GET | `/api/project/{project_id}/examiner_feedback_count` | Owner gets feedback counts |
| GET | `/api/project/{project_id}/my_rating` | Examiner gets their rating |

---

## 12. Owner Flow

### Flow A: Owner creates an article/news project

1. Owner logs in.
2. Owner lands on `/ownerdashboard`.
3. Owner opens `/createproject`.
4. Owner enters:
   - Project name.
   - Description.
   - Category as article/news.
   - Dataset CSV file.
   - Optional invited examiners.
5. Frontend posts to:

```text
POST /api/create_project
```

6. Backend:
   - Creates `project_id`.
   - Creates `dataset_id`.
   - Validates that article projects include a dataset.
   - Saves project document in Firestore `projects`.
   - Saves CSV rows in Realtime Database under `datasets/uploaded_news/{dataset_id}`.
   - Creates invitation documents for selected examiners.
7. Owner is redirected to `/myprojectowner`.
8. Owner opens `/projectdetailsowner/{project_id}`.
9. Owner creates a task from `/projects/{project_id}/tasks/create`.
10. Task appears in owner and examiner task lists.

### Flow B: Owner creates a generated conversation project

1. Owner logs in.
2. Owner opens `/createproject`.
3. Owner selects conversation/generated project category.
4. Owner can choose no dataset and generate from scratch.
5. Backend creates a project document.
6. Owner invites examiners.
7. Owner opens project details.
8. Owner creates conversation tasks:
   - Human-AI task: exactly one examiner.
   - Human-Human task: exactly two examiners.
9. After conversation tasks are completed, owner/examiner can create model-selection and labeling tasks.

### Flow C: Owner creates an uploaded conversation dataset project

1. Owner logs in.
2. Owner opens `/createproject`.
3. Owner selects Conversation.
4. Owner uploads a CSV dataset and does not choose Generate from scratch.
5. Backend saves the project with `generated_from_scratch = false`.
6. Backend ingests the CSV rows into:

```text
datasets/uploaded_conversations/{dataset_id}
```

7. Owner invites examiners.
8. Owner opens `/projectdetailsowner/{project_id}`.
9. Owner creates a `model_selection` task. This task requires exactly one examiner.
10. Later, owner can create a `labeling` task for review/feedback work suitable for uploaded conversations.

### Flow D: Owner monitors project

Owner uses `/projectdetailsowner/{project_id}` to:

- View project info.
- View examiners.
- Add/remove examiners.
- View tasks.
- Create new tasks.
- Edit tasks.
- Delete tasks.
- View feedback counts.
- Rate examiners.

---

## 13. Examiner Flow

### Flow A: Examiner accepts invitation

1. Examiner logs in.
2. Examiner opens `/invitation`.
3. Page loads invitations from:

```text
GET /api/invitations
```

4. Examiner accepts or rejects through:

```text
PATCH /api/invitations/{invitation_id}
```

5. Accepted projects appear under `/myprojectexaminer`.

### Flow B: Examiner works on article/news model-selection task

1. Examiner opens `/myprojectexaminer`.
2. Examiner opens project details.
3. Examiner clicks model-selection task.
4. Page opens:

```text
/task/{task_id}/model-selection
```

5. Examiner runs model analysis:

```text
POST /api/task/{task_id}/run_model
```

6. Examiner selects model:

```text
POST /api/task/{task_id}/select_model
```

7. Backend saves selected model on the Firestore task and marks task completed.

### Flow C: Examiner works on article/news feedback task

1. Examiner opens feedback task:

```text
/task/{task_id}/feedback
```

2. Page loads articles:

```text
GET /api/task/{task_id}/articles
```

3. Examiner reviews prediction.
4. Examiner submits feedback:

```text
POST /api/article/{article_id}/submit_feedback
```

5. Feedback is saved in Realtime Database under the article node.

### Flow D: Examiner completes Human-AI conversation

1. Examiner opens assigned Human-AI task.
2. Page opens:

```text
/conversation-ai?taskId={task_id}&projectId={project_id}
```

3. User sends message to:

```text
POST /api/ai_reply
```

4. Backend saves examiner message.
5. Backend calls `generate_reply()` from `llm_service.py`.
6. Backend saves AI response.
7. Messages are loaded with:

```text
GET /api/ai/messages?taskId={task_id}
```

8. When required turns are complete, backend marks task completed.

### Flow E: Examiner completes Human-Human conversation

1. Examiner opens assigned Human-Human task.
2. Page opens:

```text
/conversation-hh?taskId={task_id}&projectId={project_id}
```

3. Messages load through:

```text
GET /api/hh/messages?taskId={task_id}
```

4. Examiner sends message through:

```text
POST /api/hh/send
```

5. Backend prevents the same examiner from sending twice in a row.
6. Backend checks turn counts for both examiners.
7. When required turns are complete, backend marks task completed.

### Flow F: Examiner selects conversation model

1. Conversation tasks must contain completed messages.
2. Examiner opens `results.con.html` in model-selection mode.
3. Examiner chooses Logistic Regression or RNN.
4. Page calls:

```text
POST /api/run_analysis_project/{project_id}?model={logreg|rnn}&task_id={task_id}
```

5. Backend:
   - Reads conversation messages from Realtime Database.
   - Runs selected model.
   - Saves turn-level predictions.
   - Calculates summary metrics.
6. Examiner saves selected model through:

```text
POST /api/conversation/select_model_task
```

7. Task is marked completed.

### Flow G: Examiner runs uploaded conversation dataset detection

1. Examiner opens an accepted Conversation project that was created with an uploaded CSV dataset.
2. Examiner clicks a `model_selection` task.
3. Because the project is Conversation and `generated_from_scratch = false`, the task opens:

```text
/projectdetailsexaminer/{project_id}/conversation-analysis
```

4. The page uses `ConversationAnalysisResults.html`.
5. Examiner selects the baseline conversation model and runs analysis:

```text
POST /api/project/{project_id}/analyze_conversations
```

6. Backend reads uploaded rows from:

```text
datasets/uploaded_conversations/{dataset_id}
```

7. Backend groups rows into dialogues, predicts each turn, computes summary metrics, and saves the run under:

```text
analysis_results/conversations/{model_key}/{project_id}/runs/{run_id}
```

8. The page reads results from:

```text
GET /api/project/{project_id}/conversation_analysis_results
```

9. Examiner can view dialogue details or export enriched rows.

### Flow H: Examiner labels generated conversation feedback

1. Examiner opens a labeling task.
2. Page opens:

```text
/results_con?projectId={project_id}&taskId={task_id}&mode=feedback
```

3. Page calls:

```text
GET /api/task/{task_id}/conversation_feedback_list
```

4. Backend checks that a model-selection task is completed.
5. Backend loads model predictions.
6. Examiner reviews each turn.
7. Examiner can:
   - Agree with model.
   - Select Human/AI manually.
   - Add explanation.
8. Feedback is submitted to:

```text
POST /api/task/{task_id}/conversation_feedback/{conversation_id}/turn/{turn_index}/submit
```

9. Backend saves turn feedback and updates task progress/completion.

---

## 14. Machine Learning Logic

### Article/news Logistic Regression pipeline

Loaded in `app.py`:

```python
news_pipeline = joblib.load('news_baseline_pipeline.pkl')
```

Used for article/news prediction. The backend reads article text, runs the pipeline, and converts probability output into:

```text
Human
AI
confidence
```

### Article/news RNN

Loaded in `app.py`:

```python
rnn_model = tf.keras.models.load_model("news_rnn_baseline.keras")
rnn_tokenizer = joblib.load("news_rnn_tokenizer.pkl")
```

Text is tokenized and padded:

```text
maxlen = 600
padding = post
truncating = post
```

The model output is interpreted as AI probability. Human probability is:

```text
1 - p_ai
```

### Article explanation/chunking

`split_into_3_chunks(text)` splits article text into three chunks. The analysis APIs use chunk-level predictions to provide explanation-style details for parts of the article.

### Conversation Logistic Regression

Loaded in `app.py`:

```python
conv_logreg_model = joblib.load("models/conversation_logistic_regression.joblib")
```

Used through model key:

```text
tfidf_logreg
```

The model predicts Human/AI labels for conversation turns. The same logistic baseline file is also loaded lazily by `conversation_baseline_model.py` for uploaded conversation dataset detection.

### Conversation RNN

Loaded in `app.py`:

```python
conv_rnn_model = tf.keras.models.load_model("Model-Gen-Con/rnn_v2_model.keras")
conv_rnn_tokenizer = joblib.load("Model-Gen-Con/rnn_v2_tokenizer.pkl")
```

Configuration file:

```json
{
  "use_previous_turn": true,
  "max_len": 5085,
  "col_text": "text",
  "col_prev_text": "prev_text",
  "sep_token": "[SEP]"
}
```

This means the model may use the current turn text plus previous-turn context.

### Uploaded conversation dataset detection

The uploaded conversation dataset route uses:

```python
from conversation_baseline_model import predict_one_turn
```

The helper loads:

```text
models/conversation_logistic_regression.joblib
```

and accepts current turn text plus optional previous-turn text. Uploaded rows are normalized through helper logic in `app.py`, which attempts to detect flexible CSV column names such as:

```text
dialogue_id / conversation_id / chat_id
turn_index / turn_number / message_index
text / utterance / message / content
sender / role / author
ground_truth / label / target / is_ai
```

If no dialogue ID exists in the uploaded dataset, the backend groups all valid rows into one automatic dialogue for analysis.

---

## 15. Task Status Logic

Task statuses used across the app:

```text
pending
progress
completed
active
```

Important behavior:

- New tasks start as `pending`.
- Conversation tasks become `completed` after required turns are reached.
- Article model-selection task becomes `completed` after selected model is saved.
- Conversation model-selection task becomes `completed` after selected model is saved.
- Conversation labeling task becomes:
  - `pending` if no conversations are reviewed.
  - `progress` if some feedback exists.
  - `completed` if all conversations are reviewed.

Project status is derived from task statuses in several API responses. If all tasks are completed, project status is treated as completed. If some are in progress, project status is treated as progress.

---

## 16. Important Implementation Notes

### Most logic is currently in `app.py`

The codebase is functional but centralized. A future team may later split `app.py` into blueprints/services, but that would be a structural refactor. The current project structure keeps all backend routes together.

### Firebase is required for most pages

Most authenticated pages depend on Firestore and Realtime Database. Running the app without Firebase credentials will fail during startup because `firebase_admin_setup.py` loads `service-account.json`.

### LLM loading happens at import time

`llm_service.py` loads the Llama model when imported. Because `app.py` imports `generate_reply` from `llm_service.py`, startup can be slow and can fail if:

- `HF_TOKEN` is missing.
- Hugging Face login fails.
- Model download/cache is unavailable.
- The local machine lacks memory for the model.

### ML models load at startup

The following models load when `app.py` starts:

```text
models/conversation_logistic_regression.joblib
Model-Gen-Con/rnn_v2_model.keras
Model-Gen-Con/rnn_v2_tokenizer.pkl
news_baseline_pipeline.pkl
news_rnn_baseline.keras
news_rnn_tokenizer.pkl
```

If any file is missing, the app will fail before serving pages.

`conversation_baseline_model.py` also expects the logistic conversation model to exist at:

```text
models/conversation_logistic_regression.joblib
```

### Some frontend pages contain inline JavaScript

Most templates include their own CSS and JS directly inside the HTML file. This means behavior for each page is mostly local to that template rather than shared in separate frontend modules.

### Translation support exists in two places

Shared files:

```text
static/js/i18n.js
static/js/translations.js
```

Several templates also include their own language switching logic using `localStorage` or `sessionStorage`.

### Theme support exists per page

Many templates store theme state in:

```text
localStorage.theme
```

or similar keys. The theme implementation is page-specific rather than fully centralized.

---

## 17. Active Learning Support

The project now includes the application-side workflow needed to collect Active Learning samples for the Logistic Regression models.

Active Learning currently applies only to Logistic Regression based models:

```text
news_baseline_pipeline.pkl
models/conversation_logistic_regression.joblib
```

RNN models can still run detection, but they are not part of the Active Learning retraining workflow.

### Confidence and uncertainty

The backend calculates both `confidence` and `uncertainty` for supported detection outputs.

For a binary model probability `p` for the AI/Machine class:

```python
confidence = max(p, 1 - p)
uncertainty = abs(p - 0.5)
```

Lower `uncertainty` means the model is closer to a 50/50 decision. These are the samples most useful for examiner review.

### Sample selection rule

For Active Learning review, the app selects:

```text
20% of available samples, capped at 50 samples
```

In code terms:

```python
selected_count = min(ceil(total * 0.20), 50)
```

If at least one sample exists, the selection keeps at least one sample.

### News Active Learning

For article/news projects:

- The selection unit is the article.
- The backend sorts article results by `uncertainty`.
- The feedback page shows the selected review-priority samples.
- The examiner submits feedback through:

```text
POST /api/article/{article_id}/submit_feedback
```

The feedback page supports filters and sorting by:

- Review targets only.
- Pending only.
- Reviewed only.
- Human prediction.
- AI prediction.
- Most uncertain first.
- Least uncertain first.
- Lowest confidence first.
- Highest confidence first.

### Conversation Active Learning

For generated and uploaded conversation projects:

- The selection unit is the conversation turn.
- The UI still displays the dialogue/conversation context.
- Turns selected for review are marked as review targets.
- Non-selected turns are shown as context only.
- Feedback is saved at turn level.

Generated conversation feedback uses:

```text
POST /api/task/{task_id}/conversation_feedback/{conversation_id}/turn/{turn_index}/submit
```

Uploaded conversation feedback uses:

```text
POST /api/task/{task_id}/uploaded_conversation_feedback/{dialogue_id}/turn/{turn_index}/submit
```

The conversation feedback page supports:

- Dialogue filters:
  - All dialogues.
  - Dialogues with review targets.
  - Pending targets.
  - Reviewed targets.
- Dialogue sorting:
  - Most uncertain first.
  - Least uncertain first.
  - Most pending targets.
- Turn filters inside each dialogue:
  - All turns.
  - Review targets only.
  - Context only.
  - Pending targets.
  - Reviewed targets.
- Turn sorting:
  - Default order.
  - Most uncertain first.
  - Least uncertain first.

### Exporting reviewed samples

Reviewed Active Learning samples can be exported with:

```text
GET /api/project/{project_id}/active_learning_export
```

The endpoint returns JSON rows suitable for offline retraining.

For news projects, exported rows include fields such as:

```text
title
Article
text
MachineGen
label
confidence
uncertainty
examiner_uid
submitted_at
```

For conversation projects, exported rows include fields such as:

```text
text
prev_text
MachineGen
label
confidence
uncertainty
examiner_uid
submitted_at
```

### What is not automated yet

The app currently supports:

- Calculating uncertainty.
- Selecting review-priority samples.
- Collecting examiner feedback.
- Exporting reviewed samples.

The app does not yet automatically retrain or replace model files.

The next offline step should be done in Colab or another controlled training environment:

1. Load the original training dataset.
2. Load the exported Active Learning feedback rows.
3. Merge the reviewed rows into the training data.
4. Retrain the Logistic Regression pipeline.
5. Evaluate before/after performance.
6. Save a new model file.
7. Replace or version the deployed model only after validation.

The uploaded temporary training references used for this work are under:

```text
active_learning_tmp/
```

This folder is for development and handoff only. It should not be treated as a production runtime dependency.

---

## 18. Known Technical Risks and Follow-Up Items

These are not changes made to the project; they are notes for the next team.

### 1. Secrets should not be hardcoded

Move Flask secret key to `.env`.

### 2. Runtime dependencies should be verified

The source imports packages that may not all be listed in `requirements.txt`, including:

```text
tensorflow
pandas
scikit-learn
joblib
```

The team should create a clean environment and update `requirements.txt` if needed.

### 3. PDF document should be converted to text for deeper comparison

The PDF exists in `docs/`, but this environment could not extract its text. A future team should export it to text or install a PDF extraction tool, then compare requirements against implementation.

### 4. `service-account.json` is required but not documented in code

The setup instructions should always mention it because the app imports Firebase during startup.

### 5. `app.py` is very large

This is acceptable for the current project structure, but future maintainers should be careful when editing it. Changes to one section can affect page flows elsewhere.

### 6. LLM startup can block the whole app

Because the LLM is loaded on import, a failure in `llm_service.py` can prevent the Flask app from starting. A production version may lazy-load the LLM only when `/api/ai_reply` is called.

### 7. Some old/static links remain

Some templates contain links like `Settings.html`, `Ownerdashboard.html`, `ExaminerDashord.html`, or `Explainability.html`. The active Flask routes use paths like `/profile`, `/ownerdashboard`, and `/examinerdashboard`. These should be reviewed before production.

---

## 19. Quick Handover Checklist

Before a new team runs or continues development, confirm:

- Python environment is created.
- Dependencies are installed.
- `service-account.json` exists in project root.
- `.env` contains `FIREBASE_WEB_API_KEY`.
- `.env` contains `HF_TOKEN` if Human-AI replies are needed.
- All `.pkl`, `.keras`, and tokenizer files exist.
- Firebase Firestore and Realtime Database rules allow the server-side operations.
- Test signup/login with email verification.
- Test owner project creation.
- Test examiner invitation acceptance.
- Test article model-selection task.
- Test article feedback task.
- Test Human-AI conversation.
- Test Human-Human conversation.
- Test generated conversation model selection.
- Test generated conversation feedback.
- Test uploaded conversation project creation.
- Test uploaded conversation model-selection task opens `ConversationAnalysisResults.html`.
- Test uploaded conversation analysis, dialogue details, and export.
- Test Active Learning target selection in article feedback tasks.
- Test Active Learning target selection in generated conversation feedback tasks.
- Test Active Learning target selection in uploaded conversation feedback tasks.
- Test `/api/project/{project_id}/active_learning_export` after examiner feedback is submitted.
- Retrain Logistic Regression offline only after exported feedback rows are validated.

---

## 20. Project Team

TrustLens was developed for the King Saud University graduation project.

Team G5:

- Haifa Alsaif
- Amira Aljeraisy
- Nouf Al-Muhanna
- Afnan Alzakary

Supervisor:

- Dr. Abeer Aldayel

---

## 21. Repository Reference

GitHub repository:

```text
https://github.com/HaifaAlsaif/2025-GP1-G5.git
```
