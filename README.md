# 🎯 Tailored Resume Builder

This project is a **Streamlit web app** that generates customized, job-specific resumes from a modular YAML resume.  
It supports **offline (free)** keyword matching and **GPT-powered** semantic matching with tailored summaries.

---

## 🚀 Features

- **Tailored Summary at the Top**
  - Offline keyword-based version (free)
  - GPT-powered version (paid API usage)
- **Experience Ordering**
  - Relevancy First
  - Chronological
  - Hybrid (Top N most relevant, then chronological)
- **Matching Modes**
  - Offline keyword scoring (no API cost)
  - OpenAI embeddings (semantic match, API cost)
- **Download as HTML** from the Streamlit UI
- Modular YAML resume format for easy updates

---

## 📦 Requirements

- Python 3.9+
- Git
- (Optional) OpenAI API key for GPT-powered features

---

## 🛠️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/resume-tailor.git
cd resume-tailor
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add environment variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
> Only required if you want to use **GPT-powered summary** or **semantic matching**.

---

## 📂 Resume File Setup

This project uses a modular YAML file called `modular_resume_full.yaml`.

- **Your real resume** should not be uploaded to GitHub.
- A **blank template** is provided:
  ```
  modular_resume_full_template.yaml
  ```
- **Before running the app**, copy this template and rename it:
  ```bash
  cp modular_resume_full_template.yaml modular_resume_full.yaml
  ```
- Fill in your information in `modular_resume_full.yaml` following the same structure.

**Example template structure:**
```yaml
- id: summary
  type: summary
  title: Executive Summary
  tags: []
  summary: >
    Your professional summary goes here.

- id: core_competencies
  type: skills
  title: Core Competencies
  tags: []
  summary: >
    List your main skill areas.

- id: tech_stack
  type: skills
  title: Technical Proficiencies
  tags: []
  summary: >
    List your technologies here.

- id: exp01
  type: experience
  title: Job Title
  company: Company Name
  location: City, State
  dates: YYYY – YYYY
  tags: []
  summary: >
    Brief description of your role.
  contributions:
    - description: Example contribution.
      skills_used: [Skill1, Skill2]
```

---

## ▶️ Usage

### Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

### In the browser:
1. Paste a job description.
2. Choose:
   - **Summary Mode:** Offline (free) or GPT-powered (API cost)
   - **Experience Ordering:** Relevancy First, Chronological, or Hybrid
   - Toggle **Semantic Matching** on/off
3. Click **Generate Tailored Resume**.
4. Download as HTML.

---

## 📂 Project Structure

```
resume-tailor/
├── modular_resume_full_template.yaml  # Blank resume template
├── resume_semantic_scoring_engine.py  # Scoring with embeddings & ordering
├── resume_scoring_engine.py           # Offline keyword scoring
├── streamlit_app.py                   # Main web app
├── tailored_resume_template.html      # Jinja2 HTML template
├── requirements.txt                   # Dependencies
├── .env                               # API keys (ignored by git)
├── .gitignore                         # Ignores venv, .env, and real resume
└── README.md
```

---

## 💰 API Costs (Optional Features)

| Feature                  | Model                  | Est. Cost per Use |
|--------------------------|------------------------|-------------------|
| Semantic Matching        | text-embedding-3-small | ~$0.00005         |
| GPT-powered Summary      | gpt-3.5-turbo           | ~$0.005           |

---

## ⚠️ Security Notes

- Your `.env` file is **gitignored** by default.
- Never commit your API key or your real resume to GitHub.
- Only the blank template `modular_resume_full_template.yaml` should be versioned.

---

## 📜 License
MIT License
