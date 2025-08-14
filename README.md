## Project Structure

```
Resume Builder/
├── job_hunter.py                # Job aggregation, scoring, and API ingestion
├── job_store.py                 # Database helpers, upsert logic, status/resume updates
├── openai_utils.py              # Centralized OpenAI API calls and usage logging
├── models_config.py             # Model/pricing config loader and UI helpers
├── openai_pricing.yaml          # Model selection and pricing config
├── jobs_cli.py                  # CLI tools for DB maintenance and migration
├── modular_resume_full.yaml     # User's modular resume (YAML)
├── tailored_resume_template.html# Jinja2 template for resume rendering
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (API keys, timezone, etc.)
├── README.md                    # Project documentation
└── pages/
    ├── 01_job_finder.py         # Job finder/search panel
    ├── 02_jobs_report.py        # Jobs report panel (filter, export)
    ├── 03_job_history.py        # Job history panel (status, actions)
    ├── 04_Generate_From_Jobs.py # Generate tailored resume from job
    ├── 05_Usage_Analytics.py    # OpenAI usage analytics panel
```

## Streamlit Screens Overview

- **Job Finder (01_job_finder.py):**
  Search and filter jobs from all sources, view scores, and select jobs for further action.

- **Jobs Report (02_jobs_report.py):**
  View, filter, and export job listings. Apply advanced filters and see job details in tabular format.

- **Job History (03_job_history.py):**
  Track job status (submitted, not suitable, saved), update flags, and review job actions over time.

- **Generate From Jobs (04_Generate_From_Jobs.py):**
  Generate a tailored resume for a selected job, using semantic scoring and GPT-powered paraphrasing. Download resumes as HTML or PDF.

- **Usage Analytics (05_Usage_Analytics.py):**
  View OpenAI API usage, cost analytics, and model breakdowns for all API calls made by the app.

# Tailored Resume Builder

## Overview
Tailored Resume Builder is a Python/Streamlit application that aggregates job listings from multiple sources, scores them using OpenAI embeddings, and helps users generate customized resumes for specific jobs. It supports robust filtering, cost tracking, and analytics, with a modular architecture for easy extension.

## Features
- Aggregates jobs from sources like RemoteOK, Remotive, WeWorkRemotely, and more
- Uses OpenAI embeddings for semantic job scoring (optional)
- Tracks API usage and cost analytics
- Standardizes date/time formats and supports timezone configuration
- CLI tools for database maintenance and migration
- Resume generation with Jinja2 templates and PDF export
- UI panels for job history, analytics, and tailored resume creation

## Requirements
- Python 3.8+
- Streamlit
- OpenAI API key (for embeddings/scoring)
- SQLite (default jobs.db)
- Optional: WeasyPrint (for PDF export)
- Other dependencies: see `requirements.txt`

## Quick Start
1. Clone the repository:
  ```
  git clone https://github.com/travisekarr/tailored-resume.git
  cd tailored-resume
  ```
2. Install dependencies:
  ```
  python -m venv venv
  source venv/bin/activate  # or venv\Scripts\activate on Windows
  pip install -r requirements.txt
  ```
3. Set up your `.env` file:
  ```
  OPENAI_API_KEY=your-key-here
  TIMEZONE=America/New_York
  DATETIME_DISPLAY_FORMAT=%Y-%m-%d:%I-%M %p
  ```
4. Run the application:
  ```
  streamlit run streamlit_app.py
  ```

## CLI Tools

- `jobs_cli.py`: Database maintenance, migration, and job pulls
  - Fix malformed URLs: `python jobs_cli.py fix-remoteok-urls`
  - Migrate date/time fields: `python jobs_cli.py migrate-datetime-fields`
  - Show job counts: `python jobs_cli.py counts`
  - **Manual job pull:**
    ```bash
    python jobs_cli.py pull-new-jobs --db jobs.db --sources sources.yaml --use-embeddings --embedding-model text-embedding-3-small
    ```
    - Pulls new jobs from all configured sources and updates the database.
    - Use `--use-embeddings` to score jobs with OpenAI embeddings.
    - Specify `--embedding-model` to select the model for scoring.

  - **Automated/scheduled pulls:**
    - Use Windows Task Scheduler or cron to run the above command on a schedule (e.g., daily or hourly).
    - Example (Windows Task Scheduler):
      - Action: `python jobs_cli.py pull-new-jobs --db jobs.db --sources sources.yaml --use-embeddings`
      - Trigger: Daily at 8:00 AM

  - Refer to the CLI help for all options:
    ```bash
    python jobs_cli.py --help
    ```

## Scoring & Embeddings
- Jobs are scored using either TF-IDF or OpenAI embeddings
- Only jobs scored with embeddings have their scores updated and the scoring model recorded
- The `scoring_model` field tracks which model was used for each job

## Customization
- Configure job sources in `sources.yaml`
- Adjust model/pricing in `openai_pricing.yaml`
- Change resume templates in `tailored_resume_template.html`

## Troubleshooting
- Ensure your OpenAI API key is set in `.env`
- For PDF export, install WeasyPrint: `pip install weasyprint`
- Use CLI tools for database fixes and migrations

## License
MIT

## Author
Travis Ekarr
  cp modular_resume_full_template.yaml modular_resume_full.yaml
