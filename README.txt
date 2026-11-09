THE RELAY — LOCAL COMPANY DEMO DASHBOARD

This is a local Streamlit web app that uses the existing Relay project files. It is a project dashboard, not a learning guide.

WHAT IT SHOWS
- Select small_k3, medium_k4, or large_k5_rerank.
- Enter a question and send one real Groq request.
- Inspect BM25-ranked passages, page citations, answer, and latency.
- Grade the answer and append the run to results.csv.
- View or download results.csv and preview a CSV from another run.

INSTALL
1. Extract app.py into the project root, beside the existing src, data, and .env paths.
2. Activate the project's .venv.
3. Install Streamlit in that environment:

   python -m pip install streamlit

4. Keep the existing GROQ_API_KEY line in the project-root .env and add the selected model, for example:

   GROQ_MODEL=openai/gpt-oss-120b

   Use only a model enabled for your Groq account. Do not share or commit the .env file.
5. From the project root, launch the dashboard:

   python -m streamlit run .\app.py

Streamlit opens a local browser tab, usually at http://localhost:8501.

NOTES
- This dashboard reads the project's existing data/processed/chunks_*.jsonl and results.csv files.
- Clicking “Run real Groq request” sends one actual API request and consumes the account's applicable limits.
- An answer is saved to results.csv after you select its quality grade. API failures are also logged.
- It does not deploy publicly. The local Python process reads the API key, so the key is not put into browser-side code.
