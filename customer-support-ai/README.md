# 🎫 AI Customer Ticket Classifier

An end-to-end NLP pipeline that automatically **classifies** customer support tickets into queues, **predicts** priority levels, and **generates** professional draft responses using a Large Language Model.

Built as a full-stack enterprise-style web application with a FastAPI backend and a clean dashboard UI.

---

## Architecture

```
Customer Ticket
       │
       ▼
 ┌─────────────┐
 │  Classifier  │  ← DistilBERT (Queue) + LinearSVC (Priority)
 │  Layer       │
 └──────┬──────┘
        │
   ┌────┴────┐
   │         │
 Queue    Priority    Confidence
   │         │            │
   └────┬────┘            │
        ▼                 │
 ┌─────────────┐          │
 │  Prompt     │ ◄────────┘
 │  Builder    │
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  Gemini LLM │  ← Gemini 3.5 Flash
 └──────┬──────┘
        ▼
  Draft Response
```

The **Classifier** and the **LLM** are deliberately separate components.  
The classifier *understands* the ticket. The LLM *writes* the response.

---

## Features

- **Queue Classification** — DistilBERT fine-tuned on customer support data to route tickets to the correct department.
- **Priority Prediction** — TF-IDF + LinearSVC pipeline classifies tickets as High / Medium / Low priority.
- **AI Response Generation** — Gemini 3.5 Flash drafts a professional, context-aware reply using the classifier's output.
- **Enterprise Dashboard UI** — Clean, light-themed SaaS-style interface with sidebar navigation, metrics cards, and a typewriter response effect.
- **Full ML Experimentation Trail** — Notebooks documenting the progression from Word2Vec → RNN → BiLSTM → DistilBERT.

---

## Screenshots

> **Add these screenshots to a `screenshots/` folder in your repo and update the paths below.**

| View | Description |
|------|-------------|
| `screenshots/dashboard.png` | The main Triage Dashboard with the text input area |
| `screenshots/results.png` | After clicking "Analyze" — shows Queue, Priority, Confidence metrics and the AI-drafted response |
| `screenshots/inbox.png` | The Inbox view showing queued tickets |

```markdown
<!-- Uncomment after adding screenshots -->
<!-- ![Dashboard](screenshots/dashboard.png) -->
<!-- ![Results](screenshots/results.png) -->
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Queue Classifier | DistilBERT (HuggingFace Transformers) |
| Priority Classifier | TF-IDF + LinearSVC (scikit-learn) |
| Response Generation | Google Gemini 3.5 Flash |
| Backend API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JS |
| Experiment Tracking | MLflow |
| Training Framework | PyTorch + HuggingFace Accelerate |

---

## Project Structure

```
customer-support-ai/
├── src/
│   ├── api/
│   │   ├── app.py                 # FastAPI server
│   │   └── static/                # Frontend (HTML, CSS, JS)
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── inference_pipeline.py  # DistilBERT + LinearSVC inference
│   ├── services/
│   │   └── llm_service.py         # Gemini API integration
│   ├── config/
│   ├── entity/
│   └── utils/
├── notebooks/                      # Research notebooks (Word2Vec → DistilBERT)
├── configs/                        # YAML config files
├── artifacts/                      # Trained model weights (gitignored)
├── train_priority.py               # Script to train the priority classifier
├── requirements.txt
└── .env                            # GEMINI_API_KEY (not committed)
```

---

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/your-username/NLP-customer-ticket-classifier.git
cd NLP-customer-ticket-classifier/customer-support-ai

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Add your Gemini API Key

Create a `.env` file in `customer-support-ai/`:

```
GEMINI_API_KEY=your_api_key_here
```

Get a free key at [Google AI Studio](https://aistudio.google.com/apikey).

### 3. Train the Models

```bash
# Train the DistilBERT queue classifier + full ML pipeline
python main.py

# Train the priority classifier
python train_priority.py
```

### 4. Run the App

```bash
python -m uvicorn src.api.app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Notebooks

The `notebooks/` folder documents the full research journey:

| # | Notebook | What it covers |
|---|----------|---------------|
| 02 | `word2vec.ipynb` | Word2Vec embeddings on ticket data |
| 03 | `sentence_embeddings.ipynb` | Sentence-level embeddings |
| 04 | `rnn_intuition.ipynb` | Vanilla RNN baseline |
| 05 | `lstm_intuition.ipynb` | LSTM architecture |
| 06 | `bilstm_training.ipynb` | BiLSTM with full training loop |
| 07 | `distilbert_intuition_and_training.ipynb` | DistilBERT fine-tuning (final model) |

---

## License

MIT
