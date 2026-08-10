<div align="center">

# 🎫 Enterprise AI Customer Support & Ticket Classifier

An end-to-end NLP & LLM platform that automatically classifies customer support tickets, predicts priority levels, and generates empathetic, high-quality draft responses.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Gemini 3.5 Flash](https://img.shields.io/badge/Google-Gemini_3.5_Flash-4285F4?style=flat&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)

</div>

---

## 💡 Overview

In enterprise customer support, speed and accuracy are critical. This project decouples ticket **understanding** (classification & routing) from ticket **generation** (writing the customer response):

```
                       CUSTOMER TICKET
                             │
                             ▼
                ┌─────────────────────────┐
                │    Classifier Layer     │
                │  (DistilBERT + Linear)  │
                └────────────┬────────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
     Department Queue    Priority Level   Confidence %
            │                │                │
            └────────────────┼────────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │     Prompt Builder      │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │    LLM Generation Layer │
                │    (Gemini 3.5 Flash)   │
                └────────────┬────────────┘
                             │
                             ▼
                     SUGGESTED RESPONSE
```

- **Classifier Layer**: Fast, deterministic intent understanding (Queue & Priority).
- **LLM Layer**: Empathetic, context-aware natural language drafting.

---

## ✨ Features

- 🎯 **Queue Classification**: Fine-tuned **DistilBERT** model routing tickets to exact technical/billing queues.
- ⚡ **Priority Scoring**: **TF-IDF + LinearSVC** pipeline determining ticket urgency (*High*, *Medium*, *Low*).
- 🤖 **AI Response Generation**: Powered by **Google Gemini 3.5 Flash**, crafting grounded replies based on department & priority context.
- 🖥️ **SaaS Dashboard UI**: Clean, light-themed responsive web console with tabbed views (*Triage*, *Inbox*, *Analytics*).
- 🔬 **Iterative Model Research**: Full experimental history documented across notebooks (*Word2Vec → RNN → BiLSTM → Transformers*).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Department Classification** | DistilBERT (`transformers`, PyTorch) |
| **Priority Classification** | LinearSVC + TF-IDF (`scikit-learn`) |
| **Generative Drafting** | Google Gemini 3.5 Flash (`google-generativeai`) |
| **Backend Service** | FastAPI, Uvicorn, Pydantic |
| **Frontend UI** | HTML5, Modern Vanilla CSS, JavaScript (ES6 SPA) |
| **ML Tracking & Logs** | MLflow, Joblib |

---

## 📁 Repository Structure

```
NLP-customer-ticket-classifier/
├── customer-support-ai/
│   ├── src/
│   │   ├── api/
│   │   │   ├── app.py                # FastAPI endpoints
│   │   │   └── static/               # Enterprise Dashboard (HTML/CSS/JS)
│   │   ├── pipeline/
│   │   │   ├── inference_pipeline.py # Production inference (DistilBERT + SVC)
│   │   │   └── training_pipeline.py  # Model retraining execution
│   │   └── services/
│   │       └── llm_service.py        # Gemini LLM orchestration
│   ├── notebooks/                    # Sequential research notebooks (02 - 07)
│   ├── artifacts/                    # Saved weights & encoders (gitignored)
│   ├── train_priority.py              # Priority classifier trainer
│   ├── requirements.txt
│   └── .env                          # API Environment keys
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone Repository & Setup Environment

```bash
git clone https://github.com/I-try-to-code/NLP--customer-ticket-classfier.git
cd NLP--customer-ticket-classfier/customer-support-ai

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in `customer-support-ai/`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```
> Get your API key at [Google AI Studio](https://aistudio.google.com/apikey).

### 3. Run the Application

```bash
python -m uvicorn src.api.app:app --reload
```

Open your browser and navigate to: **`http://127.0.0.1:8000`**

---

## 🧪 Model Exploration & Evaluation

Research and benchmarking notebooks located in `customer-support-ai/notebooks/`:

1. `02_word2vec.ipynb` — Word2Vec embedding representations.
2. `03_sentence_embeddings.ipynb` — Sentence transformer baselines.
3. `04_rnn_intuition.ipynb` — Vanilla Recurrent Neural Network.
4. `05_lstm_intuition.ipynb` — Long Short-Term Memory network.
5. `06_bilstm_training.ipynb` — Bidirectional LSTM training loop.
6. `07_distilbert_intuition_and_training.ipynb` — Fine-tuning DistilBERT (Selected for production).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
