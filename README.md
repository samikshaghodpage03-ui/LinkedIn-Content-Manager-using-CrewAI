# 🤖 Autonomous LinkedIn Content Manager
### Powered by CrewAI — 5-Agent Sequential Pipeline

A fully autonomous AI system that researches trends, writes, critiques, optimises, and schedules LinkedIn posts — all in one command.

---

## Architecture

```
User Input (topic)
       │
       ▼
┌─────────────────────┐
│  Trend Researcher   │  ← SerperDevTool + ScrapeWebsiteTool
│  (finds what's hot) │
└────────┬────────────┘
         │ research brief
         ▼
┌─────────────────────┐
│  Content Writer     │  ← GPT-4o
│  (drafts the post)  │
└────────┬────────────┘
         │ draft post
         ▼
┌─────────────────────┐
│  Content Critic     │  ← GPT-4o
│  (scores & reviews) │
└────────┬────────────┘
         │ critique report
         ▼
┌─────────────────────┐
│  Content Optimizer  │  ← GPT-4o
│  (final rewrite)    │
└────────┬────────────┘
         │ publish-ready post
         ▼
┌─────────────────────┐
│  Scheduling Agent   │  ← GPT-4o
│  (timing + brief)   │
└─────────────────────┘
         │
         ▼
   Publishing Brief
```

---

## Quick Start

### 1. Clone / copy the project
```bash
git clone <your-repo> linkedin-manager
cd linkedin-manager
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and SERPER_API_KEY
```

**Where to get keys:**
| Key | URL |
|-----|-----|
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| `SERPER_API_KEY` | https://serper.dev (free tier available) |

### 5. Run the pipeline

**Interactive mode:**
```bash
python main.py
```

**With topic as argument:**
```bash
python main.py "AI in Healthcare"
python main.py "Future of Remote Work"
python main.py "Python for Data Science"
```

---

## Output

The system prints a live log of each agent's reasoning, then concludes with a **complete Publishing Brief** containing:

- ✅ Recommended posting time (day, time, timezone)
- ✅ Final copy-paste–ready LinkedIn post
- ✅ Tiered hashtag strategy
- ✅ First-hour engagement playbook
- ✅ Performance prediction

---

## Configuration

| `.env` Variable | Default | Description |
|-----------------|---------|-------------|
| `OPENAI_API_KEY` | — | Required |
| `OPENAI_MODEL` | `gpt-4o` | Any OpenAI chat model |
| `SERPER_API_KEY` | — | Required for web search |

---

## Cost Estimate

A single full pipeline run uses roughly **8,000–15,000 tokens** across all 5 agents with `gpt-4o`. At standard pricing this is approximately **$0.05–$0.12 per run**.

---

## Project Structure

```
linkedin_content_manager/
├── main.py            # All agents, tasks, and crew config
├── requirements.txt   # Python dependencies
├── .env.example       # API key template
└── README.md          # This file
```