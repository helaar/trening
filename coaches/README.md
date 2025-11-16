# Agentic AI Component

A simple agentic AI component using CrewAI framework with researcher and writer agents.

## Installation

```bash
cd agentic
poetry install
```

## Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API key. CrewAI supports multiple LLM providers:

**OpenAI (default):**
```
OPENAI_API_KEY=your_openai_api_key
```

**Other LLM Providers:**
- **Anthropic Claude:** Set `ANTHROPIC_API_KEY` and use `llm="claude-3-sonnet-20240229"`
- **Google Gemini:** Set `GOOGLE_API_KEY` and use `llm="gemini-pro"`
- **Ollama (local):** Install Ollama locally and use `llm="ollama/llama2"`
- **Any LangChain LLM:** Pass a LangChain LLM object directly to the `llm` parameter

To use a different model, modify the `model_name` in [`config.py`](src/agentic/config.py) or pass a custom LLM object to the agents.

## Usage

Run the main script with a research topic:

```bash
poetry run python src/main.py "artificial intelligence trends"
```

## Project Structure

- `src/agentic/config.py` - Pydantic settings configuration
- `src/agentic/agents.py` - Agent definitions (researcher, writer)
- `src/agentic/tasks.py` - Task definitions
- `src/agentic/crew.py` - Main crew orchestration
- `src/main.py` - Entry point script

## Example

```python
from agentic import AgenticCrew

crew = AgenticCrew()
result = crew.run_research_and_write("machine learning in healthcare")
print(result)