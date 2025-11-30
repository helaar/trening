import json
import os


from datetime import UTC, datetime
from pathlib import Path
from langchain_core.memory import BaseMemory

class SimpleFileMemory(BaseMemory):
    """
    Minimal CrewAI-compatible memory.
    Stores conversation history as a JSON file.
    """
    def __init__(self, path: Path):
        self.path = path
        self._data = self._load()

    def _load(self):
        
        print(f"Loading memory from {self.path} exists={os.path.exists(self.path)}")
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        return {"history": []}

    def _save(self):
        print(f"Saving memory to {self.path}")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    @property
    def memory_variables(self) -> list[str]:
        """Define what memory variables this class provides."""
        return ["history"]

    def load_memory_variables(self, inputs: dict | None = None):
        print(f"Loading memory variables with inputs={inputs}")
        """Return the memory that will be injected into the agent's prompt."""
        return {"history": self._data["history"]}

    def save_context(self, inputs: dict, outputs: dict):
        """Save the current interaction."""
        print(f"Saving context. inputs={inputs} outputs={outputs}")
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "inputs": inputs,
            "outputs": outputs
        }
        self._data["history"].append(record)
        self._save()

    def clear(self):
        self._data = {"history": []}
        self._save()
