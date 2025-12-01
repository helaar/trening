from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.storage.interface import Storage

import json
import os
from typing import Any

from datetime import UTC, datetime
from pathlib import Path

class SimpleFileStorage(Storage):
    """
    Minimal CrewAI-compatible memory.
    Stores conversation history as a JSON file.
    """
    def __init__(self, path: Path):
        self.path:Path = path
        self._data: dict[str, list] = self._load()

    def _load(self) -> dict[str, list]:
        
        print(f"Loading memory from {self.path} exists={os.path.exists(self.path)}")
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        return {"memories": []}

    def save(self, value, metadata):
        print(f"SimpleFileStorage.save() called with value: {value[:100] if isinstance(value, str) else str(value)[:100]}...")
        self._data["memories"].append({
            "value": value,
            "metadata": metadata,
        })
        self._save()

    def search(self, query, limit=10, score_threshold=0.5):
        # Implement your search logic here
        return [m for m in self._data["memories"] if query.lower() in str(m["value"]).lower()]

    def reset(self):
        self._data = {"memories": []}
        self._save()

    def _save(self):
        print(f"Saving memory to {self.path}")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    @staticmethod
    def memory(storage_path: Path) -> LongTermMemory:
        """Create a simple file-based memory storage."""
        
        storage = SimpleFileStorage(storage_path)
        long_term_memory = LongTermMemory(storage=storage)
        
        return long_term_memory