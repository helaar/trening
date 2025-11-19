""" Tool to read a athlete's profile from a file """

from pathlib import Path

import yaml

from crewai.tools import BaseTool

class AthleteLookupTool(BaseTool):
    """Tool that retrieves a single athlete's information from a YAML file."""
    name: str = "athlete_lookup"
    description: str = (
        "Use this tool to retrieve information about an athlete by unique athlete ID. "
        "Returns data in a textual, JSON-like form."
    )
    yaml_path: Path
    _cache: dict[str,object] | None = None

    def _run(self, athlete_id: str) -> str:
        data = self._load_yaml()
        athlete = data[athlete_id]
        if not athlete_id:
            return f"Athlete {athlete_id} not found."

        formatted = "\n".join(f"{k}: {v}" for k, v in athlete.items())
        return formatted

    def _load_yaml(self) -> dict[str, object]:
        if self._cache is not None:
            return self._cache

        with self.yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._cache = data
        return data

def create_athlete_reader_tool(athlete_file : str | Path) -> AthleteLookupTool:
    return AthleteLookupTool(yaml_path=Path(athlete_file))