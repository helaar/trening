#!/usr/bin/env python3
"""
Shared settings and formatting utilities for workout analysis.
"""
import yaml
from pathlib import Path
from pydantic import BaseModel, Field


class ApplicationSettings(BaseModel):
    """Application settings model."""
    output_dir: str = Field(default=".", alias="output-dir")
    
    def get_output_path(self) -> Path:
        """Get the configured output directory as a Path object."""
        output_path = Path(self.output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


def load_settings(path: str) -> dict[str, object]:
    """Load settings from YAML file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


__all__ = ['ApplicationSettings', 'load_settings']