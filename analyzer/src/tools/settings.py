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


def save_settings(path: str, settings: dict[str, object]) -> None:
    """
    Save settings to YAML file.
    
    Args:
        path: Path to the YAML file
        settings: Settings dictionary to save
    """
    settings_path = Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(settings_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(settings, fh, default_flow_style=False, sort_keys=False)


def get_commute_routes(athlete_settings: dict[str, object], athlete_id: str) -> dict[str, str]:
    """
    Get known commute routes for an athlete.
    
    Args:
        athlete_settings: Full athlete settings dictionary
        athlete_id: Athlete identifier
        
    Returns:
        Dictionary mapping commute route names to normalized polylines
    """
    athletes = athlete_settings.get("athletes", {})
    if not isinstance(athletes, dict):
        return {}
    
    athlete_data = athletes.get(athlete_id, {})
    if not isinstance(athlete_data, dict):
        return {}
    
    commutes = athlete_data.get("commute_routes", {})
    return commutes if isinstance(commutes, dict) else {}


def save_commute_route(
    athlete_settings_path: str,
    athlete_id: str,
    route_name: str,
    normalized_polyline: str
) -> None:
    """
    Save a new commute route to athlete settings.
    
    Args:
        athlete_settings_path: Path to athlete settings file
        athlete_id: Athlete identifier
        route_name: Name for the commute route
        normalized_polyline: Normalized polyline string for the route
    """
    # Load current settings
    settings_path = Path(athlete_settings_path)
    if settings_path.exists():
        settings = load_settings(str(settings_path))
    else:
        settings = {}
    
    # Ensure athletes structure exists
    if "athletes" not in settings:
        settings["athletes"] = {}
    
    athletes = settings["athletes"]
    if not isinstance(athletes, dict):
        athletes = {}
        settings["athletes"] = athletes
    
    # Ensure athlete exists
    if athlete_id not in athletes:
        athletes[athlete_id] = {}
    
    athlete_data = athletes[athlete_id]
    if not isinstance(athlete_data, dict):
        athlete_data = {}
        athletes[athlete_id] = athlete_data
    
    # Ensure commute_routes exists
    if "commute_routes" not in athlete_data:
        athlete_data["commute_routes"] = {}
    
    commute_routes = athlete_data["commute_routes"]
    if not isinstance(commute_routes, dict):
        commute_routes = {}
        athlete_data["commute_routes"] = commute_routes
    
    # Save the route
    commute_routes[route_name] = normalized_polyline
    
    # Save settings back to file
    save_settings(str(settings_path), settings)


__all__ = [
    'ApplicationSettings',
    'load_settings',
    'save_settings',
    'get_commute_routes',
    'save_commute_route'
]