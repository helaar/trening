"""Pydantic models for coach role definitions."""

from pydantic import BaseModel
from typing import Dict


class CoachRole(BaseModel):
    """Model for a single coach role definition."""
    role: str
    goal: str
    backstory: str


class CoachConfig(BaseModel):
    """Model for the entire coaches configuration file."""
    coaches: Dict[str, CoachRole]
    
    def get_coach_by_name(self, name: str) -> CoachRole | None:
        """Get a specific coach definition by name."""
        return self.coaches.get(name)
    
    def extract_crewai_params(self, coach_name: str) -> Dict[str, str]:
        """Extract CrewAI agent parameters for a specific coach."""
        coach_role = self.get_coach_by_name(coach_name)
        if not coach_role:
            raise ValueError(f"Coach '{coach_name}' not found")
        
        return {
            "role": coach_role.role,
            "goal": coach_role.goal,
            "backstory": coach_role.backstory
        }
    
    def list_coaches(self) -> list[str]:
        """List all available coach names."""
        return list(self.coaches.keys())