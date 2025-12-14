"""Tool for reading workout records from local disk files."""

from datetime import datetime
from pathlib import Path
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from crew.loaders import WorkoutsLoader
from crew.config import config

class FindWorkoutArgs(BaseModel):
    date: str = Field(description="Date in YYYY-MM-DD format to search for workout files")
    athlete: str = Field(description="Unique identifier of the athlete")

class WorkoutFileListerTool(BaseTool):
    """Tool for listing workout files for a specific date without reading their content."""
    args_schema: Type[BaseModel] = FindWorkoutArgs
    name: str = "workout_file_lister"
    description: str = "List all workout files for a specific date. Files are expected to be prefixed with the date (YYYY-MM-DD format)."
    
    def _run(self, athlete: str, date: str) -> str:
        """
        List all workout files for a specific date.
        
        Args:
            athlete: Unique identifier of the athlete
            date: Date in YYYY-MM-DD format to search for workout files
            
        Returns:
            String containing the list of files or error message
        """
        try:
            # Validate date format
            datetime.fromisoformat(date)
        except ValueError:
            return f"Error: Invalid date format '{date}'. Expected YYYY-MM-DD format."
        
        analyses_dir = config.get_athlete_analyses_dir(athlete)
        loader = WorkoutsLoader(analyses_dir)
        files = loader.get_workout_files(athlete=athlete, workout_date=datetime.fromisoformat(date).date())
        return "\n".join([str(f.name) for f in files]) if files else "No workout files found for the given date."
    

class DailyWorkoutReaderTool(BaseTool):
    """Tool for reading workout records from local disk files prefixed with dates."""
    
    name: str = "daily_workout_reader"
    description: str = "Read workout details for a given date and athlete."
    args_schema: Type[BaseModel] = FindWorkoutArgs
    
    def _run(self, date: str, athlete: str) -> str:
        """
        Read workout file(s) for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format to search for workout files
            athlete: Unique identifier of the athlete
            
        Returns:
            String containing the workout data or error message
        """
        try:
            # Validate date format
            datetime.fromisoformat(date)
        except ValueError:
            return f"Error: Invalid date format '{date}'. Expected YYYY-MM-DD format."
        
        analyses_dir = config.get_athlete_analyses_dir(athlete)
        loader = WorkoutsLoader(analyses_dir)
        workouts = loader.read_workouts(athlete=athlete, end_date=datetime.fromisoformat(date).date(), days_history=1)
        
        return f"# Workouts for {athlete} on {date}\n\n" + "\n\n".join(workouts)


