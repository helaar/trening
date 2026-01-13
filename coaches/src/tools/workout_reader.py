"""Tool for reading workout records from local disk files."""

from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Type, Literal
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from crew.loaders import WorkoutsLoader, SelfAssessmentLoader
from crew.config import config

class FindWorkoutArgs(BaseModel):
    start_date: str = Field(description="Start date in YYYY-MM-DD format to search for workout files")
    end_date: str = Field(description="End date in YYYY-MM-DD format to search for workout files (inclusive)")
    athlete: str = Field(description="Unique identifier of the athlete")

class WorkoutFileListerTool(BaseTool):
    """Tool for listing workout files for a date range without reading their content."""
    args_schema: Type[BaseModel] = FindWorkoutArgs
    name: str = "workout_file_lister"
    description: str = "List all workout files for a date range. Files are expected to be prefixed with the date (YYYY-MM-DD format). Date range is inclusive on both ends."
    file_type: Literal["json", "md"] = "md"
    
    def _run(self, athlete: str, start_date: str, end_date: str) -> str:
        """
        List all workout files for a date range.
        
        Args:
            athlete: Unique identifier of the athlete
            start_date: Start date in YYYY-MM-DD format to search for workout files
            end_date: End date in YYYY-MM-DD format to search for workout files (inclusive)
            
        Returns:
            String containing the list of files or error message
        """
        try:
            # Validate date formats
            start_dt = datetime.fromisoformat(start_date).date()
            end_dt = datetime.fromisoformat(end_date).date()
        except ValueError as e:
            return f"Error: Invalid date format. Expected YYYY-MM-DD format. {str(e)}"
        
        if start_dt > end_dt:
            return f"Error: Start date ({start_date}) cannot be after end date ({end_date})."
        
        analyses_dir = config.get_athlete_analyses_dir(athlete)
        loader = WorkoutsLoader(analyses_dir, file_type=self.file_type)
        
        all_files = []
        current_date = start_dt
        while current_date <= end_dt:
            files = loader.get_workout_files(athlete=athlete, workout_date=current_date)
            if files:
                all_files.extend([f"{current_date.isoformat()}: {f.name}" for f in files])
            current_date += timedelta(days=1)
        
        return "\n".join(all_files) if all_files else f"No workout files found for the date range {start_date} to {end_date}."
    

class DailyWorkoutReaderTool(BaseTool):
    """Tool for reading workout records from local disk files for a date range."""
    
    name: str = "daily_workout_reader"
    description: str = "Read workout details for a date range and athlete. Date range is inclusive on both ends."
    args_schema: Type[BaseModel] = FindWorkoutArgs
    file_type: Literal["json", "md"] = "md"
    
    def _run(self, start_date: str, end_date: str, athlete: str) -> str:
        """
        Read workout file(s) for a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format to search for workout files
            end_date: End date in YYYY-MM-DD format to search for workout files (inclusive)
            athlete: Unique identifier of the athlete
            
        Returns:
            String containing the workout data or error message
        """
        try:
            # Validate date formats
            start_dt = datetime.fromisoformat(start_date).date()
            end_dt = datetime.fromisoformat(end_date).date()
        except ValueError as e:
            return f"Error: Invalid date format. Expected YYYY-MM-DD format. {str(e)}"
        
        if start_dt > end_dt:
            return f"Error: Start date ({start_date}) cannot be after end date ({end_date})."
        
        try:
            analyses_dir = config.get_athlete_analyses_dir(athlete)
            loader = WorkoutsLoader(analyses_dir, file_type=self.file_type)
            
            all_workouts = []
            current_date = start_dt
            while current_date <= end_dt:
                workouts = loader.read_workouts(athlete=athlete, end_date=current_date, days_history=1)
                if workouts:
                    all_workouts.append(f"# Workouts for {athlete} on {current_date.isoformat()}\n\n" + "\n\n".join(workouts))
                current_date += timedelta(days=1)
            
            if not all_workouts:
                return f"No workouts found for {athlete} in the date range {start_date} to {end_date}."
            
            return "\n\n---\n\n".join(all_workouts)
        except Exception as e:
            print(f"Error reading workout data: {str(e)}")  # TODO: Temporary debug print
            return f"Error reading workout data for {athlete} between {start_date} and {end_date}: {str(e)}"


class SelfAssessmentFileListerTool(BaseTool):
    """Tool for listing self-assessment files for a date range without reading their content."""
    args_schema: Type[BaseModel] = FindWorkoutArgs
    name: str = "self_assessment_file_lister"
    description: str = "List all self-assessment files for a date range. Files are expected to be prefixed with the date (YYYY-MM-DD format). Date range is inclusive on both ends."
    file_type: Literal["json", "md"] = "md"
    
    def _run(self, athlete: str, start_date: str, end_date: str) -> str:
        """
        List all self-assessment files for a date range.
        
        Args:
            athlete: Unique identifier of the athlete
            start_date: Start date in YYYY-MM-DD format to search for self-assessment files
            end_date: End date in YYYY-MM-DD format to search for self-assessment files (inclusive)
            
        Returns:
            String containing the list of files or error message
        """
        try:
            # Validate date formats
            start_dt = datetime.fromisoformat(start_date).date()
            end_dt = datetime.fromisoformat(end_date).date()
        except ValueError as e:
            return f"Error: Invalid date format. Expected YYYY-MM-DD format. {str(e)}"
        
        if start_dt > end_dt:
            return f"Error: Start date ({start_date}) cannot be after end date ({end_date})."
        
        self_assessments_dir = config.get_athlete_self_assessments_dir(athlete)
        loader = SelfAssessmentLoader(self_assessments_dir, file_type=self.file_type)
        
        all_files = []
        current_date = start_dt
        while current_date <= end_dt:
            files = loader.get_self_assessment_files(athlete=athlete, assessment_date=current_date)
            if files:
                all_files.extend([f"{current_date.isoformat()}: {f.name}" for f in files])
            current_date += timedelta(days=1)
        
        return "\n".join(all_files) if all_files else f"No self-assessment files found for the date range {start_date} to {end_date}."


class DailySelfAssessmentReaderTool(BaseTool):
    """Tool for reading self-assessment records from local disk files for a date range."""
    
    name: str = "daily_self_assessment_reader"
    description: str = "Read self-assessment details for a date range and athlete. Date range is inclusive on both ends."
    args_schema: Type[BaseModel] = FindWorkoutArgs
    file_type: Literal["json", "md"] = "md"
    
    def _run(self, start_date: str, end_date: str, athlete: str) -> str:
        """
        Read self-assessment file(s) for a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format to search for self-assessment files
            end_date: End date in YYYY-MM-DD format to search for self-assessment files (inclusive)
            athlete: Unique identifier of the athlete
            
        Returns:
            String containing the self-assessment data or error message
        """
        try:
            # Validate date formats
            start_dt = datetime.fromisoformat(start_date).date()
            end_dt = datetime.fromisoformat(end_date).date()
        except ValueError as e:
            return f"Error: Invalid date format. Expected YYYY-MM-DD format. {str(e)}"
        
        if start_dt > end_dt:
            return f"Error: Start date ({start_date}) cannot be after end date ({end_date})."
        
        try:
            self_assessments_dir = config.get_athlete_self_assessments_dir(athlete)
            loader = SelfAssessmentLoader(self_assessments_dir, file_type=self.file_type)
            
            all_assessments = []
            current_date = start_dt
            while current_date <= end_dt:
                assessments = loader.read_self_assessments(athlete=athlete, end_date=current_date, days_history=1)
                if assessments:
                    all_assessments.append(f"# Self-assessments for {athlete} on {current_date.isoformat()}\n\n" + "\n\n".join(assessments))
                current_date += timedelta(days=1)
            
            if not all_assessments:
                return f"No self-assessments found for {athlete} in the date range {start_date} to {end_date}."
            
            return "\n\n---\n\n".join(all_assessments)
        except Exception as e:
            print(f"Error reading self-assessment data: {str(e)}")  # TODO: Temporary debug print
            return f"Error reading self-assessment data for {athlete} between {start_date} and {end_date}: {str(e)}"


