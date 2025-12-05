"""Tool for reading workout records from local disk files."""

import json
from datetime import datetime
from pathlib import Path
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class WorkoutFileListerTool(BaseTool):
    """Tool for listing workout files for a specific date without reading their content."""
    
    name: str = "workout_file_lister"
    description: str = "List all workout files for a specific date. Files are expected to be prefixed with the date (YYYY-MM-DD format)."
    workout_files_directory: str = Field(default="workout_data", description="Directory path where workout files are stored")
    
    def _run(self, date: str) -> str:
        """
        List all workout files for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format to search for workout files
            
        Returns:
            String containing the list of files or error message
        """
        try:
            # Validate date format
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return f"Error: Invalid date format '{date}'. Expected YYYY-MM-DD format."
        
        workout_dir = Path(self.workout_files_directory)
        
        if not workout_dir.exists():
            return f"Error: Workout directory '{workout_dir}' does not exist."
        
        # Find files that start with the given date
        matching_files = []
        for file_path in workout_dir.iterdir():
            if file_path.is_file() and file_path.name.startswith(date):
                matching_files.append(file_path.name)
        
        if not matching_files:
            return f"No workout files found for date {date} in directory '{workout_dir}'."
        
        # Sort files for consistent output
        matching_files.sort()
        
        # Format the result
        result = f"Workout files for {date}:\n"
        result += f"Found {len(matching_files)} file(s) in '{workout_dir}':\n\n"
        
        for filename in matching_files:
            result += f"- {filename}\n"
        
        return result
class FindWorkoutArgs(BaseModel):
    date: str = Field(description="Date in YYYY-MM-DD format to search for workout files")
    athlete: str = Field(description="Unique identifier of the athlete")

class WorkoutFileReaderTool(BaseTool):
    """Tool for reading workout records from local disk files prefixed with dates."""
    
    name: str = "workout_file_reader"
    description: str = "Read detailed workout records from local files. Files are expected to be prefixed with the date (YYYY-MM-DD format)."
    args_schema: Type[BaseModel] = FindWorkoutArgs
    workout_files_directory: str = Field(default="workout_data", description="Directory path where workout files are stored")
    
    def _run(self, date: str, athlete: str) -> str:
        """
        Read workout file(s) for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format to search for workout files
            
        Returns:
            String containing the workout data or error message
        """
        try:
            # Validate date format
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return f"Error: Invalid date format '{date}'. Expected YYYY-MM-DD format."
        
        workout_dir = Path(self.workout_files_directory)
        
        if not workout_dir.exists():
            return f"Error: Workout directory '{workout_dir}' does not exist."
        
        # Find files that start with the given date
        matching_files = []
        for file_path in workout_dir.iterdir():
            if file_path.is_file() and file_path.name.startswith(date):
                matching_files.append(file_path)
        
        if not matching_files:
            return f"No workout files found for date {date} in directory '{workout_dir}'."
        
        # Read and combine data from all matching files
        workout_data = []
        for file_path in matching_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Try to parse as JSON first, fall back to plain text
                    try:
                        data = json.load(f)
                        workout_data.append({
                            "filename": file_path.name,
                            "data": data,
                            "type": "json"
                        })
                    except json.JSONDecodeError:
                        # If not JSON, read as plain text
                        f.seek(0)  # Reset file pointer
                        content = f.read()
                        workout_data.append({
                            "filename": file_path.name,
                            "data": content,
                            "type": "text"
                        })
            except Exception as e:
                workout_data.append({
                    "filename": file_path.name,
                    "error": f"Failed to read file: {str(e)}",
                    "type": "error"
                })
        
        # Format the result for the agent
        result = f"Workout data for {date}:\n"
        result += f"Found {len(matching_files)} file(s)\n\n"
        
        for item in workout_data:
            result += f"=== File: {item['filename']} ===\n"
            if item['type'] == 'error':
                result += f"Error: {item['error']}\n"
            elif item['type'] == 'json':
                result += f"JSON Data:\n{json.dumps(item['data'], indent=2)}\n"
            else:  # text
                result += f"Text Content:\n{item['data']}\n"
            result += "\n"
        
        return result


def create_workout_lister_tool(workout_directory: str = "workout_data") -> WorkoutFileListerTool:
    """Factory function to create a workout file lister tool."""
    return WorkoutFileListerTool(workout_files_directory=workout_directory)


def create_workout_reader_tool(workout_directory: str = "workout_data") -> WorkoutFileReaderTool:
    """Factory function to create a workout file reader tool."""
    return WorkoutFileReaderTool(workout_files_directory=workout_directory)