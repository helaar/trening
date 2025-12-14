"""Tool for loading the freshest long time analysis from athlete's load folder."""

from pathlib import Path
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from crew.config import config


class LongTimeAnalysisArgs(BaseModel):
    athlete: str = Field(description="Unique identifier of the athlete")


class LongTimeAnalysisLoaderTool(BaseTool):
    """Tool for loading the most recent long time analysis file from athlete's load directory."""
    
    args_schema: Type[BaseModel] = LongTimeAnalysisArgs
    name: str = "long_time_analysis_loader"
    description: str = (
        "Load the freshest (most recently created) long time analysis file from the athlete's load folder. "
        "Returns the content of the most recent analysis file."
    )
    
    def _run(self, athlete: str) -> str:
        """
        Load the freshest long time analysis file from athlete's load directory.
        
        Args:
            athlete: Unique identifier of the athlete
            
        Returns:
            String containing the analysis content or error message
        """
        
        load_dir = config.get_athlete_load_dir(athlete)
        
        if not load_dir.exists():
            return f"Error: Athlete load directory '{load_dir}' does not exist."
        
        if not load_dir.is_dir():
            return f"Error: '{load_dir}' is not a directory."
        
        # Find all files in the directory
        analysis_files = [
            f for f in load_dir.iterdir()
            if f.is_file() and not f.name.startswith('.')
        ]
        
        if not analysis_files:
            return f"No analysis files found in '{load_dir}'."
        
        # Find the most recent file (by modification time)
        freshest_file = max(analysis_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with freshest_file.open("r", encoding="utf-8") as f:
                content = f.read().strip()
                
            if not content:
                return f"The freshest analysis file '{freshest_file.name}' is empty."
            
            return f"# Freshest Long Time Analysis: {freshest_file.name}\n\n{content}"
            
        except Exception as e:
            return f"Error reading analysis file '{freshest_file.name}': {str(e)}"