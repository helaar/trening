"""Tool for loading the freshest long time analysis from exchange/load folder."""

from pathlib import Path
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class LongTimeAnalysisArgs(BaseModel):
    athlete: str = Field(description="Unique identifier of the athlete")


class LongTimeAnalysisLoaderTool(BaseTool):
    """Tool for loading the most recent long time analysis file from exchange/load directory."""
    
    args_schema: Type[BaseModel] = LongTimeAnalysisArgs
    name: str = "long_time_analysis_loader"
    description: str = (
        "Load the freshest (most recently created) long time analysis file from the exchange/load folder. "
        "Returns the content of the most recent analysis file."
    )
    load_dir: Path = Field(
        default=Path("exchange/load"), 
        description="Directory path where long time analysis files are stored"
    )
    
    def _run(self, athlete: str) -> str:
        """
        Load the freshest long time analysis file from exchange/load directory.
        
        Args:
            athlete: Unique identifier of the athlete (parameter included for consistency)
            
        Returns:
            String containing the analysis content or error message
        """
        
        if not self.load_dir.exists():
            return f"Error: Exchange load directory '{self.load_dir}' does not exist."
        
        if not self.load_dir.is_dir():
            return f"Error: '{self.load_dir}' is not a directory."
        
        # Find all files in the directory
        analysis_files = [
            f for f in self.load_dir.iterdir() 
            if f.is_file() and not f.name.startswith('.')
        ]
        
        if not analysis_files:
            return f"No analysis files found in '{self.load_dir}'."
        
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