"""
Lists historical analysis files for an athlete.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Literal
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from crew.config import config


class ListParameters(BaseModel):
    athlete: str = Field(description="Unique identifier of the athlete")
    end_date: date = Field(description="The last date to consider for historical analysis")
    days_history: int = Field(description="Number of days to look back for historical analysis files")
    type: Literal["details", "feedback", "all"] = Field(description="Type of historical files to list")

class FeedbackListerTool(BaseTool):
    """Tool that lists historical analysis files for an athlete."""
    args_schema: type = ListParameters
    name: str = "feedback_history_lister"
    description: str = (
        "Use this tool to list historical analysis files for a given athlete. "
        "Returns a list of analysis summaries in a textual form."
    )

    def _run(self, athlete: str, end_date: date, days_history: int, type: Literal["details", "feedback", "all"] = "feedback") -> str:
        """
        Read all text files in the athlete's daily directory for the given period.
        Files are expected to be named starting with the date (e.g., YYYY-MM-DD_*.md).
        """

        feedback_dir = config.get_athlete_daily_dir(athlete) 
        
        # Calculate start date (inclusive range)
        start_date = end_date - timedelta(days=days_history - 1)
        
        # Collect all matching files
        feedback_files = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.isoformat()
            # Find all text files starting with this date
            match type:
                case "details" | "feedback":
                    pattern = f"{date_str}_{type}.md"
                case "all":
                    pattern = f"{date_str}_*.md"

            matching_files = list(feedback_dir.glob(pattern))
            
            for file_path in sorted(matching_files):
                feedback_files.append((current_date, file_path))
            
            current_date += timedelta(days=1)
        
        if not feedback_files:
            return f"No historical analysis files found for athlete {athlete} between {start_date} and {end_date}. Reading path: {feedback_dir}"
        
        # Read and concatenate file contents
        result_parts = []
        for file_date, file_path in feedback_files:
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        result_parts.append(f"=== {file_date} ({file_path.name}) ===\n{content}")
            except Exception as e:
                result_parts.append(f"=== {file_date} ({file_path.name}) ===\nError reading file: {e}")
        
        if not result_parts:
            return f"Found {len(feedback_files)} file(s) but all were empty for athlete {athlete} between {start_date} and {end_date}."
        
        return "\n\n".join(result_parts)