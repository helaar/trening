"""Main entry point for the agentic AI component."""

import argparse
from datetime import datetime
from pathlib import Path

from tools.athlete_reader import create_athlete_reader_tool
from crewai import Crew
from crew.config import config
from crew.loaders import CoachLoader, TaskLoader
from tools.workout_reader import create_workout_reader_tool

def daily_analysis(athlete: str, date: str, output_dir: str) -> None:
    """Function to perform daily workout analysis using AI coach agents."""

    # Initialize configuration and loader
    
    coach_loader = CoachLoader(config)
    task_loader = TaskLoader(config)
    
    # Create a workout reader tool configured with the specified directory
    workout_tool = create_workout_reader_tool(config.workouts)
    athlete_reader = create_athlete_reader_tool(config.athletes)
    
    # Create the specified coach agent with toos
    analyzer = coach_loader.create_coach_agent("performance_analysis_assistant", [workout_tool, athlete_reader])
    head_coach = coach_loader.create_coach_agent("head_coach", [athlete_reader]) 
    


    athlete = "Helge"
    # Create a task for the agent
    analysis_task = task_loader.create_task("dayly_analysis_task", agent=analyzer)
    feedback_task = task_loader.create_task("daily_feedback_task", agent=head_coach,context=[analysis_task])
    
    
    
    if not analysis_task or not feedback_task:   
        raise ValueError("Tasks not found in tasks file.")
    
    # Create a crew with the agent and task
    crew = Crew(            
        agents=[analyzer, head_coach],
        tasks=[analysis_task, feedback_task],
        verbose=True
    )
    
    # Execute the analysis
    print(f"\nStarting daily workout analysis for {athlete} on {date}...")
    result = crew.kickoff(
        inputs={
            "athlete":athlete, 
            "date": date,
            "output_dir":output_dir
        })
    
    print("\n" + "="*50)
    print("WORKOUT ANALYSIS REPORT:")
    print("="*50)
    print(result)
        

def main():
    """Main function to run the agentic AI system."""
    parser = argparse.ArgumentParser(description="Daily workout analysis using AI coach agents")
    
    
    parser.add_argument("--date", "-d",
                        default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date for workout analysis in YYYY-MM-DD format (default: today)")
    
    parser.add_argument("--period", "-p",
                        default="daily",
                        choices=["daily","weekly"],
                        help="Perform analysis for the given period ending on the specified date (default: P1D for one day)")
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Error: Invalid date format '{args.date}'. Expected YYYY-MM-DD format.")
        return 1
    
    try:
        match args.period:
            case "daily":
                daily_analysis(athlete="Helge", date=args.date, output_dir=config.exchange_dir)
        
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())