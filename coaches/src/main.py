"""Main entry point for the agentic AI component."""

import argparse
from datetime import datetime
from pathlib import Path
from tasks import TaskLoader
from tools.athlete_reader import create_athlete_reader_tool
from crewai import Crew
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from roles.config import config
from roles.loader import CoachLoader
from tools.workout_reader import create_workout_reader_tool


def main():
    """Main function to run the agentic AI system."""
    parser = argparse.ArgumentParser(description="Daily workout analysis using AI coach agents")
    
    parser.add_argument("--coaches-config", "-c",
                        required=True,
                        help="Path to the coaches YAML configuration file")
    
    parser.add_argument("--date", "-d",
                        default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date for workout analysis in YYYY-MM-DD format (default: today)")
    
    parser.add_argument("--workout-dir", "-w",
                        default="workout_data",
                        help="Directory containing workout files (default: workout_data)")
    
    parser.add_argument("--coach",
                        default="daily_analysis_coach",
                        help="Coach name to use for analysis (default: daily_analysis_coach)")

    parser.add_argument("--athletes-file", "-a",
                        default="./athletes.yaml",
                        help="Athletes database file. Defaults to ./athletes.yaml")

    parser.add_argument("--tasks-file", "-t",
                        default="./tasks.yaml",
                        help="Tasks YAML file. Defaults to ./tasks.yaml")


    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Error: Invalid date format '{args.date}'. Expected YYYY-MM-DD format.")
        return 1
    
    try:
        # Initialize configuration and loader
        
        loader = CoachLoader(config)
        tasks = TaskLoader(Path(args.tasks_file))
        
        # Create a workout reader tool configured with the specified directory
        workout_tool = create_workout_reader_tool(args.workout_dir)
        athlete_loader = create_athlete_reader_tool(args.athletes_file)
        
        # Create the specified coach agent with tools
        analyzer = loader.load_agent_from_file(args.coaches_config, "performance_analysis_assistant", [workout_tool])
        head_coach = loader.load_agent_from_file(args.coaches_config, "head_coach", []) 
        translator = loader.load_agent_from_file(args.coaches_config, "translator")
    
        print(f"Workout directory: {args.workout_dir}")
        athlete = "Helge"
        # Create a task for the agent
        analysis_task = tasks.create_task("dayly_analysis_task", agent=analyzer)
        feedback_task = tasks.create_task("daily_feedback_task", agent=head_coach,context=[analysis_task])
        translate_task = tasks.create_task("translate_task", agent=translator, context=[feedback_task])
        
        
        if not analysis_task or not feedback_task or not translate_task:   
            raise ValueError("Tasks not found in tasks file.")
        
        athlete_knowledge = StringKnowledgeSource(content=athlete_loader._run(athlete_id=athlete))

        # Create a crew with the agent and task
        crew = Crew(            
            agents=[analyzer, head_coach, translator],
            tasks=[analysis_task, feedback_task, translate_task],
            knowledge_sources=[athlete_knowledge],
            verbose=True
        )
        
        # Execute the analysis
        print(f"\nStarting workout analysis for {athlete} on {args.date}...")
        result = crew.kickoff(
            inputs={
                "athlete":athlete, 
                "date": args.date,
                "exchange_dir":args.workout_dir
            })
        
        print("\n" + "="*50)
        print("WORKOUT ANALYSIS REPORT:")
        print("="*50)
        print(result)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())