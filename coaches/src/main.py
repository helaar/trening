"""Main entry point for the agentic AI component."""

import argparse
from datetime import datetime
from pathlib import Path


from tools.athlete_reader import AthleteLookupTool, AthletePlanTool
from crewai import Crew
from crew.config import config
from crew.loaders import CoachLoader, PlansLoader, TaskLoader, KnowledgeLoader

from tools.history_lister import FeedbackListerTool
from tools.workout_reader import DailyWorkoutReaderTool, WorkoutFileListerTool

def daily_analysis(athlete: str, date: str, output_dir: str) -> None:
    """Function to perform daily workout analysis using AI coach agents."""

    # Initialize configuration and loader
    analyst_memory = False # SimpleFileStorage.memory(Path(config.exchange_dir) / f"{athlete}_analyst.json")
    main_coach_memory = False #SimpleFileStorage.memory(Path(config.exchange_dir) / f"{athlete}_main_coach.json")
    coach_loader = CoachLoader(config)
    task_loader = TaskLoader(config)
    knowledge_loader = KnowledgeLoader(config)
    
    # Create a workout reader tool configured with the specified directory
    workout_tool = DailyWorkoutReaderTool(workout_files_directory=config.workouts)
    workout_lister_tool = WorkoutFileListerTool(workout_files_directory=config.workouts)
    athlete_reader = AthleteLookupTool(yaml_path=Path(config.athletes))
    plans_reader_tool = AthletePlanTool(loader=PlansLoader(config))
    
    #athlete_knowledge = StringKnowledgeSource(content=athlete_reader._run(athlete=athlete))
    common_knowledge = knowledge_loader.get_knowledge()
    
    # Create the specified coach agent with toos
    analyzer = coach_loader.create_coach_agent("performance_analysis_assistant", memory=analyst_memory, tools=[athlete_reader, workout_tool, plans_reader_tool])
    head_coach = coach_loader.create_coach_agent("head_coach", memory=main_coach_memory, reasoning=True, tools=[athlete_reader, workout_lister_tool,plans_reader_tool]) 
    
    # Create a task for the agent
    analysis_task = task_loader.create_task("dayly_analysis_task", agent=analyzer)
    feedback_task = task_loader.create_task("daily_feedback_task", agent=head_coach,context=[analysis_task])
    
    
    
    # Create a crew with the agent and task
    crew = Crew(            
        agents=[analyzer, head_coach],
        tasks=[analysis_task, feedback_task],
        knowledge_sources=[common_knowledge],
        verbose=True,
        
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
        

def long_term_analysis(athlete: str, date: str, output_dir: str, days_history: int=14, days_ahead: int=7) -> None:
    """Function to perform long-term workout analysis using AI coach agents."""


    coach_loader = CoachLoader(config)
    task_loader = TaskLoader(config)
    knowledge_loader = KnowledgeLoader(config)
    plans_reader_tool = AthletePlanTool(loader=PlansLoader(config))
    workout_tool = DailyWorkoutReaderTool(workout_files_directory=config.workouts) #todo: create long-term workout reader tool
    list_analysis_tool = FeedbackListerTool(feedback_dir=Path(config.exchange_dir) / "daily")
    
    athlete_reader = AthleteLookupTool(yaml_path=Path(config.athletes))
    analyzer = coach_loader.create_coach_agent("performance_analysis_assistant", memory=False, reasoning=True, tools=[athlete_reader, workout_tool, plans_reader_tool])
    head_coach = coach_loader.create_coach_agent("head_coach", memory=False, reasoning=True, tools=[athlete_reader, list_analysis_tool,plans_reader_tool]) 
    analysis_task = task_loader.create_task("long_term_analysis_task", agent=analyzer)
    feedback_task = task_loader.create_task("long_term_feedback_task", agent=head_coach, context=[analysis_task])
    common_knowledge = knowledge_loader.get_knowledge()

    
    crew = Crew(            
        agents=[analyzer,head_coach],
        tasks=[analysis_task, feedback_task],
        knowledge_sources=[common_knowledge],
        verbose=True,
    )

    print(f"Performing long-term analysis for {athlete} up to {date}...")
    result = crew.kickoff(
        inputs={
            "athlete":athlete, 
            "date": date,
            "output_dir":output_dir,
            "days_history": days_history,
            "days_ahead": days_ahead
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
                        choices=["daily","weekly", "test"],
                        help="Perform analysis for the given period ending on the specified date (default: P1D for one day)")
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.fromisoformat(args.date)
    except ValueError:
        print(f"Error: Invalid date format '{args.date}'. Expected YYYY-MM-DD format.")
        return 1
    
    try:
        match args.period:
            case "daily":
                daily_analysis(athlete="Helge", date=args.date, output_dir=config.exchange_dir)
            case "weekly":
                long_term_analysis(athlete="Helge", date=args.date, output_dir=config.exchange_dir, days_history=14, days_ahead=7)
            case "test" :
                list_analysis_tool = DailyWorkoutReaderTool(workout_files_directory=config.workouts)
                result = list_analysis_tool._run(athlete="Helge", date=args.date)
                print(f"Test result: {result}")
            
        
    except Exception as e:
        print(f"Error: {e} {str(e.args)}")
        return 1


if __name__ == "__main__":
    exit(main())