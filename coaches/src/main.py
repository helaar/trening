"""Main entry point for the agentic AI component."""

import argparse
from datetime import datetime
from pathlib import Path


from tools.athlete_reader import AthleteLookupTool, AthletePlanTool
from crewai import Crew
from crew.config import config
from crew.loaders import CoachLoader, PlansLoader, TaskLoader, KnowledgeLoader

from tools.history_lister import FeedbackListerTool
from tools.workout_reader import DailyWorkoutReaderTool, WorkoutFileListerTool,  DailySelfAssessmentReaderTool
from tools.long_time_analysis_loader import LongTimeAnalysisLoaderTool

def daily_analysis(athlete: str, date: str, output_dir: str) -> None:
    """Function to perform daily workout analysis using AI coach agents."""

    # Initialize configuration and loader
    analyst_memory = False 
    main_coach_memory = False 
    coach_loader = CoachLoader(config)
    task_loader = TaskLoader(config)
    knowledge_loader = KnowledgeLoader(config)
    
    # Create a workout reader tool configured with the athlete's directory
    workout_tool = DailyWorkoutReaderTool(file_type="json")
    workout_lister_tool = WorkoutFileListerTool(file_type="json")
    self_assessments_tool = DailySelfAssessmentReaderTool()
    athlete_reader = AthleteLookupTool(yaml_path=Path(config.athlete_settings))
    plans_reader_tool = AthletePlanTool(loader=PlansLoader(config))
    
    common_knowledge = knowledge_loader.get_knowledge()
    
    # Create the specified coach agent with tools
    analyzer = coach_loader.create_coach_agent("performance_analysis_assistant", memory=False, tools=[athlete_reader, workout_tool, self_assessments_tool, plans_reader_tool],  knowledge=[common_knowledge])
    #qa_inspector = coach_loader.create_coach_agent("qa_agent", memory=False, tools=[athlete_reader, workout_tool, self_assessments_tool, plans_reader_tool], knowledge=[common_knowledge])
    head_coach = coach_loader.create_coach_agent("head_coach", memory=main_coach_memory, tools=[athlete_reader, workout_lister_tool, self_assessments_tool, plans_reader_tool], knowledge=[common_knowledge])
    translator = coach_loader.create_coach_agent("translator", memory=False, tools=[])
    
    # Create a task for the agent
    analysis_task = task_loader.create_task("dayly_analysis_task", agent=analyzer)
    #validation_task = task_loader.create_task("validate_analysis_task", agent=qa_inspector, context=[analysis_task])
    feedback_task = task_loader.create_task("daily_feedback_task", agent=head_coach, context=[analysis_task])
    translate_task = task_loader.create_task("translate_task", agent=translator, context=[feedback_task])
    
    
    
    # Create a crew with the agent and task
    crew = Crew(
        agents=[analyzer,  head_coach, translator],
        tasks=[analysis_task, feedback_task, translate_task],
        verbose=True,
        
    )
    
    # Execute the analysis
    print(f"\nStarting daily workout analysis for {athlete} on {date}...")
    # Calculate weekday for date context
    date_obj = datetime.fromisoformat(date)
    weekday = date_obj.strftime("%A")
    
    result = crew.kickoff(
        inputs={
            "athlete": athlete,
            "date": date,
            "weekday": weekday,
            "output_dir": str(config.get_athlete_daily_dir(athlete)),
            "preferred_language": "Norwegian"
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
    workout_tool = DailyWorkoutReaderTool() #todo: create long-term workout reader tool
    list_analysis_tool = FeedbackListerTool()
    
    athlete_reader = AthleteLookupTool(yaml_path=Path(config.athlete_settings))
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
    # Calculate weekday for date context
    date_obj = datetime.fromisoformat(date)
    weekday = date_obj.strftime("%A")
    
    result = crew.kickoff(
        inputs={
            "athlete":athlete,
            "date": date,
            "weekday": weekday,
            "output_dir": str(config.get_athlete_long_term_dir(athlete)),
            "days_history": days_history,
            "days_ahead": days_ahead
        })

    print("\n" + "="*50)
    print("WORKOUT ANALYSIS REPORT:")
    print("="*50)
    print(result)

def plan_suggestion(athlete: str, date: str, output_dir: str, days_ahead: int = 7, threshold: int = 3, lookback_days: int = 14) -> None:
    """Function to check and suggest training plans when insufficient workouts are planned."""

    self_assessments_tool = DailySelfAssessmentReaderTool()
    coach_loader = CoachLoader(config)
    task_loader = TaskLoader(config)
    knowledge_loader = KnowledgeLoader(config)
    plans_reader_tool = AthletePlanTool(loader=PlansLoader(config))
    athlete_reader = AthleteLookupTool(yaml_path=Path(config.athlete_settings))
    long_analysis_tool = LongTimeAnalysisLoaderTool()
    
    # Create specialized coach for plan suggestion with reasoning capabilities
    plan_coach = coach_loader.create_coach_agent(
        "head_coach",
        memory=False,
        reasoning=True,
        tools=[athlete_reader, plans_reader_tool, self_assessments_tool,long_analysis_tool]
    )
    
    plan_task = task_loader.create_task("plan_suggestion_task", agent=plan_coach)
    common_knowledge = knowledge_loader.get_knowledge()

    crew = Crew(
        agents=[plan_coach],
        tasks=[plan_task],
        knowledge_sources=[common_knowledge],
        verbose=True,
    )

    print(f"Checking training plan sufficiency for {athlete} from {date}...")
    # Calculate weekday for date context
    date_obj = datetime.fromisoformat(date)
    weekday = date_obj.strftime("%A")
    
    result = crew.kickoff(
        inputs={
            "athlete": athlete,
            "date": date,
            "weekday": weekday,
            "output_dir": str(config.get_athlete_planning_dir(athlete)),
            "days_ahead": days_ahead,
            "threshold": threshold,
            "lookback_days": lookback_days
        })

    print("\n" + "="*50)
    print("TRAINING PLAN SUGGESTION REPORT:")
    print("="*50)
    print(result)

def main():
    """Main function to run the agentic AI system."""
    parser = argparse.ArgumentParser(description="Daily workout analysis using AI coach agents")
    
    
    parser.add_argument("--date", "-d",
                        default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date for workout analysis in YYYY-MM-DD format (default: today)")
    
    parser.add_argument("--processing", "-p",
                        default="daily",
                        choices=["daily","weekly", "plan","test"],
                        help="Processing type: 'daily' for daily analysis, 'weekly' for long-term analysis, 'plan' for plan suggestion, 'test' for testing tools (default: daily)")
    
    parser.add_argument("--mode", "-m",
                        default="dev",
                        choices=["dev", "prod"],
                        help="LLM model mode: 'dev' for development (cheaper), 'prod' for production (more accurate) (default: dev)")
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.fromisoformat(args.date)
    except ValueError:
        print(f"Error: Invalid date format '{args.date}'. Expected YYYY-MM-DD format.")
        return 1
    
    try:
        # Set the LLM mode on config
        config.set_mode(args.mode)
        print(f"Using LLM model: {config.get_model()} (mode: {args.mode})")
        
        match args.processing:
            case "daily":
                daily_analysis(athlete="helge", date=args.date, output_dir=str(config.get_athlete_daily_dir("helge")))
            case "weekly":
                long_term_analysis(athlete="helge", date=args.date, output_dir=str(config.get_athlete_long_term_dir("helge")), days_history=14, days_ahead=7)
            case "plan":
                plan_suggestion(athlete="helge", date=args.date, output_dir=str(config.get_athlete_planning_dir("helge")), days_ahead=7, threshold=3, lookback_days=14)
            case "test" :

                
                print("\nTesting Daily Self-Assessment Reader Tool:")
                reader_tool = DailyWorkoutReaderTool()
                content_result = reader_tool._run(athlete="helge", start_date="2026-01-11", end_date="2026-01-11")
                print(f"Content result: {content_result}")
            
        
    except Exception as e:
        print(f"Error: {e} {str(e.args)}")
        return 1


if __name__ == "__main__":
    exit(main())