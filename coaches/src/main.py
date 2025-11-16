"""Main entry point for the agentic AI component."""

import argparse
from datetime import datetime
from crewai import Task, Crew
from roles.config import Config
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
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Error: Invalid date format '{args.date}'. Expected YYYY-MM-DD format.")
        return 1
    
    try:
        # Initialize configuration and loader
        config = Config()
        loader = CoachLoader(config)
        
        # Create a workout reader tool configured with the specified directory
        workout_tool = create_workout_reader_tool(args.workout_dir)
        
        # Create the specified coach agent with tools
        agent = loader.load_agent_from_file(args.coaches_config, args.coach, [workout_tool])
        
        print(f"Created {args.coach} agent for analysis date: {args.date}")
        print(f"Workout directory: {args.workout_dir}")
        
        # Create a task for the agent
        analysis_task = Task(
            description=f"Analyze workout data for {args.date} from directory {args.workout_dir}. Use the workout_file_reader tool to retrieve the workout records and provide a comprehensive analysis report.",
            agent=agent,
            expected_output="A detailed performance analysis report including session overview, quantitative summary, qualitative assessment, progress indicators, risk flags, and coach recommendations.",
        )
        
        # Create a crew with the agent and task
        crew = Crew(
            agents=[agent],
            tasks=[analysis_task],
            verbose=True
        )
        
        # Execute the analysis
        print(f"\nStarting workout analysis for {args.date}...")
        result = crew.kickoff()
        
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