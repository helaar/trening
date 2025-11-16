"""Main entry point for the agentic AI component."""

import sys
from agentic import AgenticCrew


def main():
    """Main function to run the agentic AI system."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <research_topic>")
        sys.exit(1)
    
    topic = " ".join(sys.argv[1:])
    print(f"Starting research on: {topic}")
    
    try:
        crew = AgenticCrew()
        result = crew.run_research_and_write(topic)
        print("\n" + "="*50)
        print("FINAL RESULT:")
        print("="*50)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()