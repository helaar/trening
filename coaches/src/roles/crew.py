"""Crew definition for the agentic AI component."""

from crewai import Crew
from .config import Config
from .agents import create_researcher_agent, create_writer_agent
from .tasks import create_research_task, create_writing_task


class AgenticCrew:
    """Main crew for the agentic AI component."""
    
    def __init__(self):
        self.config = Config() # type: ignore (populated by pydantic)
        self.researcher = create_researcher_agent(self.config)
        self.writer = create_writer_agent(self.config)
    
    def run_research_and_write(self, topic: str) -> str:
        """Run a research and writing workflow."""
        research_task = create_research_task(self.researcher, topic)
        writing_task = create_writing_task(self.writer)
        
        crew = Crew(
            agents=[self.researcher, self.writer],
            tasks=[research_task, writing_task],
            verbose=True
        )
        
        result = crew.kickoff()
        return result