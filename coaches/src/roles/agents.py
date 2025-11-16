"""Agent definitions for the agentic AI component."""

from crewai import Agent
from .config import Config


def create_researcher_agent(config: Config) -> Agent:
    """Create a researcher agent."""
    return Agent(
        role="Researcher",
        goal="Research and gather information on given topics",
        backstory="You are an expert researcher with access to web search capabilities.",
        verbose=True,
        allow_delegation=False,
        llm=config.model_name
    )


def create_writer_agent(config: Config) -> Agent:
    """Create a writer agent."""
    return Agent(
        role="Writer",
        goal="Write clear and engaging content based on research",
        backstory="You are a skilled writer who can create compelling content from research data.",
        verbose=True,
        allow_delegation=False,
        llm=config.model_name
    )