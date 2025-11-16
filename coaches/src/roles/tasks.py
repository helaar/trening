"""Task definitions for the agentic AI component."""

from crewai import Task
from crewai import Agent


def create_research_task(agent: Agent, topic: str) -> Task:
    """Create a research task."""
    return Task(
        description=f"Research the topic: {topic}. Gather comprehensive information and key insights.",
        expected_output="A detailed research report with key findings and insights.",
        agent=agent
    )


def create_writing_task(agent: Agent, research_context: str = "") -> Task:
    """Create a writing task."""
    return Task(
        description=f"Write a clear and engaging article based on the research findings. {research_context}",
        expected_output="A well-structured article with clear sections and engaging content.",
        agent=agent
    )