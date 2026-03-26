from fastapi import FastAPI
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
import uvicorn
import os
from pathlib import Path

def load_local_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)

load_local_env()

app = FastAPI(title="CrewAI A2A Agent")

# Tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Define the CrewAI agent
assistant = Agent(
    role="Research Assistant",
    goal="Help with any task sent through the Ammunity network. When the task requires current information, search the web and scrape relevant pages to find accurate answers.",
    backstory="You are a helpful AI research agent connected to the Ammunity agent network. You receive tasks from other agents and complete them. You have access to web search and scraping tools — use them whenever a task requires up-to-date or factual information from the internet.",
    verbose=True,
    tools=[search_tool, scrape_tool],
    llm=LLM(
        model="openai/gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY")
    )
)

@app.post("/a2a/task")
async def receive_task(payload: dict):
    task_description = payload.get("task_description", "No task provided")
    message = payload.get("payload", {}).get("message", "")

    task = Task(
        description=f"{task_description}. {message}. Use the search and scrape tools to find accurate, up-to-date information if needed.",
        expected_output="A well-researched, helpful response based on real information from the web where relevant.",
        agent=assistant
    )

    crew = Crew(
        agents=[assistant],
        tasks=[task],
        verbose=True
    )

    result = crew.kickoff()
    return {
        "status": "completed",
        "agent": "crewai-agent",
        "result": str(result)
    }

@app.get("/health")
async def health():
    return {"status": "live", "agent": "crewai-agent"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)