from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.memory import EntityMemory, LongTermMemory, ShortTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field

from .tools.push_tool import PushNotificationTool


class TrendingCompany(BaseModel):
    """Firma, która jest w wiadomościach i przyciąga uwagę"""

    name: str = Field(description="Nazwa firmy")
    ticker: str = Field(description="Symbol giełdowy firmy")
    reason: str = Field(description="Powód, dlaczego firma jest w wiadomościach")


class TrendingCompanyList(BaseModel):
    """Lista wielu firm w trendzie w wiadomościach"""

    companies: List[TrendingCompany] = Field(
        description="Lista firm, o których jest głośno w wiadomościach"
    )


class TrendingCompanyResearch(BaseModel):
    """Szczegółowa analiza firmy"""

    name: str = Field(description="Nazwa firmy")
    market_position: str = Field(
        description="Aktualna pozycja rynkowa i analiza konkurencji"
    )
    future_outlook: str = Field(
        description="Perspektywy rozwoju i potencjał inwestycyjny"
    )
    investment_potential: str = Field(
        description="Potencjał inwestycyjny i odpowiedniość dla inwestycji"
    )


class TrendingCompanyResearchList(BaseModel):
    """A list of detailed research on all the companies"""

    research_list: List[TrendingCompanyResearch] = Field(
        description="Kompleksowa analiza wszystkich firm w trendzie"
    )


@CrewBase
class AiAgentsCrewStockPicker:
    """AiAgentsCrewStockPicker crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def trending_company_finder(self) -> Agent:
        return Agent(
            config=self.agents_config["trending_company_finder"],
            tools=[SerperDevTool()],
            memory=True,
        )

    @agent
    def financial_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["financial_researcher"], tools=[SerperDevTool()]
        )

    @agent
    def stock_picker(self) -> Agent:
        return Agent(
            config=self.agents_config["stock_picker"],
            tools=[PushNotificationTool()],
            memory=True,
        )

    @task
    def find_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config["find_trending_companies"],
            output_pydantic=TrendingCompanyList,
        )

    @task
    def research_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config["research_trending_companies"],
            output_pydantic=TrendingCompanyResearchList,
        )

    @task
    def pick_best_company(self) -> Task:
        return Task(
            config=self.tasks_config["pick_best_company"],
        )

    @crew
    def crew(self) -> Crew:
        "Creates the Stock Picker Crew"

        manager = Agent(config=self.agents_config["manager"], allow_delegation=True)

        long_term_memory = LongTermMemory(
            storage=LTMSQLiteStorage(db_path="./memory/long_term_mem_store.db")
        )

        short_term_memory = ShortTermMemory(
            storage=RAGStorage(
                embedder_config={
                    "provider": "openai",
                    "model": "text-embedding-3-small",  # Bezpośrednio tutaj
                },
                type="short_term",
                path="./memory",
            )
        )

        entity_memory = EntityMemory(
            storage=RAGStorage(
                embedder_config={
                    "provider": "openai",
                    "model": "text-embedding-3-small",  # Bezpośrednio, bez zagnieżdżonego "config"
                },
                type="short_term",
                path="./memory",
            )
        )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            verbose=True,
            manager_agent=manager,
            memory=True,
            long_term_memory=long_term_memory,
            short_term_memory=short_term_memory,
            entity_memory=entity_memory,
        )
