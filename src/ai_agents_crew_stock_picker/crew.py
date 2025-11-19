"""
Moduł główny definiujący załogę agentów AI do wyboru akcji.

Ten moduł implementuje kompleksowy system agentów AI wykorzystujący CrewAI,
który współpracuje w celu znalezienia, przeanalizowania i wyboru najlepszej
firmy do inwestycji w określonym sektorze.

Kluczowe komponenty:
- Schematy Pydantic: Definicje struktury danych dla ustrukturyzowanych wyjść
- Agenci: Specjalizowani agenci do różnych zadań
- Zadania: Definicje zadań wykonywanych przez agentów
- Proces hierarchiczny: Manager LLM koordynujący pracę agentów
- Pamięć: System pamięci krótko- i długoterminowej dla agentów
"""

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

# ============================================================================
# SCHEMATY PYDANTIC - USTRUKTURYZOWANE WYJŚCIA
# ============================================================================
# Schematy Pydantic definiują strukturę danych, które agenci muszą zwracać.
# Dzięki temu LLM jest zmuszony do zwracania danych w określonym formacie JSON,
# co zapewnia stabilność i łatwość przetwarzania wyników przez kod Pythona.


class TrendingCompany(BaseModel):
    """
    Reprezentuje pojedynczą firmę, która jest w trendzie w wiadomościach.

    Ten schemat jest używany jako część ustrukturyzowanego wyjścia
    z zadania 'Find Trending Companies', zapewniając spójny format danych.
    """

    name: str = Field(description="Nazwa firmy")
    ticker: str = Field(description="Symbol giełdowy firmy (np. AAPL, MSFT)")
    reason: str = Field(
        description="Powód, dlaczego firma jest w wiadomościach i przyciąga uwagę"
    )


class TrendingCompanyList(BaseModel):
    """
    Lista firm w trendzie w wiadomościach.

    Główny schemat wyjściowy dla zadania 'Find Trending Companies'.
    Zawiera listę 2-3 firm, które są aktualnie popularne w wiadomościach.
    """

    companies: List[TrendingCompany] = Field(
        description="Lista firm, o których jest głośno w wiadomościach"
    )


class TrendingCompanyResearch(BaseModel):
    """
    Szczegółowa analiza finansowa pojedynczej firmy.

    Ten schemat reprezentuje kompleksową analizę wykonaną przez Financial Analyst
    dla każdej firmy znalezionej w trendzie. Zawiera kluczowe informacje
    potrzebne do podjęcia decyzji inwestycyjnej.
    """

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
    """
    Lista szczegółowych analiz wszystkich firm w trendzie.

    Główny schemat wyjściowy dla zadania 'Research Trending Companies'.
    Zawiera kompleksowe analizy wszystkich firm znalezionych w pierwszym kroku.
    """

    research_list: List[TrendingCompanyResearch] = Field(
        description="Kompleksowa analiza wszystkich firm w trendzie"
    )


# ============================================================================
# GŁÓWNA KLASA ZAŁOGI
# ============================================================================
# Klasa dziedzicząca z CrewBase, która definiuje całą załogę agentów.
# Używa dekoratorów @agent, @task i @crew do automatycznego rejestrowania
# komponentów w systemie CrewAI.


@CrewBase
class AiAgentsCrewStockPicker:
    """
    Główna klasa załogi agentów AI do wyboru akcji.

    Ta klasa definiuje strukturę całego systemu multi-agentowego:
    - Agenci: Specjalizowani agenci do różnych zadań
    - Zadania: Definicje zadań wykonywanych przez agentów
    - Proces: Hierarchiczny proces z managerem LLM
    - Pamięć: System pamięci dla agentów
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    # ========================================================================
    # DEFINICJE AGENTÓW
    # ========================================================================
    # Każdy agent jest zdefiniowany jako metoda z dekoratorem @agent.
    # Konfiguracja agenta (rola, cel, backstory) jest ładowana z pliku YAML.

    @agent
    def trending_company_finder(self) -> Agent:
        """
        Agent odpowiedzialny za wyszukiwanie firm w trendzie w wiadomościach.

        Narzędzia:
            - SerperDevTool: Narzędzie do wyszukiwania w internecie

        Pamięć:
            - Włączona: Agent pamięta wcześniej znalezione firmy,
              aby unikać duplikatów w kolejnych uruchomieniach
        """
        return Agent(
            config=self.agents_config["trending_company_finder"],
            tools=[SerperDevTool()],
            memory=True,  # Włączona pamięć, aby unikać duplikatów
        )

    @agent
    def financial_researcher(self) -> Agent:
        """
        Agent odpowiedzialny za szczegółową analizę finansową firm.

        Narzędzia:
            - SerperDevTool: Narzędzie do wyszukiwania informacji finansowych

        Pamięć:
            - Wyłączona: Agent wykonuje czystą analizę danych dostarczonych
              w kontekście, nie buduje trwałej wiedzy
        """
        return Agent(
            config=self.agents_config["financial_researcher"],
            tools=[SerperDevTool()],
            # Pamięć wyłączona - agent wykonuje analizę na podstawie kontekstu
        )

    @agent
    def stock_picker(self) -> Agent:
        """
        Agent odpowiedzialny za wybór najlepszej firmy do inwestycji.

        Narzędzia:
            - PushNotificationTool: Niestandardowe narzędzie do wysyłania
              powiadomień push do użytkownika

        Pamięć:
            - Włączona: Agent pamięta wcześniejsze rekomendacje,
              aby unikać wybierania tych samych firm ponownie
        """
        return Agent(
            config=self.agents_config["stock_picker"],
            tools=[PushNotificationTool()],
            memory=True,  # Włączona pamięć, aby unikać powtórzeń
        )

    # ========================================================================
    # DEFINICJE ZADAŃ
    # ========================================================================
    # Każde zadanie jest zdefiniowane jako metoda z dekoratorem @task.
    # Konfiguracja zadania (opis, oczekiwany wynik, kontekst) jest ładowana
    # z pliku YAML. Ustrukturyzowane wyjścia są wymuszane przez output_pydantic.

    @task
    def find_trending_companies(self) -> Task:
        """
        Zadanie: Znajdź firmy w trendzie w wiadomościach.

        Wykonawca: trending_company_finder
        Wyjście: Ustrukturyzowane (TrendingCompanyList)

        To zadanie wymusza zwrócenie danych w formacie JSON zgodnym
        ze schematem Pydantic, co zapewnia stabilność transferu danych.
        """
        return Task(
            config=self.tasks_config["find_trending_companies"],
            output_pydantic=TrendingCompanyList,  # Wymusza ustrukturyzowane wyjście
        )

    @task
    def research_trending_companies(self) -> Task:
        """
        Zadanie: Przeanalizuj szczegółowo znalezione firmy.

        Wykonawca: financial_researcher
        Kontekst: find_trending_companies (otrzymuje listę firm)
        Wyjście: Ustrukturyzowane (TrendingCompanyResearchList)

        To zadanie otrzymuje wyniki z poprzedniego zadania jako kontekst
        i generuje szczegółową analizę każdej firmy.
        """
        return Task(
            config=self.tasks_config["research_trending_companies"],
            output_pydantic=TrendingCompanyResearchList,  # Wymusza ustrukturyzowane wyjście
        )

    @task
    def pick_best_company(self) -> Task:
        """
        Zadanie: Wybierz najlepszą firmę do inwestycji.

        Wykonawca: stock_picker
        Kontekst: research_trending_companies (otrzymuje analizy firm)
        Wyjście: Tekstowe (raport markdown)

        To zadanie analizuje wszystkie analizy i wybiera najlepszą firmę,
        wysyłając powiadomienie push do użytkownika.
        """
        return Task(
            config=self.tasks_config["pick_best_company"],
            # Brak output_pydantic - zwraca tekstowy raport markdown
        )

    # ========================================================================
    # KONFIGURACJA ZAŁOGI
    # ========================================================================
    # Metoda @crew definiuje główną konfigurację załogi, w tym:
    # - Proces hierarchiczny z managerem LLM
    # - System pamięci (krótko- i długoterminowa)
    # - Listę agentów i zadań

    @crew
    def crew(self) -> Crew:
        """
        Tworzy i konfiguruje główną załogę agentów.

        Proces hierarchiczny:
            - Manager LLM autonomicznie decyduje o kolejności zadań
            - Manager deleguje zadania do odpowiednich agentów
            - Proces jest bardziej elastyczny niż sekwencyjny

        Pamięć:
            - Long-term: SQLite - przechowuje ważne informacje długoterminowo
            - Short-term: RAG (ChromaDB) - przechowuje ostatnie interakcje
            - Entity: RAG (ChromaDB) - przechowuje informacje o konkretnych encjach
        """

        # ====================================================================
        # MANAGER AGENT
        # ====================================================================
        # Manager to specjalny agent LLM, który koordynuje pracę załogi.
        # allow_delegation=True umożliwia managerowi delegowanie zadań
        # do innych agentów w sposób autonomiczny.
        manager = Agent(
            config=self.agents_config["manager"],
            allow_delegation=True,  # Manager może delegować zadania
        )

        # ====================================================================
        # KONFIGURACJA PAMIĘCI DŁUGOTERMINOWEJ
        # ====================================================================
        # Pamięć długoterminowa przechowuje ważne informacje w bazie SQLite.
        # Używana do przechowywania kluczowych decyzji i rekomendacji.
        long_term_memory = LongTermMemory(
            storage=LTMSQLiteStorage(db_path="./memory/long_term_mem_store.db")
        )

        # ====================================================================
        # KONFIGURACJA PAMIĘCI KRÓTKOTERMINOWEJ
        # ====================================================================
        # Pamięć krótkoterminowa przechowuje ostatnie interakcje w ChromaDB.
        # Używa embeddings do wyszukiwania podobnych kontekstów.
        # Wymaga modelu embeddings (np. text-embedding-3-small).
        short_term_memory = ShortTermMemory(
            storage=RAGStorage(
                embedder_config={
                    "provider": "openai",
                    "model": "text-embedding-3-small",  # Model do generowania embeddings
                },
                type="short_term",
                path="./memory",  # Ścieżka do przechowywania danych ChromaDB
            )
        )

        # ====================================================================
        # KONFIGURACJA PAMIĘCI ENCJI
        # ====================================================================
        # Pamięć encji przechowuje informacje o konkretnych rzeczach
        # (firmy, osoby, miejsca) w ChromaDB.
        # Umożliwia agentom zapamiętywanie szczegółów o konkretnych firmach.
        entity_memory = EntityMemory(
            storage=RAGStorage(
                embedder_config={
                    "provider": "openai",
                    "model": "text-embedding-3-small",  # Model do generowania embeddings
                },
                type="short_term",  # Typ magazynu (może być również "entity")
                path="./memory",  # Ścieżka do przechowywania danych ChromaDB
            )
        )

        # ====================================================================
        # TWORZENIE ZAŁOGI
        # ====================================================================
        # Crew łączy wszystkich agentów, zadania, proces i pamięć w jeden
        # spójny system multi-agentowy.
        return Crew(
            agents=self.agents,  # Lista wszystkich agentów wykonawczych
            tasks=self.tasks,  # Lista wszystkich zadań do wykonania
            process=Process.hierarchical,  # Proces hierarchiczny z managerem
            verbose=True,  # Szczegółowe logi podczas działania
            manager_agent=manager,  # Agent zarządzający (delegujący zadania)
            memory=True,  # Włącza system pamięci dla załogi
            long_term_memory=long_term_memory,  # Pamięć długoterminowa
            short_term_memory=short_term_memory,  # Pamięć krótkoterminowa
            entity_memory=entity_memory,  # Pamięć encji
        )
