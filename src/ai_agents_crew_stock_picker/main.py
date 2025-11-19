#!/usr/bin/env python
"""
Główny plik uruchomieniowy dla załogi agentów Stock Picker.

Ten plik służy do lokalnego uruchomienia załogi agentów AI.
Zawiera minimalną logikę - głównie konfigurację danych wejściowych
i uruchomienie procesu załogi.

Użycie:
    python -m ai_agents_crew_stock_picker.main
    lub
    crewai run
"""

import warnings

from ai_agents_crew_stock_picker.crew import AiAgentsCrewStockPicker

# Ignoruj ostrzeżenia składniowe z modułu pysbd (używanego przez CrewAI)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def main():
    """
    Główna funkcja uruchamiająca załogę agentów.

    Definiuje dane wejściowe dla załogi i uruchamia proces hierarchiczny.
    Dane wejściowe są automatycznie interpolowane do zadań i agentów
    zgodnie z konfiguracją w plikach YAML.
    """
    # ========================================================================
    # KONFIGURACJA DANYCH WEJŚCIOWYCH
    # ========================================================================
    # Te dane są przekazywane do zadań i agentów jako zmienne kontekstowe.
    # W plikach YAML można używać {sector} i {region} do interpolacji.
    inputs = {
        "sector": "technology",  # Sektor do analizy (np. technology, healthcare, finance)
        "region": "Africa",  # Region geograficzny (opcjonalny)
    }

    # ========================================================================
    # TWORZENIE I URUCHOMIENIE ZAŁOGI
    # ========================================================================
    # 1. Tworzy instancję klasy załogi
    # 2. Wywołuje metodę crew() aby uzyskać obiekt Crew
    # 3. Uruchamia proces za pomocą kickoff() z danymi wejściowymi
    crew_instance = AiAgentsCrewStockPicker()
    crew = crew_instance.crew()
    result = crew.kickoff(inputs=inputs)

    # ========================================================================
    # WYŚWIETLANIE WYNIKÓW
    # ========================================================================
    # Wyświetla surowy wynik (raw) zawierający pełny raport decyzyjny
    print("\n\n=== FINAL DECISION ===\n\n")
    print(result.raw)


if __name__ == "__main__":
    main()
