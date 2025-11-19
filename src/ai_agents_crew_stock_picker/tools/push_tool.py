"""
Niestandardowe narzędzie do wysyłania powiadomień push do użytkownika.

To narzędzie demonstruje, jak tworzyć własne narzędzia w CrewAI.
Używa serwisu Pushover do wysyłania powiadomień push na urządzenia mobilne.

Wymagane zmienne środowiskowe:
    - PUSHOVER_USER: ID użytkownika Pushover
    - PUSHOVER_TOKEN: Token aplikacji Pushover
    - PUSHOVER_URL: URL API Pushover (domyślnie: https://api.pushover.net/1/messages.json)
"""

import os
from typing import Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ============================================================================
# SCHEMAT WEJŚCIOWY PYDANTIC
# ============================================================================
# Schemat Pydantic definiuje strukturę danych wejściowych dla narzędzia.
# LLM używa tego schematu, aby wiedzieć, jakie parametry przekazać do narzędzia.


class PushNotification(BaseModel):
    """
    Schemat danych wejściowych dla narzędzia powiadomień push.
    
    Ten schemat jest używany przez LLM do zrozumienia, jakie dane
    należy przekazać do narzędzia podczas jego wywołania.
    """

    message: str = Field(
        ...,
        description="Treść powiadomienia do wysłania do użytkownika."
    )


# ============================================================================
# KLASA NARZĘDZIA
# ============================================================================
# Każde niestandardowe narzędzie w CrewAI musi dziedziczyć z BaseTool
# i implementować wymagane pola oraz metodę _run().


class PushNotificationTool(BaseTool):
    """
    Niestandardowe narzędzie do wysyłania powiadomień push.
    
    To narzędzie umożliwia agentom wysyłanie powiadomień push do użytkownika
    za pomocą serwisu Pushover. Agent może autonomicznie zdecydować,
    kiedy użyć tego narzędzia (np. po podjęciu decyzji inwestycyjnej).
    
    Atrybuty:
        name: Nazwa narzędzia widoczna dla LLM
        description: Opis narzędzia używany przez LLM do decyzji o użyciu
        args_schema: Schemat Pydantic definiujący parametry wejściowe
    """

    name: str = "Wyślij powiadomienie"
    description: str = (
        "Narzędzie do wysyłania powiadomień push do użytkownika. "
        "Użyj tego narzędzia, gdy chcesz powiadomić użytkownika o ważnej decyzji "
        "lub wyniku analizy. Agent używa tego narzędzia autonomicznie."
    )
    args_schema: Type[BaseModel] = PushNotification

    def _run(self, message: str) -> str:
        """
        Wykonuje wysłanie powiadomienia push.
        
        Metoda _run() jest wywoływana przez CrewAI, gdy agent zdecyduje
        się użyć tego narzędzia. Pobiera dane z zmiennych środowiskowych
        i wysyła żądanie HTTP do API Pushover.
        
        Args:
            message: Treść powiadomienia do wysłania
        
        Returns:
            JSON string z potwierdzeniem wysłania
        
        Raises:
            ValueError: Jeśli brakuje wymaganych zmiennych środowiskowych
        """
        # Pobierz dane uwierzytelniające z zmiennych środowiskowych
        pushover_user = os.getenv("PUSHOVER_USER")
        pushover_token = os.getenv("PUSHOVER_TOKEN")
        pushover_url = os.getenv(
            "PUSHOVER_URL", 
            "https://api.pushover.net/1/messages.json"  # Domyślny URL
        )

        # Walidacja wymaganych zmiennych środowiskowych
        if not pushover_user or not pushover_token:
            error_msg = (
                "Brak wymaganych zmiennych środowiskowych: "
                "PUSHOVER_USER i/lub PUSHOVER_TOKEN"
            )
            print(f"Błąd: {error_msg}")
            return f'{{"error": "{error_msg}"}}'

        # Wyświetl informację o wysyłanym powiadomieniu (dla debugowania)
        print(f"Push: {message}")

        # Przygotuj dane do wysłania
        payload = {
            "user": pushover_user,
            "token": pushover_token,
            "message": message,
        }

        # Wyślij żądanie HTTP POST do API Pushover
        try:
            response = requests.post(pushover_url, data=payload, timeout=10)
            response.raise_for_status()  # Rzuć wyjątek dla kodów błędów HTTP
            return '{"notification": "ok"}'
        except requests.exceptions.RequestException as e:
            error_msg = f"Błąd podczas wysyłania powiadomienia: {str(e)}"
            print(f"Błąd: {error_msg}")
            return f'{{"error": "{error_msg}"}}'


if __name__ == "__main__":
    tool = PushNotificationTool()
    tool.run(message="Test message")
