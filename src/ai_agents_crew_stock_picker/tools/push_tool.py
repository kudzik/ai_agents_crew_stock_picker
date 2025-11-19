import os
from typing import Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class PushNotification(BaseModel):
    """Message to be sent to the user"""

    message: str = Field(..., description="Message to be sent to the user.")


class PushNotificationTool(BaseTool):
    name: str = "Wyślij powiadomienie"
    description: str = "Tool do wysyłania powiadomień do użytkownika."
    args_schema: Type[BaseModel] = PushNotification

    def _run(self, message: str) -> str:
        pushover_user = os.getenv("PUSHOVER_USER")
        pushover_token = os.getenv("PUSHOVER_TOKEN")
        pushover_url = os.getenv("PUSHOVER_URL")

        print(f"Push: {message}")
        payload = {"user": pushover_user, "token": pushover_token, "message": message}
        requests.post(pushover_url, data=payload)
        return '{"notification": "ok"}'
