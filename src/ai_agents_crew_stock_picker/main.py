#!/usr/bin/env python
import warnings

from ai_agents_crew_stock_picker.crew import AiAgentsCrewStockPicker

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


"""
Run the crew.
"""
inputs = {"sector": "technology", "region": "Africa"}

result = AiAgentsCrewStockPicker().crew().kickoff(inputs=inputs)
print("\n\n=== FINAL DECISION ===\n\n")
print(result.raw)
