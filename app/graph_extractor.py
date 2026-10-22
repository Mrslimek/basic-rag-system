import json
import re
from typing import Any

import httpx2 as httpx

from app.config import get_settings


class GraphExtractor:

    def __init__(self):
        self.settings = get_settings()
        self.prompt = """
            Extract entities and relationships from this text.
            Return ONLY JSON with:
            - "entities": [{"name": "Authentication", "type": "service"}]
            - "relations": [{"source": "Auth", "target": "Billing", "relation": "triggers"}]
            Text:
            {text}
        """

    async def extract(self, text: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.settings.LLM_API_BASE.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.LLM_API_KEY}"},
                json={
                    "model": self.settings.LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": "Return ONLY valid JSON."},
                        {"role": "user", "content": self.prompt.replace("{text}", text)},
                    ],
                    "temperature": 0.1,
                },
            )
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(self._extract_json_block(content))

    @staticmethod
    def _extract_json_block(raw: str) -> str:
        """
        Pull the JSON payload out of an LLM response.

        LLMs wrap JSON in markdown fences (```json ... ```) and often add
        prose before/after. A naive `strip("```json")` is dangerous: str.strip()
        takes a *set of characters*, not a prefix, so it would also silently
        eat any leading/trailing 'j', 's', 'o' or 'n' characters.
        """
        fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", raw, re.DOTALL)
        if fenced:
            return fenced.group(1)
        # Fallback: take the outermost {...} block
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        return brace.group(0) if brace else raw
