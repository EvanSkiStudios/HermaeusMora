import asyncio
import os
import sys

import ollama
from ollama import Client, chat
from dotenv import load_dotenv

from hermaeus.HermaMora_Config import HermaeusMora_System_Prompt
from utility_scripts.system_logging import setup_logger

# configure logging
logger = setup_logger(__name__)

load_dotenv()
os.environ["OLLAMA_API_KEY"] = os.getenv("OLLAMA_API")


HM_personality = HermaeusMora_System_Prompt
HM_model_name = "HermaeusMora:latest"
HM_base_model = "deepseek-r1:7b"


class HermaeusMora:
    def __init__(self):
        self.client = None
        self.model_name = HM_model_name
        self.base_model = HM_base_model
        self.system_prompt = HM_personality
        self.options = {
            'num_ctx': 16384,
            'temperature': 0.6,
        }

    def create(self) -> None:
        try:
            self.client = Client()
            response = self.client.create(
                model=self.model_name,
                from_=self.base_model,
                system=self.system_prompt,
                stream=False,
            )

            logger.info(f"Model created: {response['status']}")

        except ConnectionError:
            logger.error("Ollama is not running")
            sys.exit(1)

        except Exception as e:
            logger.error(f"Unexpected error during model creation: {e}")
            sys.exit(1)

    def generate(self, prompt: str) -> str:
        options = self.options | {'think': False}

        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options=options,
            stream=False
        )
        return response["response"]

    def chat(self, prompt: str, context: str) -> str:
        options = self.options | {'think': True}

        response = chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": f"Use this context to respond to the user:\n{context}"},
                {"role": "user", "content": prompt}
            ],
            options=options,
            stream=False
        )
        print("CONTEXT:")
        print(context)
        print("=" * 60)
        print(response.message.thinking)
        print("=" * 60)
        return response.message.content
