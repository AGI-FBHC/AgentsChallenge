from typing import Any, List
from langchain_core.embeddings import Embeddings
import requests

class InfinigenceEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        model: str = "bge-m3",
        infinity_api_url: str = "https://cloud.infini-ai.com/maas/v1"
    ):
        self.api_key = api_key
        self.model = model
        self.infinity_api_url = infinity_api_url
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "input": texts
        }
        
        response = requests.post(
            f"{self.infinity_api_url}/embeddings",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return [data["embedding"] for data in response.json()["data"]]
        else:
            raise ValueError(f"API call failed: {response.text}")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text into a vector"""
        embeddings = self.embed_documents([text])
        return embeddings[0] 