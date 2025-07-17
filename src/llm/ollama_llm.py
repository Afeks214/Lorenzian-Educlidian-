import requests
from typing import Dict, Any

class OllamaLLM:
    """Lightweight local LLM using Ollama"""
    
    def __init__(self, host="localhost", port=11434, model="phi"):
        self.base_url = f"http://{host}:{port}"
        self.model = model
        self._verify_connection()
    
    def _verify_connection(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags")
            if r.status_code == 200:
                print("✅ Ollama connected")
        except (ConnectionError, OSError, TimeoutError) as e:
            print("❌ Ollama connection failed")
    
    def explain_trade(self, context: Dict[str, Any]) -> str:
        prompt = f"Explain this trade: {context}"
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        
        return response.json().get('response', 'No explanation')