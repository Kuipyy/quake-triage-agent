"""Quick smoke test: single Gemini (or Groq) API call."""
import os
from dotenv import load_dotenv

load_dotenv()

provider = os.getenv("LLM_PROVIDER", "gemini")
model = os.getenv("LLM_MODEL", "gemini-2.5-flash")

if provider == "gemini":
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=model,
        contents="Reply with exactly: OK",
    )
    print(f"Provider: {provider} | Model: {model}")
    print(f"Response: {response.text.strip()}")
elif provider == "groq":
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )
    print(f"Provider: {provider} | Model: {model}")
    print(f"Response: {response.choices[0].message.content.strip()}")
else:
    raise ValueError(f"Unknown provider: {provider}")
