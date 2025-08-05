import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()  # This loads variable from .env file

# token = os.environ["GITHUB_TOKEN"]
# os.environ["GITHUB_TOKEN"]=os.getenv("GITHUB_TOKEN")
token = os.getenv("GITHUB_TOKEN")

endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-mini"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
    temperature=1.0,
    top_p=1.0,
    model=model
)

print(response.choices[0].message.content)