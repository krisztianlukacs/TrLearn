import os
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that helps generate book content."
        },
        {
            "role": "user",
            "content": "Generate a list of chapter titles and brief summaries for a book titled 'artificial intelligence, machine learning, neural networks'.",
        },
    ]
)
print(chat_response.choices[0].message.content)

