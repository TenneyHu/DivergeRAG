import os
from openai import OpenAI


client = OpenAI()


response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Connection Successful' in Chinese."}
    ]
)
    

print(f"{response.choices[0].message.content}")
