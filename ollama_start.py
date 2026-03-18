import requests
import json
import ollama

url = "http://localhost:11434/api/generate"

data = {
    "model": "kimi-k2.5:cloud",
    "prompt": "Write a Python function that checks if a number is prime.",
    "max_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": False
}   

response = requests.post(url, json=data, stream=True)

if response.status_code == 200:
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            print(decoded_line)
else:
    print(f"Error: {response.status_code} - {response.text}")
    

res = ollama.chat(
    model="kimi-k2.5:cloud",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What color is the sky?"}
    ],
    stream=False
)
print(res)
