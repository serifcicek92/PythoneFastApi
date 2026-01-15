#pythonCopyimport openai
import openai
# Set your API credentials (Moonshot/Kimi API)
openai.api_key = KEy               # use your Kimi API key
openai.api_base = "https://api.moonshot.ai/v1"         # Moonshot API endpoint

# Define a simple chat prompt
messages = [
    {"role": "system", "content": "You are Kimi, an AI assistant created by Moonshot AI."},
    {"role": "user", "content": "Explain what an LLM is to a 5-year-old."}
]

# Send the request for a chat completion
response = openai.ChatCompletion.create(
    model="kimi-k2-0711-preview",   # specify the Kimi model to use (e.g., Kimi K2 preview)
    messages=messages,
    temperature=0.6,
    max_tokens=256
)

# Print out the assistant's reply
print(response["choices"][0]["message"]["content"])