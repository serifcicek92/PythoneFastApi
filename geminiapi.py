#pip install -q -U google-genai

from google import genai

client = genai.Client(
    api_key=KEY,
    http_options={'api_version': 'v1alpha'}
)

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Explain how AI works in a few words",
)

print(response.text)