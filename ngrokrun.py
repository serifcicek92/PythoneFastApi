from pyngrok import ngrok, conf

# 🔐 NGROK API KEY (AUTHTOKEN)
NGROK_TOKEN = KEY

# Authtoken set et (ngrok config add-authtoken eşdeğeri)
conf.get_default().auth_token = NGROK_TOKEN

# 8000 portunu internete aç
public_url = ngrok.connect(8000, "http")

print("🌍 Public URL:", public_url)

# Program açık kaldığı sürece ngrok açık kalır
input("Çıkmak için ENTER...")
