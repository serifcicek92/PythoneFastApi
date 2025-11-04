import requests
from flask import Flask, request, Response

app = Flask(__name__)
TARGET = "http://192.168.3.4:8000"

@app.route('/', defaults={'path': ''}, methods=["GET", "POST", "PUT", "DELETE"])
@app.route('/<path:path>', methods=["GET", "POST", "PUT", "DELETE"])
def proxy(path):
    url = f"{TARGET}/{path}"
    resp = requests.request(
        method=request.method,
        url=url,
        headers={key: value for key, value in request.headers},
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False)
    response = Response(resp.content, resp.status_code)
    for key, value in resp.headers.items():
        response.headers[key] = value
    return response

app.run(host="0.0.0.0", port=80)


#python -m pip install flask
