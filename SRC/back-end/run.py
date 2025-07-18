from app import create_app
from pyngrok import ngrok # type: ignore
import time
import os
from dotenv import load_dotenv # type: ignore
import subprocess

NGROK_DOMAIN = "trusting-manually-monkey.ngrok-free.app"
PORT = 5300

load_dotenv()
NGROK_AUTHTOKEN = os.getenv("NGROK_TOKEN")

app = create_app()

if __name__ == '__main__':
    subprocess.run(["ngrok", "config", "add-authtoken", NGROK_AUTHTOKEN])
    
    public_url = ngrok.connect(addr=PORT, bind_tls=True, domain=NGROK_DOMAIN)
    print(f" * ngrok tunnel available at: {public_url}")
    
    time.sleep(1)

    app.run(port=PORT)