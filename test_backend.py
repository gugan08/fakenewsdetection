import urllib.request
import urllib.error

try:
    r = urllib.request.urlopen("https://fakenewsdetection-c7jc.onrender.com/", timeout=30)
    print("Status:", r.status)
    print(r.read().decode())
except urllib.error.URLError as e:
    print("URLError:", e)
except TimeoutError:
    print("TIMEOUT: Backend did not respond within 30 seconds")
except Exception as e:
    print("Error:", type(e).__name__, e)
