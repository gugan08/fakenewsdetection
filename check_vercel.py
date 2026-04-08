import urllib.request
import re

try:
    html = urllib.request.urlopen("https://fakenewsdetection-ruby.vercel.app/").read().decode("utf-8")
    js_match = re.search(r'assets/(index-[^\.]+\.js)', html)
    if js_match:
        js_url = "https://fakenewsdetection-ruby.vercel.app/assets/" + js_match.group(1)
        js_content = urllib.request.urlopen(js_url).read().decode("utf-8")
        
        # Look for the URL fallback pattern: "http://10.63.22.159:8000" or similar
        print("JS Bundle URL:", js_url)
        matches = re.findall(r'https?://[^\s\"\'\`]+/predict', js_content)
        if matches:
            print("Found API Request URLs compiled into Vercel build:", matches)
        else:
            print("No /predict URLs found in JS.")
            
            # Check what VITE_API_URL actually compiled to
            fallback_match = re.findall(r'(\"[^\"]+\")\|\|\"http://10\.63\.22\.159:8000\"', js_content)
            print("Compiled VITE_API_URL logic:", fallback_match)
    else:
        print("Could not find JS file in HTML")
except Exception as e:
    print("Error:", e)
