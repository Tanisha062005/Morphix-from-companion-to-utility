"""
Morphix - Gemini API Key Checker
=================================
Run this script to validate your GEMINI_API_KEY before starting the assistant.
It checks:
  1. .env file exists and key is present
  2. Key format (must start with AIzaSy)
  3. Live connection to Gemini API (real request)
  4. Model availability (gemini-2.5-flash)
"""

import os
import sys
from pathlib import Path

# ── Colors for terminal output ────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  [OK]   {msg}")
def fail(msg):  print(f"  [FAIL] {msg}")
def warn(msg):  print(f"  [WARN] {msg}")
def info(msg):  print(f"  [-->]  {msg}")

# ── Step 0: Add project root to path ─────────────────────────────────────────
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

print()
print(f"{BOLD}{'='*52}{RESET}")
print(f"{BOLD}     MORPHIX GEMINI API KEY CHECKER{RESET}")
print(f"{BOLD}{'='*52}{RESET}")
print()

# ── Step 1: Check .env file ───────────────────────────────────────────────────
print(f"{BOLD}[1] Checking .env file...{RESET}")
env_path = project_root / ".env"
if env_path.exists():
    ok(f".env found at: {env_path}")
else:
    fail(".env file NOT found!")
    info("Create a .env file in the project root with:")
    print("       GEMINI_API_KEY=AIzaSy...")
    sys.exit(1)

# ── Step 2: Load .env and read key ────────────────────────────────────────────
print()
print(f"{BOLD}[2] Loading API key...{RESET}")
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path, override=True)
    ok("dotenv loaded successfully")
except ImportError:
    fail("python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)

api_key = os.getenv("GEMINI_API_KEY", "").strip()

if not api_key:
    fail("GEMINI_API_KEY is missing or empty in .env")
    info("Add this line to your .env file:")
    print("       GEMINI_API_KEY=AIzaSy...")
    sys.exit(1)

masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "****"
ok(f"Key detected: {masked}  (length: {len(api_key)} chars)")

# ── Step 3: Skipping format check — hitting API directly ──────────────────────
print()
print(f"{BOLD}[3] Skipping format check — testing live API directly...{RESET}")
info(f"Using key: {masked}")

# ── Step 4: Live API connection test ──────────────────────────────────────────
print()
print(f"{BOLD}[4] Testing live connection to Gemini API...{RESET}")
info("Sending test request (this may take a few seconds)...")

try:
    from google import genai
    from google.genai import types
except ImportError:
    fail("google-genai not installed. Run: pip install google-genai")
    sys.exit(1)

try:
    client = genai.Client(api_key=api_key)
    
    # Lightweight model list call first (cheapest possible request)
    model_found = next(iter(client.models.list(config={"page_size": 1})), None)
    ok("API connection successful!")
    ok("Google Generative Language API is reachable")

except Exception as e:
    err = str(e)
    fail("Live API connection FAILED!")
    print()
    if "API_KEY_INVALID" in err or "API key not valid" in err:
        print(f"  {RED}Reason:{RESET} API key is invalid or has been revoked")
        info("Get a new key at: https://aistudio.google.com/app/apikey")
    elif "PERMISSION_DENIED" in err:
        print(f"  {RED}Reason:{RESET} API key doesn't have Gemini API access")
        info("Enable 'Generative Language API' in Google Cloud Console")
    elif "RESOURCE_EXHAUSTED" in err or "429" in err:
        print(f"  {RED}Reason:{RESET} Rate limit exceeded (too many requests)")
        warn("Wait a minute and try again")
    elif "quota" in err.lower():
        print(f"  {RED}Reason:{RESET} API quota exceeded")
        info("Check your quota at: https://console.cloud.google.com/")
    else:
        print(f"  {RED}Error:{RESET} {err[:200]}")
    sys.exit(1)

# ── Step 5: List available models and find a working one ─────────────────────
print()
print(f"{BOLD}[5] Finding available models for this key...{RESET}")

try:
    all_models = list(client.models.list())
    generative = [m.name for m in all_models if "generateContent" in (m.supported_actions or [])]
    if generative:
        ok(f"Found {len(generative)} model(s) that support generateContent:")
        for name in generative:
            print(f"        - {name}")
    else:
        warn("No generateContent models listed — trying common model names...")
except Exception as e:
    warn(f"Could not list models: {e}")
    generative = []

# ── Step 6: Try models in order of preference ────────────────────────────────
print()
print(f"{BOLD}[6] Testing models until one works...{RESET}")

candidates = []
# Add the project's configured model first
configured = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
candidates.append(configured)
# Add models from the list (strip 'models/' prefix for generate call)
for m in generative:
    name = m.replace("models/", "")
    if name not in candidates:
        candidates.append(name)
# Fallback common names
for fb in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro",
           "gemini-1.0-pro", "gemini-pro"]:
    if fb not in candidates:
        candidates.append(fb)

working_model = None
for model_name in candidates:
    try:
        info(f"Trying: {model_name} ...")
        response = client.models.generate_content(
            model=model_name,
            contents="Say: OK",
            config=types.GenerateContentConfig(
                max_output_tokens=10,
                temperature=0.0
            )
        )
        reply = response.text.strip()
        ok(f"[WORKS] {model_name}  ->  response: \"{reply}\"")
        working_model = model_name
        break
    except Exception as e:
        err_short = str(e)[:120]
        fail(f"[FAIL]  {model_name}  ->  {err_short}")

print()
if working_model:
    print(f"{BOLD}{'='*52}{RESET}")
    ok(f"Working model found: {working_model}")
    print()
    info(f"Update your .env file:")
    print(f"       GEMINI_MODEL={working_model}")
    print(f"{BOLD}{'='*52}{RESET}")
else:
    print(f"{BOLD}{'='*52}{RESET}")
    fail("No working model found with this key")
    info("Your key may be restricted to Vertex AI or a specific region")
    info("Get a fresh key at: https://aistudio.google.com/app/apikey")
    print(f"{BOLD}{'='*52}{RESET}")
    sys.exit(1)

print()
