import os
import sys
from pathlib import Path

# Add the 'src' directory to the Python path so we can import the inference module
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR / "src"))
# Standardize current working directory for imports
os.chdir(BASE_DIR)

try:
    from inference.analyze import analyze_text
    print("[+] Model and analysis module loaded successfully.\n")
except ImportError as e:
    print(f"[-] Error: Could not import analyze_text. {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

print("--- MEDVERAX HEALTH MISINFORMATION DETECTION ---")
print("--- INTERACTIVE MODEL TESTING ---")
print("(Type 'exit', 'quit', or press Ctrl+C to stop)\n")

while True:
    try:
        text = input("\n[?] ENTER HEALTH CLAIM TO ANALYZE: ").strip()
        
        if not text:
            continue
            
        if text.lower() in ['exit', 'quit']:
            print("\n--- TESTING SESSION CONCLUDED ---")
            break
            
        print(f"\n[*] ANALYZING: {text}")
        
        result = analyze_text(text)
        print(f"    RESULT: {result['classification']}")
        print(f"    SCORE: {result['risk_score']}/100")
        print(f"    CONFIDENCE: {result['confidence_score']}%")
        print(f"    VERDICT: {result['verdict']}")
        
        if result.get('highlighted_phrases'):
            keys = ", ".join([h['text'] for h in result['highlighted_phrases']])
            print(f"    DETECTIONS: {keys}")
            
    except KeyboardInterrupt:
        print("\n\n--- TESTING SESSION TERMINATED BY USER ---")
        break
    except Exception as e:
        print(f"    Error: {e}")
