import requests
import sys

def check_ai_health(base_url):
    print(f"🔍 Checking AI Engine Health at {base_url}...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print(f"✅ AI Engine Online: {response.json()}")
            return True
        else:
            print(f"❌ AI Engine Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to AI Engine: {str(e)}")
        return False

def test_ai_scoring(base_url):
    print(f"\n🧪 Testing AI Scoring Engine...")
    test_opportunity = {
        "title": "Solana Data Infrastructure Grant",
        "description": "We are looking for developers to build decentralized data indexing solutions on Solana.",
        "skills": ["Rust", "Indexing", "GRPC"],
        "category": "Grant"
    }
    
    try:
        # Assuming there's a score endpoint in main.py
        response = requests.post(f"{base_url}/score", json=test_opportunity)
        if response.status_code == 200:
            print(f"✅ Scoring Sync: {response.json()}")
        else:
            print(f"⚠️ Scoring test returned status: {response.status_code}")
    except Exception as e:
        print(f"❌ Scoring test failed: {str(e)}")

if __name__ == "__main__":
    url = "http://localhost:8001"
    if len(sys.argv) > 1:
        url = sys.argv[1]
    
    if check_ai_health(url):
        # Only run scoring test if health is OK
        # and if the user specifically wants to test logic
        test_ai_scoring(url)
