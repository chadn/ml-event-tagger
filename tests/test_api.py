"""Test script for the API service."""

import time
import subprocess
import requests
import json
import sys

def test_api():
    """Test the API endpoints."""
    
    print("=" * 80)
    print("Testing ML Event Tagger API")
    print("=" * 80)
    print()
    
    # Start server in background
    print("Starting server...")
    server_process = subprocess.Popen(
        ["uvicorn", "ml_event_tagger.serve:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Wait for server to start and check for errors
    print("Waiting for server to start...")
    for i in range(10):
        time.sleep(0.5)
        # Check if process died
        if server_process.poll() is not None:
            print("\n❌ Server process died!")
            stdout, _ = server_process.communicate()
            print(f"Output:\n{stdout}")
            sys.exit(1)
        # Try to connect
        try:
            response = requests.get("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                print("✅ Server is up!")
                break
        except:
            pass
    else:
        print("⚠️  Server may not be fully ready, proceeding anyway...")
    
    base_url = "http://localhost:8000"
    
    try:
        # Test 1: Health endpoint
        print("\n" + "=" * 80)
        print("TEST 1: Health Check")
        print("=" * 80)
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            health_data = response.json()
            if health_data['model_loaded']:
                print("✅ Health check passed - Model loaded")
            else:
                print("❌ Model not loaded")
        else:
            print(f"❌ Health check failed with status {response.status_code}")
        
        # Test 2: Root endpoint
        print("\n" + "=" * 80)
        print("TEST 2: Root Endpoint")
        print("=" * 80)
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Test 3: Predict endpoint
        print("\n" + "=" * 80)
        print("TEST 3: Prediction Endpoint")
        print("=" * 80)
        
        test_event = {
            "events": [
                {
                    "name": "Days Like This - House Music",
                    "description": "Weekly house music gathering with local DJs",
                    "location": "The Pergola at Lake Merritt, Oakland, CA"
                }
            ]
        }
        
        print(f"Request: {json.dumps(test_event, indent=2)}")
        print()
        
        response = requests.post(
            f"{base_url}/predict",
            json=test_event,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            predictions = response.json()
            print(f"Response: {json.dumps(predictions, indent=2)}")
            print("\n✅ Prediction successful!")
            
            # Show top predictions
            if predictions['predictions']:
                first_pred = predictions['predictions'][0]
                print("\nTop 5 predicted tags:")
                for i, tag_pred in enumerate(first_pred['tags'][:5], 1):
                    print(f"  {i}. {tag_pred['tag']:<12} {tag_pred['confidence']:.3f}")
        else:
            print(f"❌ Prediction failed with status {response.status_code}")
            print(f"Response: {response.text}")
        
        # Test 4: Multiple events
        print("\n" + "=" * 80)
        print("TEST 4: Multiple Events")
        print("=" * 80)
        
        multi_events = {
            "events": [
                {
                    "name": "Jazz Night",
                    "description": "Live jazz performance",
                    "location": "Oakland"
                },
                {
                    "name": "Yoga in the Park",
                    "description": "Free outdoor yoga class",
                    "location": "Lake Merritt"
                }
            ]
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=multi_events,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            predictions = response.json()
            print(f"✅ Predicted tags for {len(predictions['predictions'])} events")
            for i, pred in enumerate(predictions['predictions'], 1):
                top_tag = pred['tags'][0]
                print(f"  Event {i}: {top_tag['tag']} ({top_tag['confidence']:.3f})")
        else:
            print(f"❌ Failed with status {response.status_code}")
        
        print("\n" + "=" * 80)
        print("✅ All API tests completed!")
        print("=" * 80)
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error testing API: {e}")
        sys.exit(1)
    
    finally:
        # Stop server
        print("\nStopping server...")
        server_process.terminate()
        server_process.wait(timeout=5)
        print("Server stopped.")


if __name__ == "__main__":
    test_api()

