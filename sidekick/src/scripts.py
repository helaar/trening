"""CLI scripts for development and deployment."""
import sys
import json
import uvicorn
import requests


def dev_server():
    """Run development server with auto-reload."""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


def start_task():
    """Test the asynchronous task API."""
    
    def create_task(base_url: str = "http://localhost:8000"):
        """Create a new task via the API."""
        url = f"{base_url}/api/v1/tasks"
        
        payload = {
            "task_type": "training_analysis",
            "parameters": {
                "date_range": {
                    "start": "2026-01-01",
                    "end": "2026-02-11"
                },
                "analysis_type": "comprehensive"
            }
        }
        
        print(f"🚀 Creating task at {url}...")
        
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 201:
                data = response.json()
                task_id = data["task_id"]
                
                print(f"\n✅ Task created successfully!")
                print(f"📝 Task ID: {task_id}")
                print(f"📊 Status: {data['status']}")
                print(f"📈 Progress: {data['progress'] * 100:.0f}%")
                print(f"\n🔗 Check status at:")
                print(f"   {base_url}/api/v1/tasks/{task_id}")
                
            else:
                print(f"\n❌ Failed to create task")
                print(f"Status code: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Could not connect to {base_url}")
            print(f"Make sure the server is running (use 'uv run dev')")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
    
    def get_task_status(task_id: str, base_url: str = "http://localhost:8000"):
        """Get task status via the API."""
        url = f"{base_url}/api/v1/tasks/{task_id}"
        
        print(f"🔍 Fetching task status from {url}...")
        
        try:
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"\n{'='*60}")
                print(f"Task Status")
                print(f"{'='*60}")
                print(f"Task ID: {data['task_id']}")
                print(f"Status: {data['status']}")
                print(f"Progress: {data['progress'] * 100:.1f}%")
                print(f"Created: {data['created_at']}")
                
                if data.get('started_at'):
                    print(f"Started: {data['started_at']}")
                if data.get('completed_at'):
                    print(f"Completed: {data['completed_at']}")
                if data.get('duration_seconds'):
                    print(f"Duration: {data['duration_seconds']:.2f}s")
                
                if data.get('error'):
                    print(f"\n❌ Error: {data['error']}")
                elif data.get('result'):
                    print(f"\n✅ Result:")
                    print(json.dumps(data['result'], indent=2))
                
            elif response.status_code == 404:
                print(f"\n❌ Task not found")
            elif response.status_code == 403:
                print(f"\n❌ Not authorized to access this task")
            else:
                print(f"\n❌ Failed to get task status")
                print(f"Status code: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Could not connect to {base_url}")
            print(f"Make sure the server is running (use 'uv run dev')")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  start-task create     # Create a new task")
        print("  start-task get <id>   # Get task status")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create":
        create_task()
    elif command == "get":
        if len(sys.argv) < 3:
            print("Error: task_id required")
            sys.exit(1)
        task_id = sys.argv[2]
        get_task_status(task_id)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    dev_server()
