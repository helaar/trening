"""CLI scripts for development and deployment."""
import sys
import json
import signal
import socket
import uvicorn
import requests


_DEV_PORT = 5175


def _kill_port(port: int) -> None:
    import os
    import subprocess
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True, text=True
    )
    pids = result.stdout.strip().split()
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
            print(f"Killed process {pid} occupying port {port}")
        except (ProcessLookupError, ValueError):
            pass


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _wait_for_port_free(port: int, timeout: float = 5.0) -> None:
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _port_in_use(port):
            return
        time.sleep(0.2)
    print(f"Warning: port {port} still in use after {timeout}s, proceeding anyway")


def dev_server():
    """Run development server with auto-reload."""
    if _port_in_use(_DEV_PORT):
        print(f"Port {_DEV_PORT} already in use — killing existing process...")
        _kill_port(_DEV_PORT)
        _wait_for_port_free(_DEV_PORT)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=_DEV_PORT,
        reload=True,
        log_level="info"
    )


def start_task():
    """Test the asynchronous task API."""
    
    def create_task(base_url: str = "http://localhost:5175"):
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
    
    def get_task_status(task_id: str, base_url: str = "http://localhost:5175"):
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
