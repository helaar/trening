"""CLI scripts for development and deployment."""
import uvicorn


def dev_server():
    """Run development server with auto-reload."""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    dev_server()
