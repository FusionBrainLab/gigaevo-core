#!~/venvs/main/bin/python
"""
Start the DAG Builder API server.
"""

import uvicorn

if __name__ == "__main__":
    print("🚀 Starting GigaEvo DAG Builder API...")
    print("📱 Open http://localhost:8081 in your browser")
    print("🔗 API docs available at http://localhost:8081/docs")
    uvicorn.run("api:app", host="0.0.0.0", port=8081, reload=True)
