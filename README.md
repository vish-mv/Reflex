# Research Agent API

A FastAPI backend with a LangGraph agent powered by OpenAI that can search the web using Serper API and scrape web pages to provide comprehensive answer reports with references.

## Features

- **LangGraph Agent**: Intelligent agent that orchestrates multiple tools
- **Serper API Integration**: Web search capabilities
- **Web Scraping**: Extract content from relevant web pages
- **OpenAI Powered**: Uses GPT-4 for intelligent reasoning and answer generation
- **Reference Citations**: Returns answers with proper source references
- **Chat History**: Maintains conversation context across multiple interactions using thread IDs

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure environment variables**:
   - Copy `.env.example` to `.env`
   - Add your API keys:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `SERPER_API_KEY`: Your Serper API key (get it from https://serper.dev)

3. **Run the server**:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST `/answer`

Submit a research question and get a comprehensive answer with references. Chat history is automatically maintained when using the same `thread_id`.

**Request Body**:
```json
{
  "question": "What are the latest developments in quantum computing?",
  "thread_id": "optional-thread-id"
}
```

**Response**:
```json
{
  "answer": "Comprehensive answer with references...",
  "question": "What are the latest developments in quantum computing?",
  "thread_id": "optional-thread-id",
  "chat_history": [
    {
      "role": "user",
      "content": "What are the latest developments in quantum computing?",
      "timestamp": null
    },
    {
      "role": "assistant",
      "content": "Comprehensive answer with references...",
      "timestamp": null
    }
  ]
}
```

### GET `/chat-history/{thread_id}`

Retrieve the full chat history for a specific thread.

**Response**:
```json
{
  "thread_id": "thread-123",
  "messages": [
    {
      "role": "user",
      "content": "First question",
      "timestamp": null
    },
    {
      "role": "assistant",
      "content": "First answer",
      "timestamp": null
    },
    {
      "role": "user",
      "content": "Follow-up question",
      "timestamp": null
    },
    {
      "role": "assistant",
      "content": "Follow-up answer",
      "timestamp": null
    }
  ],
  "total_messages": 4
}
```

### GET `/health`

Health check endpoint.

### GET `/`

Root endpoint with API information.

## How It Works

1. User submits a question via the `/answer` endpoint (optionally with a `thread_id` for conversation continuity)
2. The LangGraph agent receives the question and loads any previous conversation history for that thread
3. Agent uses Serper API to search for relevant keywords
4. Agent scrapes the most relevant web pages
5. Agent analyzes the information and generates a comprehensive answer
6. Answer is returned with proper references, citations, and full chat history
7. Subsequent questions with the same `thread_id` maintain conversation context

## Tools

The agent has access to two main tools:

1. **search_serper**: Searches the web using Serper API and returns relevant URLs, snippets, and information
2. **scrape_web_page**: Scrapes content from a given URL, extracting title, description, and main content

## Environment Variables

- `OPENAI_API_KEY`: Required - Your OpenAI API key
- `SERPER_API_KEY`: Required - Your Serper API key
- `PORT`: Optional - Server port (default: 8000)
- `HOST`: Optional - Server host (default: 0.0.0.0)

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Example Usage

### Basic Question

```python
import requests

response = requests.post(
    "http://localhost:8000/answer",
    json={
        "question": "What are the benefits of renewable energy?"
    }
)

print(response.json()["answer"])
```

### Conversation with Chat History

```python
import requests

# Use a consistent thread_id to maintain conversation context
thread_id = "user-123"

# First question
response1 = requests.post(
    "http://localhost:8000/answer",
    json={
        "question": "What is quantum computing?",
        "thread_id": thread_id
    }
)
print("Answer 1:", response1.json()["answer"])

# Follow-up question (agent remembers previous context)
response2 = requests.post(
    "http://localhost:8000/answer",
    json={
        "question": "What are its practical applications?",
        "thread_id": thread_id
    }
)
print("Answer 2:", response2.json()["answer"])

# Retrieve full chat history
history = requests.get(f"http://localhost:8000/chat-history/{thread_id}")
print("Chat History:", history.json()["messages"])
```

## Notes

- The agent uses GPT-4-turbo-preview model for best results
- Web scraping respects robots.txt and uses proper headers
- Content is limited to 10,000 characters per page to manage token limits
- **Chat History**: The agent maintains conversation context using thread IDs. Use the same `thread_id` across multiple requests to maintain context. If no `thread_id` is provided, a default thread is used.
- Chat history is stored in memory (using LangGraph's MemorySaver). For production, consider using a persistent storage solution.
