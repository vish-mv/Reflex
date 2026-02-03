"""FastAPI backend with LangGraph agent endpoint."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
from agent import run_agent, get_chat_history
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Research Agent API",
    description="API endpoint for LangGraph agent with Serper and web scraping capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    """Request model for the question endpoint."""
    question: str = Field(..., description="The research question to answer", min_length=1)
    thread_id: Optional[str] = Field(None, description="Optional thread ID for conversation continuity")


class Message(BaseModel):
    """Message model for chat history."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")


class AnswerResponse(BaseModel):
    """Response model for the answer endpoint."""
    answer: str = Field(..., description="The answer report with references")
    question: str = Field(..., description="The original question")
    thread_id: str = Field(..., description="The thread ID used")
    chat_history: List[Message] = Field(default_factory=list, description="Full conversation history")


class ChatHistoryResponse(BaseModel):
    """Response model for chat history endpoint."""
    thread_id: str = Field(..., description="The thread ID")
    messages: List[Message] = Field(..., description="List of conversation messages")
    total_messages: int = Field(..., description="Total number of messages")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Research Agent API",
        "version": "1.0.0",
        "endpoints": {
            "/answer": "POST - Submit a research question",
            "/chat-history/{thread_id}": "GET - Get chat history for a thread",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/answer", response_model=AnswerResponse)
async def get_answer(request: QuestionRequest):
    """
    Main endpoint to submit a research question.
    
    The agent will:
    1. Search for relevant information using Serper API
    2. Scrape relevant web pages
    3. Generate a comprehensive answer report with references
    
    Chat history is automatically maintained using the thread_id.
    If no thread_id is provided, a default thread is used.
    """
    try:
        thread_id = request.thread_id or "default"
        
        logger.info(f"Processing question for thread {thread_id}: {request.question[:100]}...")
        
        # Run the agent (returns answer and full conversation history)
        answer, chat_history = run_agent(request.question, thread_id)
        
        logger.info(f"Answer generated successfully for thread {thread_id}")
        
        # Convert chat history to Message models
        messages = [
            Message(role=msg["role"], content=msg["content"], timestamp=msg.get("timestamp"))
            for msg in chat_history
        ]
        
        return AnswerResponse(
            answer=answer,
            question=request.question,
            thread_id=thread_id,
            chat_history=messages
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.get("/chat-history/{thread_id}", response_model=ChatHistoryResponse)
async def get_chat_history_endpoint(thread_id: str):
    """
    Get chat history for a specific thread.
    
    Args:
        thread_id: The thread ID to retrieve history for
        
    Returns:
        Chat history with all messages in the conversation
    """
    try:
        logger.info(f"Retrieving chat history for thread: {thread_id}")
        
        # Get chat history from agent
        history = get_chat_history(thread_id)
        
        # Convert to Message models
        messages = [
            Message(role=msg["role"], content=msg["content"], timestamp=msg.get("timestamp"))
            for msg in history
        ]
        
        return ChatHistoryResponse(
            thread_id=thread_id,
            messages=messages,
            total_messages=len(messages)
        )
        
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving chat history: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
