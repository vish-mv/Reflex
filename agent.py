"""LangGraph agent setup with OpenAI."""
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from tools import search_serper, scrape_web_page
from config import settings
import logging

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State of the agent during execution."""
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Global agent instance (created once)
_agent_instance = None


def create_agent():
    """Create and configure the LangGraph agent."""
    global _agent_instance
    
    if _agent_instance is not None:
        return _agent_instance
    
    # Initialize OpenAI model
    llm = ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=0,
        api_key=settings.openai_api_key
    )
    
    # Bind tools to the model
    tools = [search_serper, scrape_web_page]
    llm_with_tools = llm.bind_tools(tools)
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    # Define the agent node
    def agent_node(state: AgentState):
        """Agent node that processes messages and decides on actions."""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Define the router function
    def should_continue(state: AgentState) -> str:
        """Determine if we should continue or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls, continue to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        # Otherwise, end
        return END
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile the graph with memory
    memory = MemorySaver()
    _agent_instance = workflow.compile(checkpointer=memory)
    
    return _agent_instance


def run_agent(question: str, thread_id: str = "default") -> tuple[str, list[dict]]:
    """
    Run the agent with a question and return the final answer along with full conversation history.
    
    Args:
        question: The user's question
        thread_id: Thread ID for conversation continuity
        
    Returns:
        Tuple of (final_answer, conversation_history)
    """
    try:
        agent = create_agent()
        
        # System prompt
        system_prompt = """You are a helpful research assistant. When given a question:
1. Use the search_serper tool to find relevant URLs and information
2. Use the scrape_web_page tool to get detailed content from the most relevant URLs (scrape 2-3 most relevant URLs)
3. Analyze the information and provide a comprehensive answer with references
4. Always cite your sources by including URLs in your response
5. Format your response clearly with proper structure
6. Be concise but thorough in your analysis"""
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Check if there's existing state for this thread
        has_existing_state = False
        try:
            existing_state = agent.get_state(config)
            existing_messages = existing_state.values.get("messages", [])
            has_existing_state = len(existing_messages) > 0
        except Exception:
            # No existing state - this is a new conversation
            has_existing_state = False
        
        # Prepare messages to add
        # LangGraph will automatically load previous state and append new messages
        if has_existing_state:
            # Existing conversation - just add the new question
            # System message should already be in the state
            messages_to_add = [HumanMessage(content=question)]
        else:
            # New conversation - add system message and question
            messages_to_add = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ]
        
        # Invoke the agent
        # LangGraph automatically loads previous state when thread_id exists
        # and merges it with new messages
        result = agent.invoke({"messages": messages_to_add}, config)
        
        # Extract all messages for history
        all_messages = result.get("messages", [])
        
        # Extract the final answer from the last AI message
        final_message = None
        
        # Find the last AI message that doesn't have tool calls
        for msg in reversed(all_messages):
            if isinstance(msg, AIMessage):
                # Check if this message has tool calls
                has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
                if not has_tool_calls and msg.content:
                    final_message = msg.content
                    break
        
        if not final_message:
            # If no final message found, get the last AI message with content
            for msg in reversed(all_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    final_message = msg.content
                    break
        
        if not final_message:
            logger.warning("No final message content found, returning default message")
            final_message = "I apologize, but I couldn't generate a complete response. Please try rephrasing your question."
        
        # Format conversation history
        conversation_history = []
        for msg in all_messages:
            if isinstance(msg, SystemMessage):
                continue  # Skip system messages in history
            elif isinstance(msg, HumanMessage):
                conversation_history.append({
                    "role": "user",
                    "content": msg.content,
                    "timestamp": getattr(msg, "timestamp", None)
                })
            elif isinstance(msg, AIMessage):
                # Only include final answers, not intermediate tool-calling messages
                if msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                    conversation_history.append({
                        "role": "assistant",
                        "content": msg.content,
                        "timestamp": getattr(msg, "timestamp", None)
                    })
        
        return final_message, conversation_history
        
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}", exc_info=True)
        raise


def get_chat_history(thread_id: str) -> list[dict]:
    """
    Get chat history for a specific thread.
    
    Args:
        thread_id: Thread ID to retrieve history for
        
    Returns:
        List of conversation messages
    """
    try:
        agent = create_agent()
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get state from checkpointer
        state = agent.get_state(config)
        messages = state.values.get("messages", [])
        
        # Format conversation history
        conversation_history = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                continue  # Skip system messages
            elif isinstance(msg, HumanMessage):
                conversation_history.append({
                    "role": "user",
                    "content": msg.content,
                    "timestamp": getattr(msg, "timestamp", None)
                })
            elif isinstance(msg, AIMessage):
                # Only include final answers
                if msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                    conversation_history.append({
                        "role": "assistant",
                        "content": msg.content,
                        "timestamp": getattr(msg, "timestamp", None)
                    })
        
        return conversation_history
        
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}", exc_info=True)
        return []
