"""Tools for the LangGraph agent."""
import httpx
from typing import List, Dict, Any
from langchain_core.tools import tool
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


@tool
def search_serper(query: str) -> Dict[str, Any]:
    """
    Search the web using Serper API to find relevant URLs and information.
    
    Args:
        query: The search query string
        
    Returns:
        Dictionary containing search results with URLs, titles, and snippets
    """
    from config import settings
    
    try:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": settings.serper_api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
            "num": 10  # Number of results to return
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant information
            results = {
                "query": query,
                "organic_results": [],
                "answer_box": None,
                "knowledge_graph": None
            }
            
            # Extract organic search results
            if "organic" in data:
                for item in data["organic"]:
                    results["organic_results"].append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })
            
            # Extract answer box if available
            if "answerBox" in data:
                results["answer_box"] = {
                    "answer": data["answerBox"].get("answer", ""),
                    "snippet": data["answerBox"].get("snippet", ""),
                    "title": data["answerBox"].get("title", "")
                }
            
            # Extract knowledge graph if available
            if "knowledgeGraph" in data:
                results["knowledge_graph"] = {
                    "title": data["knowledgeGraph"].get("title", ""),
                    "description": data["knowledgeGraph"].get("description", "")
                }
            
            logger.info(f"Serper search completed for query: {query}")
            return results
            
    except Exception as e:
        logger.error(f"Error in Serper search: {str(e)}")
        return {
            "query": query,
            "error": str(e),
            "organic_results": []
        }


@tool
def scrape_web_page(url: str) -> Dict[str, Any]:
    """
    Scrape content from a web page URL.
    
    Args:
        url: The URL of the web page to scrape
        
    Returns:
        Dictionary containing scraped content including title, text, and metadata
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            
            # Extract title
            title = ""
            if soup.title:
                title = soup.title.string
            elif soup.find("meta", property="og:title"):
                title = soup.find("meta", property="og:title").get("content", "")
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            
            # Extract main content
            # Try to find main content areas
            main_content = ""
            for selector in ["main", "article", "[role='main']", ".content", "#content"]:
                main_tag = soup.select_one(selector)
                if main_tag:
                    main_content = main_tag.get_text(separator=" ", strip=True)
                    break
            
            # If no main content found, get body text
            if not main_content:
                main_content = soup.get_text(separator=" ", strip=True)
            
            # Clean up whitespace
            main_content = " ".join(main_content.split())
            
            # Limit content length to avoid token limits
            max_length = 10000
            if len(main_content) > max_length:
                main_content = main_content[:max_length] + "..."
            
            # Extract meta description
            description = ""
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc:
                description = meta_desc.get("content", "")
            elif soup.find("meta", property="og:description"):
                description = soup.find("meta", property="og:description").get("content", "")
            
            result = {
                "url": url,
                "title": title.strip() if title else "No title found",
                "description": description.strip() if description else "",
                "content": main_content,
                "status": "success"
            }
            
            logger.info(f"Successfully scraped: {url}")
            return result
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error scraping {url}: {str(e)}")
        return {
            "url": url,
            "error": f"HTTP {e.response.status_code}: {str(e)}",
            "status": "error"
        }
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return {
            "url": url,
            "error": str(e),
            "status": "error"
        }
