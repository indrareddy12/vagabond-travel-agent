import json
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from config import logger, OPENAI_API_KEY
from vector_store import initialize_vector_store, check_city_in_store
from tools import search_web_tool, fetch_weather_forecast, fetch_images_tool

vector_db = initialize_vector_store()

class TravelAssistantOutput(BaseModel):
    city_name: str = Field(description="Name of the city")
    city_summary: str = Field(description="Summary of the city's key points and history")
    weather_forecast: List[Dict[str, Any]] = Field(description="7-day weather forecast details")
    image_urls: List[str] = Field(description="High-quality image URLs of the city")

class TravelState(TypedDict):
    query: str
    city: str
    skip_summary: bool
    city_summary: str
    is_stored_city: bool
    weather_forecast: List[Dict[str, Any]]
    image_urls: List[str]
    messages: List[Any]
    final_output: Optional[Dict[str, Any]]

def initialize_node(state: TravelState) -> TravelState:
    logger.info("Initializing query execution...")
    query = state.get("query", "").strip()
    previous_city = state.get("city", "")
    
    is_followup = False
    extracted_city = ""
    
    query_lower = query.lower()
    followup_keywords = ["next week", "forecast", "weather", "tomorrow", "images", "photos", "pictures", "update weather"]
    
    # Check if query is a follow-up preserving previous context
    if previous_city and any(kw in query_lower for kw in followup_keywords):
        common_cities = ["tokyo", "paris", "new york", "london", "sydney", "kyoto", "snohomish"]
        if not any(city in query_lower for city in common_cities if city != previous_city.lower()):
            is_followup = True
            extracted_city = previous_city
            logger.info(f"Memory check: follow-up query. Retaining city context: '{previous_city}'")

    if not is_followup:
        known_cities = ["tokyo", "paris", "new york", "london", "sydney", "kyoto", "snohomish"]
        for known in known_cities:
            if known in query_lower:
                extracted_city = known.title()
                break
        
        if not extracted_city and OPENAI_API_KEY:
            try:
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
                prompt = (
                    "Extract the city name from the query. "
                    "Return ONLY the city name in Title Case. If none is found, return empty.\n\n"
                    f"Query: {query}"
                )
                response = llm.invoke(prompt)
                extracted_city = response.content.strip().title()
            except Exception as e:
                logger.error(f"LLM extraction failed: {e}")
                
        if not extracted_city:
            cap_words = [w.strip("?,.!") for w in query.split() if w and w[0].isupper() and w.lower() not in ["i", "the", "a", "an"]]
            if cap_words:
                extracted_city = cap_words[0].title()
            else:
                stopwords = {"tell", "me", "about", "what", "is", "where", "the", "a", "an", "info", "on", "search", "for", "please", "query", "city"}
                query_words = [w.strip("?,.!") for w in query.split() if w.lower() not in stopwords]
                extracted_city = query_words[0].title() if query_words else "Tokyo"
            
        logger.info(f"Target city resolved to: '{extracted_city}'")
        
    return {
        "city": extracted_city,
        "skip_summary": is_followup,
        "city_summary": state.get("city_summary", "") if is_followup else "",
        "weather_forecast": state.get("weather_forecast", []) if is_followup else [],
        "image_urls": state.get("image_urls", []) if is_followup else [],
        "messages": state.get("messages", []) + [HumanMessage(content=query)]
    }

def route_retrieval(state: TravelState) -> str:
    if state.get("skip_summary", False):
        return "skip_retrieval"
        
    city = state.get("city", "")
    is_present, _ = check_city_in_store(city, vector_db)
    
    if is_present:
        return "vector_retrieve"
    else:
        return "web_search"

def vector_retrieve_node(state: TravelState) -> TravelState:
    logger.info("Retrieving city details from local vector store...")
    city = state.get("city", "")
    _, summary = check_city_in_store(city, vector_db)
    return {
        "city_summary": summary,
        "is_stored_city": True
    }

def web_search_node(state: TravelState) -> TravelState:
    logger.info("City not in local store. Executing web search fallback...")
    city = state.get("city", "")
    summary = search_web_tool(f"tell me about {city}")
    return {
        "city_summary": summary,
        "is_stored_city": False
    }

def skip_retrieval_node(state: TravelState) -> TravelState:
    logger.info("Skipping city description load due to follow-up query.")
    return {}

def trigger_parallel_fetch_node(state: TravelState) -> TravelState:
    return {}

def fetch_weather_node(state: TravelState) -> TravelState:
    logger.info("Executing weather query...")
    city = state.get("city", "")
    try:
        forecast = fetch_weather_forecast(city)
    except Exception as e:
        logger.error(f"Weather fetch failed for {city}: {e}")
        forecast = []
    return {
        "weather_forecast": forecast
    }

def fetch_images_node(state: TravelState) -> TravelState:
    logger.info("Executing image retrieval query...")
    city = state.get("city", "")
    try:
        images = fetch_images_tool(city)
    except Exception as e:
        logger.error(f"Image fetch failed for {city}: {e}")
        images = []
    return {
        "image_urls": images
    }

def generate_structured_output_node(state: TravelState) -> TravelState:
    logger.info("Compiling structured travel assistant response...")
    city = state.get("city", "")
    summary = state.get("city_summary", "")
    weather = state.get("weather_forecast", [])
    images = state.get("image_urls", [])
    
    refined_summary = summary
    if OPENAI_API_KEY:
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY)
            prompt = (
                "Synthesize a brief, captivating travel overview (3-4 sentences max) for a tourist guide. "
                "Incorporate key elements from this raw city information:\n\n"
                f"City: {city}\n"
                f"Information: {summary}\n\n"
                "Return only the synthesized overview paragraph."
            )
            response = llm.invoke(prompt)
            refined_summary = response.content.strip()
        except Exception as e:
            logger.error(f"Refining summary with OpenAI failed: {e}")
            
    output_obj = TravelAssistantOutput(
        city_name=city,
        city_summary=refined_summary,
        weather_forecast=weather,
        image_urls=images
    )
    
    final_dict = output_obj.model_dump()
    
    tool_msg = AIMessage(
        content=f"Successfully compiled structured travel summary for {city}.",
        additional_kwargs={"structured_output": final_dict}
    )
    
    return {
        "final_output": final_dict,
        "messages": state.get("messages", []) + [tool_msg]
    }

def compile_agent_graph():
    workflow = StateGraph(TravelState)
    
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("vector_retrieve", vector_retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("skip_retrieval", skip_retrieval_node)
    workflow.add_node("trigger_parallel", trigger_parallel_fetch_node)
    workflow.add_node("fetch_weather", fetch_weather_node)
    workflow.add_node("fetch_images", fetch_images_node)
    workflow.add_node("generate_structured_output", generate_structured_output_node)
    
    workflow.set_entry_point("initialize")
    
    workflow.add_conditional_edges(
        "initialize",
        route_retrieval,
        {
            "vector_retrieve": "vector_retrieve",
            "web_search": "web_search",
            "skip_retrieval": "skip_retrieval"
        }
    )
    
    workflow.add_edge("vector_retrieve", "trigger_parallel")
    workflow.add_edge("web_search", "trigger_parallel")
    workflow.add_edge("skip_retrieval", "trigger_parallel")
    
    # Parallel Fan-Out
    workflow.add_edge("trigger_parallel", "fetch_weather")
    workflow.add_edge("trigger_parallel", "fetch_images")
    
    # Parallel Fan-In
    workflow.add_edge("fetch_weather", "generate_structured_output")
    workflow.add_edge("fetch_images", "generate_structured_output")
    
    workflow.add_edge("generate_structured_output", END)
    
    return workflow
