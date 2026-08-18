import time
import random
from typing import List, Dict, Any, Optional
from langchain_community.tools import DuckDuckGoSearchRun
from config import logger, TAVILY_API_KEY, UNSPLASH_ACCESS_KEY, OPENWEATHER_API_KEY

CURATED_IMAGES = {
    "tokyo": [
        "https://images.unsplash.com/photo-1503899036084-c55cdd92da26?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1540959733332-eab4deceeaf7?auto=format&fit=crop&w=800&q=80"
    ],
    "paris": [
        "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1499856871958-5b9627545d1a?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1509060464153-44667396260f?auto=format&fit=crop&w=800&q=80"
    ],
    "new york": [
        "https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1522083165195-342750297f05?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1518391846015-55a9cc003b25?auto=format&fit=crop&w=800&q=80"
    ],
    "kyoto": [
        "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1545569341-9eb8b30979d9?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1490806843957-31f4c9a91c65?auto=format&fit=crop&w=800&q=80"
    ],
    "sydney": [
        "https://images.unsplash.com/photo-1506973035872-a4ec16b8e8d9?auto=format&fit=crop&w=800&q=80",
        "https://images.unsplash.com/photo-1524820197278-540916411e20?auto=format&fit=crop&w=800&q=80"
    ]
}

GENERIC_TRAVEL_IMAGES = [
    "https://images.unsplash.com/photo-1488646953014-85cb44e25828?auto=format&fit=crop&w=800&q=80",
    "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?auto=format&fit=crop&w=800&q=80",
    "https://images.unsplash.com/photo-1501785888041-af3ef285b470?auto=format&fit=crop&w=800&q=80"
]

def search_web_tool(query: str) -> str:
    """
    Search the web using DuckDuckGo with local fallback.
    """
    logger.info(f"Initiating web search tool for query: '{query}'")
    
    try:
        ddg = DuckDuckGoSearchRun()
        search_result = ddg.run(query)
        if search_result and len(search_result.strip()) > 50:
            logger.info("DuckDuckGo search returned successful result.")
            return search_result
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}. Using local dictionary fallback...")

    # Introduce simulated latency to mimic API delay
    time.sleep(0.5)
    
    city_name = query.replace("tell me about", "").replace("info", "").replace("search", "").strip().title()
    mock_responses = {
        "Snohomish": (
            "Snohomish is a city in Snohomish County, Washington, United States. The population was 9,098 at the 2010 census. "
            "It is located on the Snohomish River, southeast of Everett and northeast of Seattle. Snohomish is known for its historic "
            "downtown district, which is full of antique shops, historic houses built in the late 19th and early 20th centuries, "
            "and a charming historic atmosphere. It is frequently referred to as the 'Antique Capital of the Northwest'."
        ),
        "Sydney": (
            "Sydney is the state capital of New South Wales and the most populous city in Australia. Located on Australia's east coast, "
            "the metropolis surrounds Port Jackson and extends about 70 km on its periphery. The city is home to the Sydney Opera House "
            "and the Sydney Harbour Bridge, two of the most iconic structures on Earth. It has a warm, temperate climate and is "
            "renowned for its stunning beaches (like Bondi Beach), vibrant culinary culture, and beautiful harbor cruises."
        )
    }
    
    for key, val in mock_responses.items():
        if key.lower() in query.lower():
            return val
            
    return (
        f"{city_name} is a popular travel destination known for its cultural landmarks, historical significance, and local architecture. "
        f"Travel guides highlight its vibrant culinary scene, diverse neighborhoods, and rich arts culture. It remains a key center "
        f"for both regional travel and international visitors seeking unique local experiences."
    )

def fetch_weather_forecast(city: str) -> List[Dict[str, Any]]:
    """
    Fetch a 7-day weather forecast with deterministic simulation.
    """
    logger.info(f"Fetching weather forecast for: '{city}'")
    
    # Simulate network latency
    time.sleep(random.uniform(0.3, 0.7))
    
    # Deterministic generation based on city name to ensure consistency on refresh
    seed = sum(ord(c) for c in city.lower())
    random.seed(seed)
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Thunderstorm", "Windy"]
    
    city_lower = city.lower()
    if "tokyo" in city_lower or "kyoto" in city_lower:
        base_temp = 24
    elif "paris" in city_lower or "london" in city_lower:
        base_temp = 18
    elif "new york" in city_lower:
        base_temp = 22
    elif "sydney" in city_lower:
        base_temp = 16
    else:
        base_temp = random.randint(15, 30)
        
    forecast = []
    for i, day in enumerate(days):
        temp_diff = random.randint(-4, 5)
        temp = base_temp + temp_diff
        humidity = random.randint(50, 85)
        wind = round(random.uniform(5.0, 20.0), 1)
        
        if temp > 25:
            cond = random.choice(["Sunny", "Sunny", "Partly Cloudy"])
        elif temp < 15:
            cond = random.choice(["Cloudy", "Rainy", "Windy"])
        else:
            cond = random.choice(conditions)
            
        forecast.append({
            "day": day,
            "temperature": temp,
            "condition": cond,
            "humidity": humidity,
            "wind_speed": wind
        })
        
    return forecast

def fetch_images_tool(city: str) -> List[str]:
    """
    Retrieve curated city image URLs.
    """
    logger.info(f"Retrieving images for: '{city}'")
    
    time.sleep(random.uniform(0.2, 0.5))
    city_lower = city.lower().strip()
    
    for key, urls in CURATED_IMAGES.items():
        if key in city_lower or city_lower in key:
            return urls
            
    # Fallback to random signatures on high-quality stock templates
    dynamic_images = [
        f"https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?auto=format&fit=crop&w=800&q=80&sig={random.randint(1,100)}",
        f"https://images.unsplash.com/photo-1501785888041-af3ef285b470?auto=format&fit=crop&w=800&q=80&sig={random.randint(101,200)}",
        f"https://images.unsplash.com/photo-1488646953014-85cb44e25828?auto=format&fit=crop&w=800&q=80&sig={random.randint(201,300)}"
    ]
    return dynamic_images
