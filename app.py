import os
import uuid
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import config
from config import logger

st.set_page_config(
    page_title="Vagabond: Multi-Modal AI Travel Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark-themed page styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    .travel-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.25);
    }
    
    .section-title {
        color: #38BDF8;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        border-bottom: 2px solid rgba(56, 189, 248, 0.2);
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    
    .route-db {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #34D399;
        border-radius: 8px;
        padding: 8px 12px;
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 15px;
        display: inline-block;
    }
    .route-web {
        background-color: rgba(245, 158, 11, 0.15);
        border: 1px solid #F59E0B;
        color: #FBBF24;
        border-radius: 8px;
        padding: 8px 12px;
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 15px;
        display: inline-block;
    }
    .route-memory {
        background-color: rgba(139, 92, 246, 0.15);
        border: 1px solid #8B5CF6;
        color: #A78BFA;
        border-radius: 8px;
        padding: 8px 12px;
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 15px;
        display: inline-block;
    }
    
    .weather-metric {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    
    .weather-val {
        font-size: 20px;
        font-weight: bold;
        color: #F8FAFC;
    }
    
    .weather-lbl {
        font-size: 12px;
        color: #94A3B8;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🗺️ Vagabond: Multi-Modal Travel Assistant")
st.markdown("An advanced AI Agent built with **LangGraph** featuring intelligent database routing, parallel fetches, and context memory.")

@st.cache_resource
def get_compiled_graph():
    from agent import compile_agent_graph
    from langgraph.checkpoint.memory import MemorySaver
    
    builder = compile_agent_graph()
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph

def get_topology_image():
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph.png")
    if not os.path.exists(img_path):
        try:
            from generate_topology import draw_graph
            draw_graph()
        except Exception as e:
            logger.error(f"Could not generate topology: {e}")
            
    if os.path.exists(img_path):
        return Image.open(img_path)
    return None

with st.sidebar:
    st.header("⚙️ Configuration")
    api_mode = st.radio("API Mode", ["Mock APIs (Zero Keys Required)", "Live APIs (OpenAI/Tavily)"])
    
    if api_mode == "Live APIs (OpenAI/Tavily)":
        openai_key = st.text_input("OpenAI API Key", type="password", help="Used for LLM responses and embeddings.")
        tavily_key = st.text_input("Tavily API Key (Optional)", type="password", help="Fallback search provider. If not supplied, DuckDuckGo will be used.")
        
        if openai_key:
            config.OPENAI_API_KEY = openai_key
        if tavily_key:
            config.TAVILY_API_KEY = tavily_key
    else:
        config.OPENAI_API_KEY = ""
        config.TAVILY_API_KEY = ""
        
    st.markdown("---")
    st.subheader("🧠 Context & Memory (Checkpointer)")
    
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        
    thread_id = st.text_input("Active Thread ID", value=st.session_state.thread_id, 
                              help="Graph checkpoints are maintained for this ID. Modify it to clear history.")
    
    if st.button("Reset Thread Context"):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()
        
    st.markdown("---")
    st.subheader("📊 Graph Architecture")
    topo_img = get_topology_image()
    if topo_img:
        st.image(topo_img, caption="LangGraph Topology Diagram", use_container_width=True)
    else:
        st.warning("Could not render topology graph.png")

try:
    graph = get_compiled_graph()
except Exception as e:
    st.error(f"Failed to load LangGraph dependencies.\nError: {e}")
    st.stop()

st.markdown("### ⚡ Quick Queries")
col1, col2, col3, col4, col5 = st.columns(5)
quick_query = ""

with col1:
    if st.button("🇯🇵 Tokyo (Local DB Path)", use_container_width=True):
        quick_query = "Tell me about Tokyo"
with col2:
    if st.button("🇫🇷 Paris (Local DB Path)", use_container_width=True):
        quick_query = "Info on Paris"
with col3:
    if st.button("🇺🇸 New York (Local DB Path)", use_container_width=True):
        quick_query = "Tell me about New York"
with col4:
    if st.button("🌲 Snohomish (Web Search Path)", use_container_width=True):
        quick_query = "Where is Snohomish?"
with col5:
    if st.button("⏰ Time Travel Follow-up", use_container_width=True):
        quick_query = "What about next week?"

user_query = st.text_input("Search a city or ask a follow-up question:", value=quick_query or "", placeholder="e.g. Tell me about Kyoto")

if user_query:
    st.markdown("---")
    
    with st.spinner("Agent executing workflow..."):
        thread_config = {"configurable": {"thread_id": thread_id}}
        
        try:
            current_checkpoint = graph.get_state(thread_config)
            previous_state = current_checkpoint.values if current_checkpoint else {}
            
            input_payload = {
                "query": user_query,
                "city": previous_state.get("city", "")
            }
            
            result_state = graph.invoke(input_payload, config=thread_config)
            
            skip_retrieval = result_state.get("skip_summary", False)
            city_name = result_state.get("city", "Unknown City")
            is_db_city = result_state.get("is_stored_city", False)
            
            if skip_retrieval:
                st.markdown(f'<div class="route-memory">🔄 Thread Checkpointer: Preserved context for "{city_name}" (Skipped DB/Search Summary)</div>', unsafe_allow_html=True)
            elif is_db_city:
                st.markdown(f'<div class="route-db">🟢 Routed to Local DB: facts retrieved for "{city_name}" from local FAISS vector store</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="route-web">🟡 Routed to Web Search: City "{city_name}" not found in local DB. Called web search tools.</div>', unsafe_allow_html=True)
            
            output = result_state.get("final_output")
            if not output:
                st.error("Agent failed to output structured travel data.")
                st.stop()
                
            city_summary = output.get("city_summary", "")
            weather_forecast = output.get("weather_forecast", [])
            image_urls = output.get("image_urls", [])
            
            col_left, col_right = st.columns([3, 2])
            
            with col_left:
                st.markdown(f'<h3 class="section-title">📍 Exploring {city_name}</h3>', unsafe_allow_html=True)
                st.markdown(f'<div class="travel-card">{city_summary}</div>', unsafe_allow_html=True)
                
                st.markdown('<h3 class="section-title">📸 Photo Gallery</h3>', unsafe_allow_html=True)
                if image_urls:
                    cols = st.columns(len(image_urls))
                    for idx, img_url in enumerate(image_urls):
                        with cols[idx]:
                            st.image(img_url, use_container_width=True, caption=f"View of {city_name} #{idx+1}")
                else:
                    st.info("No images fetched for this destination.")
                    
            with col_right:
                st.markdown(f'<h3 class="section-title">☀️ Weather & Forecast</h3>', unsafe_allow_html=True)
                
                if weather_forecast:
                    today = weather_forecast[0]
                    
                    st.markdown(
                        f"""
                        <div class="travel-card">
                            <h4 style='text-align: center; margin-top:0; color:#38BDF8;'>Today's Outlook ({today['day']})</h4>
                            <h1 style='text-align: center; font-size:48px; margin: 10px 0;'>{today['temperature']}°C</h1>
                            <p style='text-align: center; font-weight: bold; font-size: 18px; color: #F1F5F9;'>{today['condition']}</p>
                            <div style='display: flex; justify-content: space-around; margin-top: 15px;'>
                                <div class="weather-metric">
                                    <div class="weather-val">{today['humidity']}%</div>
                                    <div class="weather-lbl">Humidity</div>
                                </div>
                                <div class="weather-metric">
                                    <div class="weather-val">{today['wind_speed']} km/h</div>
                                    <div class="weather-lbl">Wind Speed</div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("#### 📈 7-Day Temperature Trend")
                    df_weather = pd.DataFrame(weather_forecast)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_weather["day"],
                        y=df_weather["temperature"],
                        mode='lines+markers',
                        line=dict(color='#38BDF8', width=3),
                        marker=dict(size=8, color='#0284C7'),
                        name='Temperature'
                    ))
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=10, b=20),
                        font=dict(color='#94A3B8'),
                        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Temp (°C)"),
                        height=250
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    
                else:
                    st.info("Weather forecast data not available.")
                    
        except Exception as e:
            st.error(f"An error occurred during workflow execution: {e}")
            logger.exception("Error executing agent")
