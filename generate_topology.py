import os
import networkx as nx
import matplotlib.pyplot as plt
from config import logger

def draw_graph():
    logger.info("Generating Lucidchart-style LangGraph topology png...")
    
    G = nx.DiGraph()
    
    # Simple labels, clean and concise
    nodes = {
        "START": {"label": "START", "type": "endpoint"},
        "initialize": {"label": "initialize_node\n(Extract query intent)", "type": "process"},
        "vector_retrieve": {"label": "vector_retrieve_node\n(Local FAISS DB)", "type": "process"},
        "skip_retrieval": {"label": "skip_retrieval_node\n(Memory bypass)", "type": "process"},
        "web_search": {"label": "web_search_node\n(DuckDuckGo search)", "type": "process"},
        "trigger_parallel": {"label": "trigger_parallel_node\n(Sync bridge)", "type": "process"},
        "fetch_weather": {"label": "fetch_weather_node\n(Weather query)", "type": "process"},
        "fetch_images": {"label": "fetch_images_node\n(Image query)", "type": "process"},
        "generate_structured_output": {"label": "generate_structured_output_node\n(Pydantic validation)", "type": "process"},
        "END": {"label": "END", "type": "endpoint"}
    }
    
    # Balanced vertical layout coordinates
    pos = {
        "START": (2.0, 6.2),
        "initialize": (2.0, 5.1),
        
        "vector_retrieve": (0.6, 3.8),
        "skip_retrieval": (2.0, 3.8),
        "web_search": (3.4, 3.8),
        
        "trigger_parallel": (2.0, 2.7),
        
        "fetch_weather": (1.0, 1.6),
        "fetch_images": (3.0, 1.6),
        
        "generate_structured_output": (2.0, 0.5),
        "END": (2.0, -0.5)
    }
    
    edges = [
        ("START", "initialize"),
        ("initialize", "vector_retrieve"),
        ("initialize", "skip_retrieval"),
        ("initialize", "web_search"),
        ("vector_retrieve", "trigger_parallel"),
        ("skip_retrieval", "trigger_parallel"),
        ("web_search", "trigger_parallel"),
        ("trigger_parallel", "fetch_weather"),
        ("trigger_parallel", "fetch_images"),
        ("fetch_weather", "generate_structured_output"),
        ("fetch_images", "generate_structured_output"),
        ("generate_structured_output", "END")
    ]
    
    G.add_nodes_from(nodes.keys())
    G.add_edges_from(edges)
    
    # Clean white canvas (standard for documentation)
    fig = plt.figure(figsize=(10, 8), dpi=150, facecolor="#FFFFFF")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#FFFFFF")
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Draw straight, clean dark-gray arrows (matching Draw.io defaults)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        edge_color="#475569",
        width=1.5,
        arrows=True,
        arrowsize=14,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0", # Completely straight lines
        min_source_margin=18,
        min_target_margin=18,
        ax=ax
    )
    
    # Draw nodes as clean white boxes with dark slate borders
    for node, data in nodes.items():
        node_type = data["type"]
        label_text = data["label"]
        
        if node_type == "endpoint":
            # Pill shape for Start/End
            box_style = "round,pad=0.5"
            fc = "#F1F5F9"
            ec = "#475569"
        else:
            # Standard rectangles for processes
            box_style = "square,pad=0.6"
            fc = "#FFFFFF"
            ec = "#334155"
            
        ax.text(
            pos[node][0], pos[node][1],
            label_text,
            ha="center", va="center",
            fontsize=8.5,
            color="#0F172A",
            bbox=dict(
                boxstyle=box_style,
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.2,
                alpha=1.0
            )
        )
        
    plt.xlim(-0.2, 4.2)
    plt.ylim(-0.9, 6.7)
    plt.axis("off")
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph.png")
    plt.savefig(output_path, bbox_inches="tight", facecolor="#FFFFFF", pad_inches=0.2)
    plt.close()
    
    logger.info(f"Lucidchart-style topology graph saved to: {output_path}")

if __name__ == "__main__":
    draw_graph()
