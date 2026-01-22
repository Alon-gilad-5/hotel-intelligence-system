"""
Gradio Chat Interface for Databricks - With Async Support for Long Jobs

Copy this code to your Databricks notebook.
Handles 15+ minute ML analysis without timing out.
"""

import sys
sys.path.insert(0, "/Workspace/Users/alon-gilad@campus.technion.ac.il/hotel-intelligence-system")
sys.path.insert(0, "/Workspace/Users/alon-gilad@campus.technion.ac.il/hotel-intelligence-system/agents")

import builtins
builtins.dbutils = dbutils  # Required for Databricks

import gradio as gr
import base64
import threading
import time
from io import BytesIO
from PIL import Image as PILImage
from agents.coordinator import LangGraphCoordinator

# ==============================================
# CONFIGURATION
# ==============================================

HOTEL_ID = "ABB_40458495"
HOTEL_NAME = "Rental unit in Broadbeach"  
CITY = "Broadbeach"

coordinator = LangGraphCoordinator(HOTEL_ID, HOTEL_NAME, CITY)
state = coordinator.get_initial_state()

# Global storage
captured_charts = []
current_job = {
    "running": False,
    "progress": "",
    "result": None,
    "error": None,
    "start_time": None
}

# ==============================================
# CHART CAPTURE PATCH
# ==============================================

try:
    from databricks_tools import run_lr_analysis as _orig_lr
    def patched_lr(hotel_id, timeout_seconds=1200):
        global captured_charts
        result = _orig_lr(hotel_id, timeout_seconds)
        if result.get("charts"):
            captured_charts = []
            for name, b64 in result["charts"].items():
                if b64:
                    try:
                        img = PILImage.open(BytesIO(base64.b64decode(b64)))
                        captured_charts.append(img)
                    except:
                        pass
        return result
    import databricks_tools
    databricks_tools.run_lr_analysis = patched_lr
    print("[OK] Chart capture enabled")
except Exception as e:
    print(f"[Note] Chart capture not available: {e}")

# ==============================================
# ASYNC JOB HANDLING
# ==============================================

def run_analysis_thread(query: str):
    """Run analysis in background thread."""
    global state, captured_charts, current_job
    
    current_job["running"] = True
    current_job["progress"] = "Starting analysis..."
    current_job["result"] = None
    current_job["error"] = None
    current_job["start_time"] = time.time()
    captured_charts = []
    
    try:
        current_job["progress"] = "Routing to specialist agents..."
        response, state = coordinator.run(query, state)
        current_job["result"] = response
        current_job["progress"] = "Complete!"
    except Exception as e:
        current_job["error"] = str(e)
        current_job["progress"] = f"Error: {str(e)[:100]}"
    finally:
        current_job["running"] = False


def start_analysis(message: str, history: list):
    """Start analysis - returns immediately with 'processing' message."""
    global current_job
    
    if not message.strip():
        return "", history, [], ""
    
    # If already running, don't start another
    if current_job["running"]:
        elapsed = int(time.time() - current_job["start_time"]) if current_job["start_time"] else 0
        return "", history, [], f"⏳ Analysis in progress... ({elapsed}s)\n{current_job['progress']}"
    
    # Add user message to history
    history = history + [(message, "⏳ Processing... This may take 15-20 minutes for deep analysis.\n\nClick 'Check Status' to see progress.")]
    
    # Start background thread
    thread = threading.Thread(target=run_analysis_thread, args=(message,))
    thread.daemon = True
    thread.start()
    
    return "", history, [], "⏳ Analysis started..."


def check_status(history: list):
    """Check job status and update response when complete."""
    global current_job, captured_charts
    
    if not history:
        return history, [], "No analysis running"
    
    elapsed = int(time.time() - current_job["start_time"]) if current_job["start_time"] else 0
    
    if current_job["running"]:
        # Still running - show progress
        progress_bar = "█" * (elapsed // 30) + "░" * max(0, 20 - elapsed // 30)
        status = f"⏳ Running ({elapsed}s elapsed)\n[{progress_bar}]\n\n{current_job['progress']}"
        return history, captured_charts, status
    
    elif current_job["result"]:
        # Complete - update the last message
        if history and history[-1][1] and history[-1][1].startswith("⏳"):
            history[-1] = (history[-1][0], current_job["result"])
        status = f"✅ Complete! ({elapsed}s total)"
        return history, captured_charts, status
    
    elif current_job["error"]:
        # Error
        if history and history[-1][1] and history[-1][1].startswith("⏳"):
            history[-1] = (history[-1][0], f"❌ Error: {current_job['error']}")
        status = f"❌ Error after {elapsed}s"
        return history, captured_charts, status
    
    else:
        return history, captured_charts, "Ready - enter a question"


def clear_chat():
    """Clear chat and reset state."""
    global state, current_job, captured_charts
    state = coordinator.get_initial_state()
    current_job = {
        "running": False,
        "progress": "",
        "result": None,
        "error": None,
        "start_time": None
    }
    captured_charts = []
    return [], [], "Cleared - ready for new questions"


# ==============================================
# GRADIO INTERFACE
# ==============================================

with gr.Blocks(title=f"Hotel Intelligence: {HOTEL_NAME}") as demo:
    gr.Markdown(f"# 🏨 Hotel Intelligence: {HOTEL_NAME}")
    gr.Markdown("""
    *Supports long-running ML analysis (15-20 min). Click 'Check Status' to see progress.*
    
    **Quick queries** (< 1 min): Reviews, rankings, competitor info  
    **Deep analysis** (15-20 min): Feature impact, NLP comparison
    """)
    
    chatbot = gr.Chatbot(height=400)
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a question... (e.g., 'What features should I improve?')", 
            show_label=False, 
            scale=4
        )
        send_btn = gr.Button("🚀 Send", variant="primary")
    
    with gr.Row():
        check_btn = gr.Button("🔄 Check Status", variant="secondary")
        clear_btn = gr.Button("🗑️ Clear Chat", variant="stop")
    
    status_box = gr.Textbox(
        label="Status", 
        value="Ready - enter a question",
        interactive=False,
        lines=3
    )
    
    # Chart gallery
    chart_gallery = gr.Gallery(
        label="📊 Analysis Charts", 
        columns=2, 
        height=300, 
        visible=True
    )
    
    # Note: gr.Timer requires Gradio 4.x
    # For older versions, use manual "Check Status" button
    
    # Button handlers
    send_btn.click(
        fn=start_analysis,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, chart_gallery, status_box]
    )
    
    msg.submit(
        fn=start_analysis,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, chart_gallery, status_box]
    )
    
    check_btn.click(
        fn=check_status,
        inputs=[chatbot],
        outputs=[chatbot, chart_gallery, status_box]
    )
    
    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, chart_gallery, status_box]
    )
    
    gr.Markdown("""
    ---
    ### Example Questions
    - "What are guests saying about wifi?" (quick)
    - "How do I compare to competitors?" (quick)
    - "What features should I improve to increase my rating?" (deep - 15 min)
    
    *Charts appear above when running feature impact analysis*
    """)

demo.queue()
demo.launch(share=True)
