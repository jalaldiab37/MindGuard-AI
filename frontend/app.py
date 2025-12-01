"""
MindGuard AI - Streamlit Frontend
Interactive chat interface for mental health text classification.
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="MindGuard AI",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #e8e8e8;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    h1, h2, h3, h4, h5 {
        font-family: 'Inter', sans-serif;
        color: #ffffff !important;
    }
    
    .stMarkdown h3 {
        color: #ffffff !important;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #00d9ff, #00ff88, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .risk-card {
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .risk-normal {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(76, 175, 80, 0.1));
        border-left: 4px solid #4CAF50;
    }
    
    .risk-mild {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.2), rgba(255, 193, 7, 0.1));
        border-left: 4px solid #FFC107;
    }
    
    .risk-high {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.2), rgba(255, 152, 0, 0.1));
        border-left: 4px solid #FF9800;
    }
    
    .risk-crisis {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.3), rgba(244, 67, 54, 0.1));
        border-left: 4px solid #F44336;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .crisis-banner {
        background: linear-gradient(90deg, #F44336, #E91E63);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    .hotline-button {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem 2rem;
        border-radius: 30px;
        text-decoration: none;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .hotline-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4);
    }
    
    .coping-card {
        background: rgba(255,255,255,0.08);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.15);
        color: #ffffff !important;
    }
    
    .coping-card strong {
        color: #00d9ff !important;
    }
    
    .stMarkdown p, .stMarkdown span, .stMarkdown div {
        color: #e8e8e8 !important;
    }
    
    .stInfo {
        background-color: rgba(0, 217, 255, 0.15) !important;
        color: #ffffff !important;
    }
    
    .stInfo p {
        color: #ffffff !important;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.05);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        color: #a0a0a0;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .stTextArea textarea {
        background: #ffffff;
        border: 2px solid rgba(0,0,0,0.2);
        border-radius: 12px;
        color: #000000 !important;
        font-size: 1rem;
        padding: 1rem;
    }
    
    .stTextArea textarea::placeholder {
        color: #666666 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #00d9ff;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
        color: #000000 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00d9ff, #0099ff);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 30px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 217, 255, 0.4);
    }
    
    .sidebar .stMarkdown {
        color: #e8e8e8;
    }
    
    div[data-testid="stSidebar"] {
        background: rgba(22, 33, 62, 0.95);
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000", key="api_url")

# Crisis Resources
CRISIS_RESOURCES = {
    "hotlines": [
        {"name": "988 Suicide & Crisis Lifeline (24/7)", "number": "Call or Text: 988", "url": "tel:988"},
        {"name": "Crisis Text Line (24/7)", "number": "Text HOME to 741741", "url": "sms:741741?body=HOME"},
        {"name": "National Suicide Prevention Lifeline", "number": "1-800-273-TALK (8255)", "url": "tel:18002738255"},
        {"name": "SAMHSA National Helpline", "number": "1-800-662-4357", "url": "tel:18006624357"}
    ],
    "coping_techniques": [
        {"icon": "*", "title": "5-4-3-2-1 Grounding", "desc": "Name 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste"},
        {"icon": "*", "title": "Box Breathing", "desc": "Inhale 4 sec, hold 4 sec, exhale 4 sec, hold 4 sec"},
        {"icon": "*", "title": "Ice Cube Technique", "desc": "Hold ice in your hand to interrupt distressing thoughts"},
        {"icon": "*", "title": "Journaling", "desc": "Write down your thoughts without judgment"},
        {"icon": "*", "title": "Movement", "desc": "Take a short walk, even just around your room"},
        {"icon": "*", "title": "Hydrate", "desc": "Drink a glass of cold water slowly"}
    ],
    "grounding_exercises": [
        "Feel your feet on the ground. Notice the pressure, temperature, texture.",
        "Name 3 things you're grateful for right now.",
        "Describe your surroundings in detail as if explaining to someone.",
        "Count backwards from 100 by 7s.",
        "Think of a safe place and describe every detail of it."
    ]
}


def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_prediction(text: str) -> dict:
    """Get prediction from API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Please ensure the backend is running."}
    except Exception as e:
        return {"error": str(e)}


def create_risk_gauge(risk_score: float) -> go.Figure:
    """Create a risk gauge visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': '%', 'font': {'size': 40, 'color': '#e8e8e8'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#e8e8e8"},
            'bar': {'color': "rgba(255,255,255,0.8)"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': '#4CAF50'},
                {'range': [25, 50], 'color': '#FFC107'},
                {'range': [50, 75], 'color': '#FF9800'},
                {'range': [75, 100], 'color': '#F44336'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': risk_score * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e8e8e8'},
        height=250,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    return fig


def create_probability_chart(probabilities: dict) -> go.Figure:
    """Create probability distribution chart."""
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=[v * 100 for v in values],
            marker_color=colors,
            text=[f'{v*100:.1f}%' for v in values],
            textposition='auto',
            textfont={'color': '#e8e8e8', 'size': 14}
        )
    ])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e8e8e8'},
        height=300,
        xaxis={'title': '', 'tickfont': {'size': 12}},
        yaxis={'title': 'Probability (%)', 'range': [0, 100], 'tickfont': {'size': 12}},
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False
    )
    
    return fig


def display_crisis_resources():
    """Display crisis resources and coping techniques."""
    st.markdown("""
    <div class="crisis-banner">
        <h2 style="margin:0; color:white;">We're Here For You</h2>
        <p style="margin:0.5rem 0 0 0; font-size:1.1rem;">
            If you're in crisis or having thoughts of suicide, please reach out for help immediately.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hotlines
    st.markdown("### Crisis Hotlines")
    cols = st.columns(2)
    for i, hotline in enumerate(CRISIS_RESOURCES["hotlines"]):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="coping-card">
                <span style="color:#ffffff; font-weight:600; font-size:1rem;">{hotline['name']}</span><br>
                <span style="font-size:1.3rem; color:#00d9ff; font-weight:700;">{hotline['number']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Coping Techniques
    st.markdown("### Coping Techniques")
    for technique in CRISIS_RESOURCES["coping_techniques"]:
        st.markdown(f"""
        <div class="coping-card">
            <span style="font-size:1.2rem;">{technique['icon']}</span>
            <span style="color:#00d9ff; font-weight:600; font-size:1rem;"> {technique['title']}: </span>
            <span style="color:#ffffff; font-size:0.95rem;">{technique['desc']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Grounding Exercise
    st.markdown("### Grounding Exercise")
    exercise = st.selectbox(
        "Choose an exercise:",
        CRISIS_RESOURCES["grounding_exercises"],
        label_visibility="collapsed"
    )
    st.markdown(f"""
    <div style="background: rgba(0, 217, 255, 0.15); padding: 1rem; border-radius: 8px; border-left: 4px solid #00d9ff;">
        <span style="color:#ffffff; font-size:1rem;"><strong style="color:#00d9ff;">Try this now:</strong> {exercise}</span>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application."""
    # Header
    st.markdown('<h1 class="hero-title">MindGuard AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">AI-Powered Mental Health Text Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Settings")
        
        # API Status
        api_status = check_api_health()
        if api_status:
            st.success("API Connected")
        else:
            st.error("API Offline")
            st.info("Start the API with:\n```\ncd api && python main.py\n```")
        
        st.markdown("---")
        st.markdown("## About")
        st.markdown("""
        **MindGuard AI** uses transformer-based NLP 
        to analyze text for mental health indicators.
        
        **Risk Levels:**
        - Normal (Green)
        - Mild Negative (Yellow)
        - High Negative (Orange)
        - Crisis-Risk (Red)
        
        This is a support tool, not a replacement 
        for professional mental health care.
        """)
        
        st.markdown("---")
        st.markdown("## Resources")
        st.markdown("[SAMHSA](https://www.samhsa.gov/)")
        st.markdown("[NAMI](https://www.nami.org/)")
        st.markdown("[Mental Health America](https://mhanational.org/)")
    
    # Main content
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Share Your Thoughts")
        
        # Text input
        user_text = st.text_area(
            "How are you feeling today?",
            height=150,
            placeholder="Type here... Your privacy is important. This text is only processed locally.",
            label_visibility="collapsed"
        )
        
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            analyze_btn = st.button("Analyze", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("Clear", use_container_width=True)
        
        if clear_btn:
            st.rerun()
        
        # Store results in session state
        if 'result' not in st.session_state:
            st.session_state.result = None
        
        if analyze_btn and user_text.strip():
            with st.spinner("Analyzing..."):
                result = get_prediction(user_text)
                st.session_state.result = result
        
        # Display results
        if st.session_state.result:
            result = st.session_state.result
            
            if 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                # Risk level card
                risk_class = result['class_label'].lower().replace(' ', '-').replace('-risk', '')
                class_css = {
                    'normal': 'risk-normal',
                    'mild-negative': 'risk-mild',
                    'high-negative': 'risk-high',
                    'crisis': 'risk-crisis'
                }.get(risk_class, 'risk-normal')
                
                st.markdown(f"""
                <div class="risk-card {class_css}">
                    <h3 style="margin:0 0 0.5rem 0;">Classification: {result['class_label']}</h3>
                    <p style="margin:0; opacity:0.8;">
                        Confidence: {result['confidence']*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Crisis indicators
                if result.get('crisis_indicators'):
                    st.warning(f"⚠️ Detected indicators: {', '.join(result['crisis_indicators'])}")
    
    with col2:
        if st.session_state.result and 'error' not in st.session_state.result:
            result = st.session_state.result
            
            st.markdown("### Risk Assessment")
            
            # Risk gauge
            st.plotly_chart(
                create_risk_gauge(result['risk_score']),
                use_container_width=True,
                config={'displayModeBar': False}
            )
            
            # Probability chart
            st.markdown("### Class Probabilities")
            st.plotly_chart(
                create_probability_chart(result['all_probabilities']),
                use_container_width=True,
                config={'displayModeBar': False}
            )
        else:
            st.markdown("### Risk Assessment")
            st.info("Enter text and click 'Analyze' to see results")
    
    # Crisis resources (show when crisis detected)
    if st.session_state.result:
        result = st.session_state.result
        if 'error' not in result and (result['class_id'] == 3 or result['risk_score'] > 0.7):
            st.markdown("---")
            display_crisis_resources()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#999; padding:1rem;">
        <p style="font-size:1.1rem; margin-bottom:0.5rem;">MindGuard AI v1.0 | Built for mental health awareness</p>
        <p style="font-size:1rem; color:#00d9ff; font-weight:600; margin-bottom:0.5rem;">Made by Jalal Diab</p>
        <p style="font-size:0.85rem; color:#666;">
            <strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace 
            professional mental health diagnosis or treatment. If you're in crisis, please contact emergency services.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


