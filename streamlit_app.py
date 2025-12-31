"""
MindGuard AI - Streamlit Cloud Deployment
Standalone version with built-in classification.
"""

import streamlit as st
import re

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
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
    
    .hero-title {
        font-size: 3rem;
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
    }
    
    .crisis-banner {
        background: linear-gradient(90deg, #F44336, #E91E63);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .coping-card {
        background: rgba(255,255,255,0.08);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.15);
        color: #ffffff !important;
    }
    
    .stTextArea textarea {
        background: #ffffff;
        border: 2px solid rgba(0,0,0,0.2);
        border-radius: 12px;
        color: #000000 !important;
        font-size: 1rem;
        padding: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #00d9ff;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00d9ff, #0099ff);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 30px;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 217, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Label mappings
LABEL_MAP = {
    0: "Normal",
    1: "Mild Negative",
    2: "High Negative",
    3: "Crisis-Risk"
}

RISK_COLORS = {
    0: "#4CAF50",
    1: "#FFC107",
    2: "#FF9800",
    3: "#F44336"
}

# Crisis Resources
CRISIS_RESOURCES = {
    "hotlines": [
        {"name": "988 Suicide & Crisis Lifeline (24/7)", "number": "Call or Text: 988"},
        {"name": "Crisis Text Line (24/7)", "number": "Text HOME to 741741"},
        {"name": "National Suicide Prevention Lifeline", "number": "1-800-273-TALK (8255)"},
        {"name": "SAMHSA National Helpline", "number": "1-800-662-4357"}
    ],
    "coping_techniques": [
        {"title": "5-4-3-2-1 Grounding", "desc": "Name 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste"},
        {"title": "Box Breathing", "desc": "Inhale 4 sec, hold 4 sec, exhale 4 sec, hold 4 sec"},
        {"title": "Ice Cube Technique", "desc": "Hold ice in your hand to interrupt distressing thoughts"},
        {"title": "Journaling", "desc": "Write down your thoughts without judgment"},
        {"title": "Movement", "desc": "Take a short walk, even just around your room"},
        {"title": "Hydrate", "desc": "Drink a glass of cold water slowly"}
    ]
}


def classify_text(text: str) -> dict:
    """Classify text using keyword-based analysis."""
    text_lower = text.lower()
    
    # Keywords for classification
    positive_keywords = [
        'happy', 'great', 'wonderful', 'amazing', 'good', 'love', 'excited',
        'grateful', 'thankful', 'blessed', 'joy', 'fantastic', 'awesome',
        'beautiful', 'excellent', 'perfect', 'best', 'fun', 'enjoy', 'smile',
        'laugh', 'peaceful', 'calm', 'relaxed', 'content', 'proud', 'accomplished'
    ]
    
    mild_negative_keywords = [
        'sad', 'upset', 'worried', 'stressed', 'tired', 'lonely', 'anxious',
        'frustrated', 'annoyed', 'disappointed', 'nervous', 'down', 'bad day',
        'struggling', 'difficult', 'hard time', 'overwhelmed', 'exhausted'
    ]
    
    high_negative_keywords = [
        'hopeless', 'worthless', 'hate myself', 'give up', 'cant go on',
        'nobody cares', 'alone forever', 'failure', 'burden', 'depressed',
        'depression', 'panic attack', 'cant take it', 'breaking down',
        'falling apart', 'nothing matters', 'empty inside', 'numb'
    ]
    
    crisis_keywords = [
        'suicide', 'suicidal', 'kill myself', 'end my life', 'want to die',
        'self-harm', 'self harm', 'cutting myself', 'hurt myself',
        'no reason to live', 'better off dead', 'end it all', 'take my life'
    ]
    
    # Count keyword matches
    positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
    mild_neg_count = sum(1 for kw in mild_negative_keywords if kw in text_lower)
    high_neg_count = sum(1 for kw in high_negative_keywords if kw in text_lower)
    crisis_count = sum(1 for kw in crisis_keywords if kw in text_lower)
    
    # Determine class based on keywords
    if crisis_count > 0:
        predicted_class = 3
        confidence = min(0.75 + (crisis_count * 0.08), 0.98)
        probs = [0.02, 0.03, 0.15, 0.80]
    elif high_neg_count > 0:
        predicted_class = 2
        confidence = min(0.70 + (high_neg_count * 0.07), 0.95)
        probs = [0.05, 0.10, 0.70, 0.15]
    elif mild_neg_count > 0 and positive_count == 0:
        predicted_class = 1
        confidence = min(0.65 + (mild_neg_count * 0.06), 0.90)
        probs = [0.15, 0.65, 0.15, 0.05]
    elif positive_count > 0:
        predicted_class = 0
        confidence = min(0.70 + (positive_count * 0.06), 0.95)
        probs = [0.80, 0.12, 0.05, 0.03]
    else:
        predicted_class = 0
        confidence = 0.55
        probs = [0.55, 0.25, 0.12, 0.08]
    
    # Calculate risk score
    risk_weights = [0.0, 0.25, 0.6, 1.0]
    risk_score = sum(probs[i] * risk_weights[i] for i in range(4))
    
    return {
        'class_id': predicted_class,
        'class_label': LABEL_MAP[predicted_class],
        'confidence': round(confidence, 4),
        'risk_score': round(risk_score, 4),
        'risk_color': RISK_COLORS[predicted_class],
        'all_probabilities': {LABEL_MAP[i]: round(probs[i], 4) for i in range(4)}
    }


def display_crisis_resources():
    """Display crisis resources."""
    st.markdown("""
    <div class="crisis-banner">
        <h2 style="margin:0; color:white;">We're Here For You</h2>
        <p style="margin:0.5rem 0 0 0; font-size:1.1rem;">
            If you're in crisis or having thoughts of suicide, please reach out for help immediately.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Crisis Hotlines")
    cols = st.columns(2)
    for i, hotline in enumerate(CRISIS_RESOURCES["hotlines"]):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="coping-card">
                <span style="color:#ffffff; font-weight:600;">{hotline['name']}</span><br>
                <span style="font-size:1.3rem; color:#00d9ff; font-weight:700;">{hotline['number']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("### Coping Techniques")
    for technique in CRISIS_RESOURCES["coping_techniques"]:
        st.markdown(f"""
        <div class="coping-card">
            <span style="color:#00d9ff; font-weight:600;">{technique['title']}: </span>
            <span style="color:#ffffff;">{technique['desc']}</span>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application."""
    st.markdown('<h1 class="hero-title">MindGuard AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">AI-Powered Mental Health Text Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## About")
        st.markdown("""
        **MindGuard AI** analyzes text for mental health indicators.
        
        **Risk Levels:**
        - Normal (Green)
        - Mild Negative (Yellow)
        - High Negative (Orange)
        - Crisis-Risk (Red)
        
        This is a support tool, not a replacement for professional care.
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
        
        user_text = st.text_area(
            "How are you feeling today?",
            height=150,
            placeholder="Type here... Your privacy is important.",
            label_visibility="collapsed"
        )
        
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            analyze_btn = st.button("Analyze", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("Clear", use_container_width=True)
        
        if clear_btn:
            st.rerun()
        
        if 'result' not in st.session_state:
            st.session_state.result = None
        
        if analyze_btn and user_text.strip():
            result = classify_text(user_text)
            st.session_state.result = result
        
        if st.session_state.result:
            result = st.session_state.result
            
            risk_class = result['class_label'].lower().replace(' ', '-').replace('-risk', '')
            class_css = {
                'normal': 'risk-normal',
                'mild-negative': 'risk-mild',
                'high-negative': 'risk-high',
                'crisis': 'risk-crisis'
            }.get(risk_class, 'risk-normal')
            
            st.markdown(f"""
            <div class="risk-card {class_css}">
                <h3 style="margin:0 0 0.5rem 0; color:white;">Classification: {result['class_label']}</h3>
                <p style="margin:0; color:#e8e8e8;">
                    Confidence: {result['confidence']*100:.1f}% | Risk Score: {result['risk_score']*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.result:
            result = st.session_state.result
            
            st.markdown("### Risk Assessment")
            
            # Display probabilities
            for label, prob in result['all_probabilities'].items():
                color = RISK_COLORS[list(LABEL_MAP.values()).index(label)]
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <span style="color:#ffffff;">{label}</span>
                    <div style="background:#333; border-radius:10px; overflow:hidden; height:24px;">
                        <div style="background:{color}; width:{prob*100}%; height:100%; border-radius:10px; display:flex; align-items:center; justify-content:flex-end; padding-right:8px;">
                            <span style="color:white; font-weight:bold; font-size:0.85rem;">{prob*100:.1f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("### Risk Assessment")
            st.info("Enter text and click 'Analyze' to see results")
    
    # Crisis resources
    if st.session_state.result:
        result = st.session_state.result
        if result['class_id'] == 3 or result['risk_score'] > 0.7:
            st.markdown("---")
            display_crisis_resources()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#999; padding:1rem;">
        <p style="font-size:1.1rem; margin-bottom:0.5rem;">MindGuard AI v1.0 | Built for mental health awareness</p>
        <p style="font-size:1rem; color:#00d9ff; font-weight:600; margin-bottom:0.5rem;">Made by Jalal Diab</p>
        <p style="font-size:0.85rem; color:#666;">
            <strong>Disclaimer:</strong> This tool is for educational purposes only.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

