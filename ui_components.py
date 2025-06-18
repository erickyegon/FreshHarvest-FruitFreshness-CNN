"""
Professional UI Components for FreshHarvest Streamlit Applications
================================================================

This module provides professional UI components, styling, and layouts
for the FreshHarvest fruit freshness classification system.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

def load_professional_css():
    """Load professional CSS styling for the application."""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        font-size: 1.3rem;
        font-weight: 400;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    .performance-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        margin-top: 1rem;
    }
    
    /* Card Styles */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Status Indicators */
    .status-success {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    .status-info {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    /* Prediction Results */
    .prediction-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 2rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .prediction-header {
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f0f0;
    }
    
    .prediction-result {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background: #f8f9fa;
    }
    
    .result-fresh {
        border-left: 5px solid #28a745;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
    
    .result-spoiled {
        border-left: 5px solid #dc3545;
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    }
    
    /* Upload Area */
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #f0f4ff 0%, #e0ebff 100%);
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Footer */
    .footer {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 3rem;
        border-top: 3px solid #667eea;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .main-subtitle {
            font-size: 1rem;
        }
        .metric-card {
            margin: 0.25rem;
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def create_header(title, subtitle, performance_metric=None):
    """Create a professional header with branding."""
    performance_badge = ""
    if performance_metric:
        performance_badge = f'<div class="performance-badge">🏆 {performance_metric}</div>'
    
    st.markdown(f"""
    <div class="main-header">
        <div class="main-title">{title}</div>
        <div class="main-subtitle">{subtitle}</div>
        {performance_badge}
    </div>
    """, unsafe_allow_html=True)

def create_metric_cards(metrics):
    """Create professional metric cards."""
    cols = st.columns(len(metrics))
    
    for i, (label, value, icon) in enumerate(metrics):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{icon} {value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

def create_status_indicator(status_type, message):
    """Create a professional status indicator."""
    status_class = f"status-{status_type}"
    st.markdown(f"""
    <div class="{status_class}">
        {message}
    </div>
    """, unsafe_allow_html=True)

def create_info_card(title, content, icon="ℹ️"):
    """Create an information card."""
    st.markdown(f"""
    <div class="info-card">
        <h4>{icon} {title}</h4>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

def create_prediction_display(result, model_info=None):
    """Create a professional prediction results display."""
    if not result:
        return
    
    # Determine result styling
    result_class = "result-fresh" if result['condition'] == "Fresh" else "result-spoiled"
    condition_icon = "✅" if result['condition'] == "Fresh" else "⚠️"
    confidence_color = "#28a745" if result['confidence'] > 0.8 else "#ffc107" if result['confidence'] > 0.6 else "#dc3545"
    
    # Model info header
    model_status = ""
    if model_info:
        model_status = f"""
        <div style="text-align: center; margin-bottom: 1rem; padding: 0.5rem; background: #e8f4fd; border-radius: 8px;">
            <strong>🏆 {model_info}</strong>
        </div>
        """
    
    st.markdown(f"""
    <div class="prediction-container">
        <div class="prediction-header">
            <h3>🔍 AI Analysis Results</h3>
            {model_status}
        </div>
        
        <div class="prediction-result {result_class}">
            <div style="flex: 1;">
                <h4>🍎 Fruit Type</h4>
                <p style="font-size: 1.2rem; font-weight: 600; margin: 0;">{result['fruit_type']}</p>
            </div>
            <div style="flex: 1; text-align: center;">
                <h4>📊 Condition</h4>
                <p style="font-size: 1.2rem; font-weight: 600; margin: 0;">{condition_icon} {result['condition']}</p>
            </div>
            <div style="flex: 1; text-align: right;">
                <h4>🎯 Confidence</h4>
                <p style="font-size: 1.2rem; font-weight: 600; margin: 0; color: {confidence_color};">{result['confidence']:.1%}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_confidence_gauge(confidence):
    """Create a professional confidence gauge using Plotly."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Inter"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_prediction_chart(predictions, class_names):
    """Create a professional prediction probability chart."""
    if len(predictions) != len(class_names):
        min_length = min(len(predictions), len(class_names))
        predictions = predictions[:min_length]
        class_names = class_names[:min_length]
    
    # Sort by confidence
    sorted_data = sorted(zip(class_names, predictions), key=lambda x: x[1], reverse=True)
    top_classes = [x[0] for x in sorted_data[:8]]
    top_probs = [x[1] for x in sorted_data[:8]]
    
    # Create color scheme
    colors = ['#667eea' if i == 0 else '#a8a8a8' for i in range(len(top_classes))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_probs,
            y=top_classes,
            orientation='h',
            marker_color=colors,
            text=[f'{p:.1%}' for p in top_probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Top Prediction Probabilities",
        xaxis_title="Confidence",
        yaxis_title="Classes",
        height=400,
        font={'family': "Inter"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )
    
    return fig

def create_upload_area():
    """Create a professional file upload area."""
    st.markdown("""
    <div class="upload-area">
        <h3>📸 Upload Fruit Image</h3>
        <p>Drag and drop an image file or click to browse</p>
        <p style="font-size: 0.9rem; color: #666;">Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)

def create_footer():
    """Create a professional footer."""
    current_year = datetime.now().year
    st.markdown(f"""
    <div class="footer">
        <p><strong>🍎 FreshHarvest AI System</strong></p>
        <p>Powered by TensorFlow & Streamlit | © {current_year} FreshHarvest Logistics</p>
        <p style="font-size: 0.9rem; color: #666;">
            🏆 Production Model: 96.50% Accuracy | 🚀 Enterprise-Grade AI Solution
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar_info(model_available=False, model_accuracy=None):
    """Create professional sidebar information."""
    st.sidebar.markdown("## 📊 System Information")
    
    # Model Status
    if model_available and model_accuracy:
        st.sidebar.success(f"🏆 Production Model Active\n\n**Accuracy**: {model_accuracy}")
    else:
        st.sidebar.warning("🚧 Demo Mode Active")
    
    # System Stats
    st.sidebar.markdown("### 🎯 Model Performance")
    if model_available:
        st.sidebar.metric("Validation Accuracy", "96.50%", "6.5%")
        st.sidebar.metric("Precision", "96.85%", "8.85%")
        st.sidebar.metric("Recall", "96.19%", "8.19%")
    else:
        st.sidebar.info("Production metrics available when model is loaded")
    
    # Technical Info
    st.sidebar.markdown("### ⚙️ Technical Details")
    st.sidebar.markdown("""
    - **Architecture**: Lightweight CNN
    - **Classes**: 16 (8 fruits × 2 conditions)
    - **Input Size**: 224×224×3
    - **Framework**: TensorFlow/Keras
    - **Deployment**: Streamlit
    """)
    
    # Quick Actions
    st.sidebar.markdown("### 🚀 Quick Actions")
    if st.sidebar.button("🔄 Refresh Model"):
        st.experimental_rerun()
    
    if st.sidebar.button("📊 View Metrics"):
        st.sidebar.info("Detailed metrics displayed in main area")

def create_feature_showcase():
    """Create a feature showcase section."""
    st.markdown("### 🌟 Key Features")
    
    features = [
        ("🎯", "96.50% Accuracy", "Industry-leading precision for fruit classification"),
        ("⚡", "Real-time Analysis", "Instant results with optimized inference pipeline"),
        ("🔬", "Advanced AI", "Lightweight CNN with production-grade performance"),
        ("📱", "User-friendly", "Professional interface with intuitive design"),
        ("🛡️", "Robust & Reliable", "Comprehensive error handling and validation"),
        ("🚀", "Production Ready", "Enterprise-grade deployment and monitoring")
    ]
    
    cols = st.columns(3)
    for i, (icon, title, description) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="info-card" style="text-align: center; min-height: 120px;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                <h5>{title}</h5>
                <p style="font-size: 0.9rem; color: #666;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
