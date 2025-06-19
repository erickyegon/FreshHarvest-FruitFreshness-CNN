#!/usr/bin/env python3
"""
Test HTML rendering in Streamlit to debug the raw HTML display issue.
"""

import streamlit as st

# Configure page
st.set_page_config(
    page_title="HTML Rendering Test",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 HTML Rendering Test")

# Test 1: Basic HTML with unsafe_allow_html=True
st.header("Test 1: Basic HTML with unsafe_allow_html=True")
st.markdown("""
<div style="background: #e8f4fd; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
    <strong>🏆 Production Model (96.50% Accuracy)</strong>
</div>
""", unsafe_allow_html=True)

# Test 2: Complex HTML structure
st.header("Test 2: Complex HTML Structure")
st.markdown("""
<div class="prediction-result result-spoiled">
    <div style="flex: 1;">
        <h4>🍎 Fruit Type</h4>
        <p style="font-size: 1.2rem; font-weight: 600; margin: 0;">Orange</p>
    </div>
    <div style="flex: 1; text-align: center;">
        <h4>📊 Condition</h4>
        <p style="font-size: 1.2rem; font-weight: 600; margin: 0;">⚠️ Spoiled</p>
    </div>
    <div style="flex: 1; text-align: right;">
        <h4>🎯 Confidence</h4>
        <p style="font-size: 1.2rem; font-weight: 600; margin: 0; color: #28a745;">99.8%</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Test 3: CSS Styling
st.header("Test 3: CSS Styling")
st.markdown("""
<style>
.test-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    text-align: center;
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
.result-spoiled {
    border-left: 5px solid #dc3545;
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
}
</style>

<div class="test-box">
    <h3>This should be styled with gradient background</h3>
    <p>If you see this with a blue-purple gradient, HTML rendering is working correctly!</p>
</div>
""", unsafe_allow_html=True)

# Test 4: Without unsafe_allow_html (should show raw HTML)
st.header("Test 4: Without unsafe_allow_html (should show raw HTML)")
st.markdown("""
<div style="background: red; color: white; padding: 1rem;">
    This should show as raw HTML text, not rendered
</div>
""")

# Test 5: Using st.write with HTML
st.header("Test 5: Using st.write with HTML")
st.write("""
<div style="background: green; color: white; padding: 1rem;">
    This should also show as raw HTML text
</div>
""")

# Test 6: Function-based HTML rendering
def create_test_display():
    """Test function that creates HTML display."""
    st.markdown("""
    <div style="background: #f0f8ff; padding: 1rem; border-radius: 10px; border-left: 5px solid #2E8B57; margin: 1rem 0;">
        <h3>🍎 Fruit Type: <span style="color: #2E8B57;">Test Fruit</span></h3>
        <h3>📊 Condition: <span style="color: #DC143C; font-weight: bold;">Test Condition</span></h3>
        <h3>🎯 Confidence: <span style="color: #4682B4;">95.5%</span></h3>
    </div>
    """, unsafe_allow_html=True)

st.header("Test 6: Function-based HTML rendering")
create_test_display()

# Summary
st.header("Summary")
st.success("✅ If all tests above show properly styled content (except tests 4 & 5), HTML rendering is working correctly!")
st.info("ℹ️ Tests 4 & 5 should show raw HTML text - this is expected behavior without unsafe_allow_html=True")
st.warning("⚠️ If any of tests 1, 2, 3, or 6 show raw HTML instead of styled content, there's an issue with HTML rendering")
