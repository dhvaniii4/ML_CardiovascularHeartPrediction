import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Page Config ---
st.set_page_config(
    page_title="CardioCare AI Dashboard",
    page_icon="🏥",
    layout="wide"
)

# --- THEME & VISIBILITY FIX (Force Dark Text) ---
# st.markdown("""
#     <style>
#     /* Force background color */
#     .stApp { background-color: #f8f9fa; }
    
#     /* Global Text Visibility */
#     label, p, span, div, .stMetric, .stMarkdown {
#         color: #1a1a1a !important;
#     }
    
#     /* Header Styling */
#     h1, h2, h3 {
#         color: #004b95 !important;
#         font-weight: bold;
#     }

#     /* 1. BLUE GENERATE BUTTON */
#     div[data-testid="stFormSubmitButton"] button {
#         background-color: #004b95 !important;
#         color: white !important;
#         border-radius: 8px !important;
#         width: 100% !important;
#         border: none !important;
#         padding: 0.5rem 0 !important;
#         font-weight: bold !important;
#     }
    
#     div[data-testid="stFormSubmitButton"] button:hover {
#         background-color: #003366 !important;
#         color: white !important;
#     }

#     /* 2. WHITE TEXT FOR DROPDOWNS (Gender, Chol, Gluc) */
#     /* Target the selected value text */
#     div[data-baseweb="select"] div {
#         color: white !important;
#     }
    
#     /* Set dropdown background to blue so white text is readable */
#     div[data-baseweb="select"] {
#         background-color: #004b95 !important;
#         border-radius: 4px;
#     }

#     /* Style the metrics boxes */
#     [data-testid="stMetricValue"] {
#         color: #004b95 !important;
#         font-size: 1.8rem !important;
#     }

#     /* Style the input form container */
#     div[data-testid="stForm"] {
#         background-color: #ffffff;
#         border-radius: 12px;
#         border: 1px solid #dee2e6;
#         padding: 2rem;
#     }
#     </style>
#     """, unsafe_allow_html=True)
st.markdown("""
    <style>
    /* Force background color */
    .stApp { background-color: #f8f9fa; }
    
    /* Global Text Visibility */
    label, p, span, div, .stMetric, .stMarkdown {
        color: #1a1a1a !important;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        color: #004b95 !important;
        font-weight: bold;
    }

    /* 1. BLACK GENERATE BUTTON WITH WHITE TEXT */
    div[data-testid="stFormSubmitButton"] button {
        background-color: #000000 !important; /* Changed to Black */
        color: #ffffff !important;           /* Changed to White */
        border-radius: 8px !important;
        width: 100% !important;
        border: none !important;
        padding: 0.5rem 0 !important;
        font-weight: bold !important;
    }
    
    /* Hover effect: slightly lighter gray so user knows it's clickable */
    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: white !important;
        color: white !important;
    }

    /* 2. WHITE TEXT FOR DROPDOWNS (Gender, Chol, Gluc) */
    div[data-baseweb="select"] div {
        color: white !important;
    }
    
    div[data-baseweb="select"] {
        background-color: #004b95 !important;
        border-radius: 4px;
    }

    /* Target the dropdown menu items when list is open */
    ul[role="listbox"] li {
        background-color: #004b95 !important;
        color: white !important;
    }

    /* Style the metrics boxes */
    [data-testid="stMetricValue"] {
        color: #004b95 !important;
        font-size: 1.8rem !important;
    }

    /* Style the input form container */
    div[data-testid="stForm"] {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #dee2e6;
        padding: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
# --- Logic ---
def get_bmi_category(bmi):
    if bmi <= 18.5: return 'Underweight', 'orange'
    elif bmi <= 25: return 'Normal', 'green'
    elif bmi <= 30: return 'Overweight', 'orange'
    else: return 'Obese', 'red'

st.title("🏥 CardioCare")

# --- Input Form ---
with st.form("medical_form"):
    st.subheader("📋 Physiological & Clinical Data")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age (Years)", 1, 120, 45)
        gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x==1 else "Male")
    with c2:
        height = st.number_input("Height (cm)", 100, 250, 170)
        weight = st.number_input("Weight (kg)", 30.0, 250.0, 75.0)
    with c3:
        ap_hi = st.number_input("Systolic BP", 80, 240, 120)
        ap_lo = st.number_input("Diastolic BP", 40, 180, 80)
    with c4:
        # GLUCOSE FIELD (Now explicitly visible)
        chol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])
        gluc = st.selectbox("Glucose Level", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])

    st.write("---")
    st.subheader("🥗 Lifestyle & Habits")
    l1, l2, l3 = st.columns(3)
    with l1: smoke = st.toggle("Active Smoker")
    with l2: alco = st.toggle("Alcohol User")
    with l3: active = st.toggle("Regularly Active", value=True)

    submitted = st.form_submit_button("GENERATE REPORT")

# --- Result Output ---
if submitted:
    st.header("🔬 Analysis Report")
    
    bmi = weight / ((height/100)**2)
    bmi_cat, bmi_color = get_bmi_category(bmi)
    
    # Simple risk logic for demonstration
    risk_score = 0
    if ap_hi > 135: risk_score += 1
    if chol > 1 or gluc > 1: risk_score += 1
    if smoke: risk_score += 1
    prediction = 1 if risk_score >= 2 else 0

    # DISPLAY METRICS (Visibility Fixed)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Calculated BMI", f"{bmi:.1f}", bmi_cat)
    m2.metric("BP Status", f"{ap_hi}/{ap_lo}", "High" if ap_hi > 130 else "Normal")
    m3.metric("Glucose Result", "Elevated" if gluc > 1 else "Normal")
    m4.metric("Lifestyle Risk", "High" if smoke else "Low")

    st.divider()

    if prediction == 1:
        st.error("### 🚩 FINAL VERDICT: HIGH RISK DETECTED")
        st.write("Patient shows multiple cardiovascular markers. Clinical follow-up required.")
    else:
        st.success("### ✅ FINAL VERDICT: LOW RISK")
        st.write("Patient is within healthy parameters. Advise continued physical activity.")

    # Graphics Section (Performance Requirement)
    st.subheader("📊 Analytical Data Context")
    
    
    
    # col_g1, col_g2 = st.columns(2)
    # with col_g1:
    #     fig, ax = plt.subplots(figsize=(5, 4))
    #     cm_data = [[4510, 1490], [1210, 4790]] 
    #     disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm_data), display_labels=["Healthy", "Cardio"])
    #     disp.plot(ax=ax, cmap='Blues', colorbar=False)
    #     st.pyplot(fig)
    # with col_g2:
    #     st.write("**Feature Importance**")
    #     importance = pd.DataFrame({
    #         'Feature': ['BP', 'Cholesterol', 'Glucose', 'BMI', 'Smoking'],
    #         'Weight': [0.4, 0.25, 0.15, 0.1, 0.1]
    #     })
    #     st.bar_chart(importance, x='Feature', y='Weight', color="#004b95")


    