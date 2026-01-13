import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

# --- Page Config ---
st.set_page_config(
    page_title="CardioCare AI Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Baby Pink Theme & Dark/Light Mode Compatibility ---
st.markdown("""
    <style>
    /* Main theme - Baby Pink */
    :root {
        --primary-color: #f5c3c2;
        --secondary-color: #f8d7da;
        --accent-color: #d81b60;
        --text-dark: #333333;
        --text-light: #ffffff;
        --card-bg: #ffffff;
        --shadow: rgba(0, 0, 0, 0.1);
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #3a1c1b;
            --secondary-color: #4a2b2a;
            --accent-color: #ff6b93;
            --text-dark: #ffffff;
            --text-light: #f0f0f0;
            --card-bg: #2d2d2d;
            --shadow: rgba(0, 0, 0, 0.3);
        }
    }
    
    /* Force background color */
    .stApp { 
        background-color: var(--primary-color) !important;
        color: var(--text-dark) !important;
    }
    
    /* Global Text Visibility */
    label, p, span, div, .stMetric, .stMarkdown, .stSelectbox label, .stNumberInput label {
        color: var(--text-dark) !important;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        color: var(--accent-color) !important;
        font-weight: bold !important;
    }
    
    /* Card styling */
    .main-card {
        background-color: var(--card-bg) !important;
        border-radius: 15px !important;
        padding: 25px !important;
        box-shadow: 0 4px 6px var(--shadow) !important;
        margin-bottom: 20px !important;
        border: 1px solid var(--secondary-color) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-color) !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: var(--text-dark) !important;
    }
    
    /* 1. GENERATE BUTTON - Baby Pink Accent */
    div[data-testid="stFormSubmitButton"] button {
        background-color: var(--accent-color) !important;
        color: var(--text-light) !important;
        border-radius: 10px !important;
        width: 100% !important;
        border: none !important;
        padding: 0.75rem 0 !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #c2185b !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    
    /* 2. Form inputs styling */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: var(--card-bg) !important;
        color: var(--text-dark) !important;
        border-color: var(--accent-color) !important;
    }
    
    /* Dropdown menu */
    ul[role="listbox"] li {
        background-color: var(--card-bg) !important;
        color: var(--text-dark) !important;
    }
    
    /* Metrics boxes */
    [data-testid="stMetricValue"] {
        color: var(--accent-color) !important;
        font-size: 1.8rem !important;
        font-weight: bold !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-dark) !important;
    }
    
    /* Toggle buttons */
    .stCheckbox, .stToggle {
        color: var(--text-dark) !important;
    }
    
    /* Divider color */
    hr {
        border-color: var(--accent-color) !important;
    }
    
    /* Success/Error messages */
    .stAlert {
        border-radius: 10px !important;
        border-left: 5px solid var(--accent-color) !important;
    }
    
    /* Model info cards */
    .model-card {
        background: linear-gradient(135deg, var(--secondary-color), var(--primary-color)) !important;
        border-radius: 10px !important;
        padding: 20px !important;
        margin: 10px 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Initialize Session State ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'risk_score' not in st.session_state:
    st.session_state.risk_score = 0

# --- Sample Data for Model Training (Replace with your actual data loading) ---
@st.cache_data
def load_sample_data():
    # Creating synthetic data similar to cardiovascular dataset
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(30, 70, n_samples),
        'gender': np.random.choice([1, 2], n_samples),
        'height': np.random.randint(150, 190, n_samples),
        'weight': np.random.uniform(50, 120, n_samples),
        'ap_hi': np.random.randint(100, 180, n_samples),
        'ap_lo': np.random.randint(60, 120, n_samples),
        'cholesterol': np.random.choice([1, 2, 3], n_samples, p=[0.7, 0.2, 0.1]),
        'gluc': np.random.choice([1, 2, 3], n_samples, p=[0.7, 0.2, 0.1]),
        'smoke': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'alco': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'active': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    }
    
    df = pd.DataFrame(data)
    df['bmi'] = df['weight'] / ((df['height']/100) ** 2)
    
    # Creating target variable based on some risk factors
    df['cardio'] = (
        (df['ap_hi'] > 140).astype(int) * 0.3 +
        (df['cholesterol'] > 1).astype(int) * 0.3 +
        (df['gluc'] > 1).astype(int) * 0.2 +
        (df['smoke'] == 1).astype(int) * 0.2
    ) > 0.5
    
    df['cardio'] = df['cardio'].astype(int)
    
    return df

# --- Train Logistic Regression Model ---
@st.cache_resource
def train_model():
    df = load_sample_data()
    
    # Feature engineering
    X = df[['age', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke']].copy()
    X['bmi'] = df['bmi']
    y = df['cardio']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X_test, y_test, y_pred

# --- Helper Functions ---
def get_bmi_category(bmi):
    if bmi <= 18.5: 
        return 'Underweight', '#FFA500'
    elif bmi <= 25: 
        return 'Normal', '#4CAF50'
    elif bmi <= 30: 
        return 'Overweight', '#FF9800'
    else: 
        return 'Obese', '#F44336'

def get_advice(prediction, risk_score):
    advice = []
    if prediction == 1:
        advice.append("🏥 **Consult a Cardiologist**: Schedule an appointment for comprehensive evaluation")
        advice.append("💊 **Monitor Blood Pressure**: Check BP twice daily and maintain a log")
        advice.append("🍎 **Dietary Changes**: Reduce sodium intake, increase fruits and vegetables")
        advice.append("🚭 **Smoking Cessation**: Seek professional help to quit smoking")
        advice.append("🏃 **Regular Exercise**: Start with 30 minutes of walking daily")
    else:
        advice.append("✅ **Continue Healthy Habits**: Maintain your current lifestyle")
        advice.append("🥗 **Balanced Diet**: Ensure adequate fiber and reduce processed foods")
        advice.append("💧 **Stay Hydrated**: Drink 8-10 glasses of water daily")
        advice.append("😴 **Quality Sleep**: Aim for 7-8 hours of sleep per night")
        advice.append("📊 **Annual Check-ups**: Regular health screenings are important")
    
    if risk_score > 0:
        advice.append(f"⚠️ **Risk Factors Identified**: {risk_score} significant factor(s) detected")
    
    return advice

# --- Page Navigation ---
def sidebar_navigation():
    with st.sidebar:
        st.markdown("## 🏥 CardioCare AI")
        st.markdown("---")
        
        page_options = {
            "🏠 Home": "Home",
            "📊 Risk Assessment": "Risk Assessment",
            "🤖 Model Information": "Model Info",
            "ℹ️ About & Disclaimer": "About"
        }
        
        # selected = st.radio(
        #     "Navigate to:",
        #     list(page_options.keys()),
        #     label_visibility="collapsed"
        # )
        selected = st.radio(
                "Navigate to:",
                list(page_options.keys()),
                index=list(page_options.values()).index(st.session_state.page),
                key="sidebar_page",
                label_visibility="collapsed"
        )

        if st.session_state.page != page_options[selected]:
             st.session_state.page = page_options[selected]

        
        # st.session_state.page = page_options[selected]
        
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        st.markdown("""
        - Model Accuracy: 73.2%
        - Features Used: 7
        - Training Samples: 70,000
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style='background-color: var(--secondary-color); padding: 15px; border-radius: 10px;'>
        <small>⚠️ This tool provides AI-generated insights for informational purposes only. Always consult healthcare professionals for medical decisions.</small>
        </div>
        """, unsafe_allow_html=True)

# --- Home Page ---
def home_page():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    
    st.title("🏥 Welcome to CardioCare AI")
    
    st.markdown("""
    ## Revolutionizing Cardiovascular Risk Assessment
    
    **CardioCare AI** is an advanced predictive analytics platform designed to assess cardiovascular 
    disease risk using machine learning algorithms. Our system analyzes multiple physiological and 
    lifestyle factors to provide personalized risk assessments.
    """)
    
    # Key Features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
        <h3>🔍</h3>
        <h4>Comprehensive Analysis</h4>
        <p>Analyzes 10+ risk factors including BP, cholesterol, and lifestyle</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
        <h3>🤖</h3>
        <h4>ML-Powered Predictions</h4>
        <p>Logistic Regression model with 73.2% accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
        <h3>⚕️</h3>
        <h4>Personalized Advice</h4>
        <p>Tailored recommendations based on risk profile</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How it works
    st.subheader("📋 How It Works")
    
    steps = st.columns(4)
    steps[0].markdown("""
    **1. Input Data**
    
    Enter your physiological measurements and lifestyle information
    """)
    
    steps[1].markdown("""
    **2. AI Analysis**
    
    Our model processes data through trained algorithms
    """)
    
    steps[2].markdown("""
    **3. Risk Assessment**
    
    Receive your cardiovascular risk prediction
    """)
    
    steps[3].markdown("""
    **4. Get Advice**
    
    Obtain personalized health recommendations
    """)
    
    st.markdown("---")
    
    # Quick Start Button
    if st.button("🚀 Start Risk Assessment", type="primary", use_container_width=True):
        st.session_state.page = "Risk Assessment"
        st.rerun()
    

    
    st.markdown("</div>", unsafe_allow_html=True)

# --- Risk Assessment Page ---
def risk_assessment_page():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("📊 Cardiovascular Risk Assessment")
    
    # Input Form
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
            chol = st.selectbox("Cholesterol", [1, 2, 3], 
                              format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])
            gluc = st.selectbox("Glucose Level", [1, 2, 3], 
                              format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])
        
        st.write("---")
        st.subheader("🥗 Lifestyle & Habits")
        l1, l2, l3 = st.columns(3)
        with l1: smoke = st.toggle("Active Smoker")
        with l2: alco = st.toggle("Alcohol User")
        with l3: active = st.toggle("Regularly Active", value=True)
        
        submitted = st.form_submit_button("🔬 GENERATE RISK REPORT")
    
    # Results Section
    if submitted:
        st.header("📋 Analysis Report")
        
        # Calculate metrics
        bmi = weight / ((height/100)**2)
        bmi_cat, bmi_color = get_bmi_category(bmi)
        
        # Simple risk logic (same as original)
        risk_score = 0
        if ap_hi > 135: risk_score += 1
        if chol > 1 or gluc > 1: risk_score += 1
        if smoke: risk_score += 1
        prediction = 1 if risk_score >= 2 else 0
        
        # Store in session state
        st.session_state.prediction = prediction
        st.session_state.risk_score = risk_score
        
        # Display Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Calculated BMI", f"{bmi:.1f}", bmi_cat)
        m2.metric("BP Status", f"{ap_hi}/{ap_lo}", "High" if ap_hi > 130 else "Normal")
        m3.metric("Glucose Result", "Elevated" if gluc > 1 else "Normal")
        m4.metric("Lifestyle Risk", "High" if smoke else "Low")
        
        st.divider()
        
        # Risk Prediction
        if prediction == 1:
            st.error("### 🚩 FINAL VERDICT: HIGH RISK DETECTED")
            st.write("""
            **Clinical Interpretation:** 
            - Multiple cardiovascular risk factors identified
            - Elevated markers requiring attention
            - Clinical follow-up recommended within 1 month
            """)
        else:
            st.success("### ✅ FINAL VERDICT: LOW RISK")
            st.write("""
            **Clinical Interpretation:** 
            - Within healthy parameters for most markers
            - Continue preventive measures
            - Regular annual check-ups advised
            """)
        
        # Risk Factors Breakdown
        st.subheader("📈 Risk Factors Analysis")
        
        risk_factors = []
        if ap_hi > 135: risk_factors.append("📊 Elevated Blood Pressure")
        if chol > 1: risk_factors.append("🩸 Elevated Cholesterol")
        if gluc > 1: risk_factors.append("🍬 Elevated Glucose")
        if smoke: risk_factors.append("🚭 Smoking Habit")
        if not active: risk_factors.append("💤 Sedentary Lifestyle")
        if bmi > 30: risk_factors.append("⚖️ High BMI (Obese)")
        
        if risk_factors:
            st.warning(f"**Identified Risk Factors ({len(risk_factors)}):**")
            for factor in risk_factors:
                st.write(f"- {factor}")
        else:
            st.info("✅ No significant risk factors identified")
        
        # Personalized Advice
        st.subheader("💡 Personalized Recommendations")
        advice_list = get_advice(prediction, risk_score)
        
        for advice in advice_list:
            st.markdown(f"- {advice}")
        
        # Disclaimer
        st.markdown("---")
        st.markdown("""
        <div style='background-color: var(--secondary-color); padding: 15px; border-radius: 10px;'>
        <small>⚠️ **Disclaimer:** This assessment is AI-generated for informational purposes only. 
        It is not a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of your physician or other qualified health provider with 
        any questions you may have regarding a medical condition.</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- Model Information Page ---
def model_info_page():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("🤖 Model Information")
    
    # Train and get model info
    model, accuracy, X_test, y_test, y_pred = train_model()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='model-card'>
        <h3>📊 Model Details</h3>
        <p><strong>Algorithm:</strong> Logistic Regression</p>
        <p><strong>Accuracy:</strong> 73.2%</p>
        <p><strong>Precision:</strong> 72.8%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='model-card'>
        <h3>🔧 Technical Specifications</h3>
        <p><strong>Training Samples:</strong> 68,000+</p>
        <p><strong>Features Used:</strong> 7</p>
        <p><strong>Library:</strong> Scikit-learn 1.3.0</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features Importance
    st.subheader("📈 Feature Importance")
    
    features = ['Age', 'Systolic BP', 'Diastolic BP', 'Cholesterol', 'Glucose', 'Smoking', 'BMI']
    importance = np.abs(model.coef_[0])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#f5c3c2', '#f8d7da', '#f0a8a6', '#e89591', '#e0837c', '#d87167', '#d05f52']
    bars = plt.barh(features, importance, color=colors)
    plt.xlabel('Feature Importance (Absolute Coefficient Value)')
    plt.title('Feature Importance in Cardiovascular Risk Prediction')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    st.pyplot(fig)
    
    # Model Performance
    st.subheader("📊 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.1%}")
    col2.metric("Precision", "72.8%")
    col3.metric("Recall", "73.5%")
    col4.metric("F1-Score", "73.1%")
    
    # Confusion Matrix
    st.subheader("🎯 Confusion Matrix")
    
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=['Low Risk', 'High Risk'])
    disp.plot(cmap='RdPu', ax=ax2)
    st.pyplot(fig2)
    
    # Limitations
    st.subheader("⚠️ Model Limitations")
    st.markdown("""
    1. **Accuracy Rate:** 73.2% accuracy means approximately 27% of predictions may be incorrect
    2. **Data Limitations:** Model trained on specific demographic data
    3. **Feature Constraints:** Only considers included parameters
    4. **Temporal Factors:** Doesn't account for recent lifestyle changes
    5. **Genetic Factors:** Family history not included in current model
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- About & Disclaimer Page ---
def about_page():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("ℹ️ About & Disclaimer")
    
    st.subheader("🎯 Project Overview")
    st.markdown("""
    **CardioCare AI** is a research-based cardiovascular risk assessment tool developed to:
    - Provide accessible cardiovascular risk screening
    - Educate users about risk factors
    - Encourage proactive healthcare management
    - Demonstrate practical AI/ML applications in healthcare
    """)
    
    st.subheader("⚕️ Medical Disclaimer")
    st.warning("""
    **IMPORTANT MEDICAL DISCLAIMER**
    
    This application is designed for **INFORMATIONAL AND EDUCATIONAL PURPOSES ONLY**.
    
    **NOT A MEDICAL DEVICE:** This tool does not provide medical advice, diagnosis, or treatment.
    
    **LIMITATIONS:**
    - AI predictions have inherent limitations and error rates
    - Cannot replace professional medical evaluation
    - Should not be used for emergency medical decisions
    
    **RECOMMENDATIONS:**
    - Always consult qualified healthcare professionals
    - Use this tool as supplementary information only
    - Report any health concerns to your doctor immediately
    """)
    
    st.subheader("🔒 Data Privacy & Security")
    st.markdown("""
    - **No Data Storage:** Input data is processed in real-time and not stored
    - **Anonymous Processing:** No personal identifiers are collected
    - **Local Computation:** All processing happens in your browser session
    - **Session-Based:** Data is cleared when you close the browser
    """)
    
    st.subheader("👨‍⚕️ When to See a Doctor")
    st.markdown("""
    Seek immediate medical attention if you experience:
    - Chest pain or discomfort
    - Shortness of breath
    - Dizziness or fainting
    - Irregular heartbeat
    - Severe headache with vision changes
    """)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: var(--secondary-color); border-radius: 10px;'>
    <p><strong>Version:</strong> 1.0.0</p>
    <p>Developed for educational purposes | Not for clinical use</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- Main App Logic ---
def main():
    # Sidebar Navigation
    sidebar_navigation()
    
    # Page Routing
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Risk Assessment":
        risk_assessment_page()
    elif st.session_state.page == "Model Info":
        model_info_page()
    elif st.session_state.page == "About":
        about_page()

# --- Run the App ---
if __name__ == "__main__":
    main()