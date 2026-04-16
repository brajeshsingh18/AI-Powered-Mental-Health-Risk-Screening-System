import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MindScan — Mental Health Risk Analyser",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #060b14;
    color: #dde6f0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 3rem 2.5rem; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080e1a 0%, #050b14 100%);
    border-right: 1px solid #152236;
}
[data-testid="stSidebar"] .block-container { padding: 1.2rem 0.8rem; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0b1a2e 0%, #080f1e 50%, #060c18 100%);
    border: 1px solid #162840;
    border-radius: 20px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(0,200,255,0.07) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 15%;
    width: 380px; height: 180px;
    background: radial-gradient(ellipse, rgba(120,50,240,0.06) 0%, transparent 65%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.7rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00c8ff 0%, #a855f7 50%, #00c8ff 100%);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.35rem 0;
    letter-spacing: -1.5px;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #445a70;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-desc {
    font-size: 0.92rem;
    color: #6a8aaa;
    max-width: 600px;
    line-height: 1.6;
}

/* Stat cards */
.stat-grid { display: flex; gap: 1rem; margin-bottom: 1.8rem; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 150px;
    background: linear-gradient(135deg, #0c1a2c, #09131f);
    border: 1px solid #152438;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    transition: border-color 0.25s, transform 0.2s;
}
.stat-card:hover { border-color: #00c8ff33; transform: translateY(-2px); }
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #3a5a7a;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    margin-bottom: 0.4rem;
}
.stat-value { font-size: 1.75rem; font-weight: 800; color: #00c8ff; }
.stat-sub { font-size: 0.72rem; color: #3a5a7a; margin-top: 0.15rem; }

/* Section header */
.sec-header {
    font-size: 0.78rem;
    font-weight: 700;
    color: #00c8ff;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    margin: 1.6rem 0 0.8rem 0;
    padding-left: 0.7rem;
    border-left: 2px solid #a855f7;
}

/* Result containers */
.result-wrap {
    border-radius: 16px;
    padding: 2rem 2.2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-high { background: linear-gradient(135deg,#2a0808,#1a0404); border:1px solid #ff3d3d55; }
.result-medium { background: linear-gradient(135deg,#2a1608,#1a0e04); border:1px solid #ff8c2255; }
.result-low { background: linear-gradient(135deg,#082a14,#041a0c); border:1px solid #00ff7755; }
.result-title { font-size: 1.6rem; font-weight: 800; margin-bottom: 0.3rem; }
.result-desc { font-family:'DM Mono',monospace; font-size:0.82rem; color:#7a9ab8; margin-top:0.5rem; }

/* Risk badge */
.badge {
    display:inline-block; border-radius:20px;
    padding:0.25rem 0.9rem;
    font-family:'DM Mono',monospace; font-size:0.72rem;
    letter-spacing:1px; font-weight:500;
}
.badge-high { background:#ff1a1a22; border:1px solid #ff3d3d66; color:#ff6060; }
.badge-medium { background:#ff7a0022; border:1px solid #ff8c2266; color:#ffaa44; }
.badge-low { background:#00ff5522; border:1px solid #00ff7766; color:#44ff88; }

/* Factor tags */
.factor-tag {
    display:inline-block; margin:0.2rem;
    background:#0c1a2c; border:1px solid #1a3050;
    border-radius:20px; padding:0.22rem 0.8rem;
    font-family:'DM Mono',monospace; font-size:0.72rem; color:#5a8aaa;
}

/* Inputs */
div[data-baseweb="select"] > div {
    background:#0c1929 !important; border-color:#18304a !important;
    color:#dde6f0 !important; border-radius:10px !important;
}
div[data-baseweb="input"] input {
    background:#0c1929 !important; color:#dde6f0 !important;
}
.stSlider > div > div > div > div { background:#00c8ff !important; }
[data-testid="stSlider"] > div > div { background:#0c1929 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg,#00c8ff18,#a855f718) !important;
    border: 1px solid #00c8ff55 !important;
    color: #00c8ff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 1.5px !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.8rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#00c8ff33,#a855f733) !important;
    border-color: #00c8ff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(0,200,255,0.18) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background:#080f1a; border-radius:12px; padding:4px; gap:3px;
}
.stTabs [data-baseweb="tab"] {
    background:transparent; color:#3a5a7a; border-radius:9px;
    font-family:'Syne',sans-serif; font-weight:600; font-size:0.85rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#00c8ff18,#a855f718) !important;
    color: #00c8ff !important;
}

/* Plotly charts bg */
.js-plotly-plot .plotly { background:transparent !important; }

/* Sidebar logo area */
.sidebar-logo {
    text-align:center; padding:1rem 0 1.5rem 0;
    border-bottom:1px solid #152236; margin-bottom:1.2rem;
}
.sidebar-logo-icon { font-size:2.4rem; margin-bottom:0.3rem; }
.sidebar-logo-title {
    font-family:'Syne',sans-serif; font-weight:800;
    font-size:1.3rem; color:#00c8ff; letter-spacing:-0.5px;
}
.sidebar-logo-sub {
    font-family:'DM Mono',monospace; font-size:0.62rem;
    color:#334a60; letter-spacing:2.5px; text-transform:uppercase;
}

/* Warning/info boxes */
.info-box {
    background:#0c1929; border:1px solid #182a40;
    border-radius:12px; padding:1rem 1.2rem;
    font-size:0.85rem; color:#6a8aaa; line-height:1.5;
    margin:0.8rem 0;
}

hr { border-color:#12233a; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("clf.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("ohe.pkl", "rb") as f:
        ohe = pickle.load(f)
    return clf, ohe

try:
    clf, ohe = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Could not load model files: {e}. Make sure clf.pkl and ohe.pkl are in the same directory as app.py.")

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
CITIES = ['Agra','Ahmedabad','Bangalore','Bhopal','Chennai','Delhi','Faridabad',
          'Ghaziabad','Hyderabad','Indore','Jaipur','Kalyan','Kanpur','Kolkata',
          'Lucknow','Ludhiana','Meerut','Mumbai','Nagpur','Nashik','Patna',
          'Pune','Rajkot','Srinagar','Surat','Thane','Vadodara','Varanasi',
          'Vasai-Virar','Visakhapatnam']

PROFESSIONS = ['Student','Civil Engineer','Architect','UX/UI Designer','Digital Marketer',
               'Content Writer','Educational Consultant','Teacher','Manager','Chef',
               'Doctor','Lawyer','Entrepreneur','Pharmacist']

SLEEP_MAP = {
    'Less than 5 hours': 1,
    '5-6 hours': 2,
    '7-8 hours': 3,
    'More than 8 hours': 4
}

DEGREES = ['B.Arch','B.Com','B.Ed','B.Pharm','B.Tech','BA','BBA','BCA','BE','BHM',
           'BSc','Class 12','LLB','LLM','M.Com','M.Ed','M.Pharm','M.Tech','MA',
           'MBA','MBBS','MCA','MD','ME','MHM','MSc','Others','PhD']

DIETARY = ['Healthy', 'Moderate', 'Unhealthy', 'Others']

CAT_COLS  = ['Gender','City','Profession','Dietary Habits','Degree',
             'Have you ever had suicidal thoughts ?','Family History of Mental Illness']

NUM_COLS  = ['Age','Academic Pressure','Work Pressure','CGPA',
             'Study Satisfaction','Job Satisfaction','Sleep Duration',
             'Work/Study Hours','Financial Stress']

# ─────────────────────────────────────────────
#  HELPER: build feature vector
# ─────────────────────────────────────────────
def build_input(inputs: dict) -> pd.DataFrame:
    """
    Replicates the exact notebook preprocessing:
    1. OHE on 7 categorical columns
    2. Concat with numeric columns
    3. Align to clf.feature_names_in_
    """
    # Numeric part
    num_df = pd.DataFrame([[
        inputs['Age'], inputs['Academic Pressure'], inputs['Work Pressure'],
        inputs['CGPA'], inputs['Study Satisfaction'], inputs['Job Satisfaction'],
        SLEEP_MAP[inputs['Sleep Duration']], inputs['Work/Study Hours'],
        inputs['Financial Stress']
    ]], columns=NUM_COLS)

    # Categorical part → OHE
    cat_df = pd.DataFrame([[
        inputs['Gender'], inputs['City'], inputs['Profession'],
        inputs['Dietary Habits'], inputs['Degree'],
        inputs['Have you ever had suicidal thoughts ?'],
        inputs['Family History of Mental Illness']
    ]], columns=CAT_COLS)

    ohe_arr = ohe.transform(cat_df)
    ohe_cols = ohe.get_feature_names_out(CAT_COLS)
    ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols)

    combined = pd.concat([num_df.reset_index(drop=True),
                          ohe_df.reset_index(drop=True)], axis=1)

    # Align to model's exact column order, fill missing with 0
    model_cols = clf.feature_names_in_
    combined = combined.reindex(columns=model_cols, fill_value=0)
    return combined


def risk_level(prob: float):
    if prob >= 0.65:
        return "HIGH", "#ff4d4d", "result-high", "badge-high", "⚠️"
    elif prob >= 0.40:
        return "MODERATE", "#ffaa44", "result-medium", "badge-medium", "⚡"
    else:
        return "LOW", "#44ff88", "result-low", "badge-low", "✅"


def make_gauge(prob: float, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={'suffix': '%', 'font': {'size': 36, 'color': color,
                                         'family': 'Syne'}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickcolor': '#2a3a4a',
                'tickfont': {'color': '#3a5570'},
                'showticklabels': True
            },
            'bar': {'color': color, 'thickness': 0.22},
            'bgcolor': '#0c1929',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(0,255,85,0.1)'},
                {'range': [40, 65], 'color': 'rgba(255,140,0,0.1)'},
                {'range': [65, 100], 'color': 'rgba(255,0,0,0.1)'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.75,
                'value': round(prob * 100, 1)
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(t=20, b=10, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#dde6f0',
    )
    return fig


def make_factor_bar(inputs: dict):
    """Simple risk factor visualisation based on user inputs."""
    factors = {
        'Academic Pressure': inputs['Academic Pressure'] / 5,
        'Financial Stress':  inputs['Financial Stress'] / 5,
        'Work/Study Hours':  inputs['Work/Study Hours'] / 12,
        'Work Pressure':     inputs['Work Pressure'] / 5,
        'Study Satisfaction': 1 - (inputs['Study Satisfaction'] / 5),
        'Sleep Quality':     1 - (SLEEP_MAP[inputs['Sleep Duration']] / 4),
    }
    labels = list(factors.keys())
    values = [round(v * 100, 1) for v in factors.values()]
    colors = ['#ff4d4d' if v > 60 else '#ffaa44' if v > 35 else '#00c8ff'
              for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f'{v}%' for v in values],
        textposition='outside',
        textfont=dict(color='#6a8aaa', family='DM Mono', size=11),
        hovertemplate='%{y}: %{x}%<extra></extra>',
    ))
    fig.update_layout(
        height=270,
        margin=dict(t=10, b=10, l=10, r=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 120], showgrid=False, zeroline=False,
                   showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(color='#6a8aaa',
                   family='DM Mono', size=11)),
        font_color='#dde6f0',
    )
    return fig


def make_radar(inputs: dict):
    categories = ['Academic\nPressure','Financial\nStress','Work/Study\nHours',
                  'Work\nPressure','Sleep\nDeficit','Dietary\nRisk']
    values = [
        inputs['Academic Pressure'] / 5 * 100,
        inputs['Financial Stress'] / 5 * 100,
        inputs['Work/Study Hours'] / 12 * 100,
        inputs['Work Pressure'] / 5 * 100,
        (1 - SLEEP_MAP[inputs['Sleep Duration']] / 4) * 100,
        {'Healthy': 10, 'Moderate': 45, 'Unhealthy': 85, 'Others': 50
         }.get(inputs['Dietary Habits'], 50),
    ]
    values_closed = values + [values[0]]
    cats_closed   = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed, theta=cats_closed,
        fill='toself',
        fillcolor='rgba(0,200,255,0.07)',
        line=dict(color='#00c8ff', width=2),
        marker=dict(color='#00c8ff', size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor='#162840', tickcolor='#162840',
                            tickfont=dict(color='#2a4060', size=8),
                            linecolor='#162840'),
            angularaxis=dict(gridcolor='#162840', linecolor='#162840',
                             tickfont=dict(color='#5a7a9a', family='DM Mono',
                                           size=9)),
        ),
        height=300,
        margin=dict(t=20, b=20, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        font_color='#dde6f0',
    )
    return fig


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🧠</div>
        <div class="sidebar-logo-title">MindScan</div>
        <div class="sidebar-logo-sub">Mental Health AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-header">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["🔍  Risk Analyser", "📊  Data Insights", "ℹ️  About"],
                    label_visibility="collapsed")

    st.markdown('<div class="sec-header">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <b style="color:#00c8ff;">Model:</b> Gradient Boosting Classifier<br>
        <b style="color:#00c8ff;">Accuracy:</b> ~84.6%<br>
        <b style="color:#00c8ff;">Dataset:</b> 27,901 students<br>
        <b style="color:#00c8ff;">Features:</b> 113 (after encoding)<br>
        <b style="color:#00c8ff;">Target:</b> Depression Risk (0/1)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:0.6rem; font-size:0.75rem; color:#334a60;">
        ⚕️ This tool is for educational purposes only. 
        Always consult a qualified mental health professional for diagnosis.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE: RISK ANALYSER
# ─────────────────────────────────────────────
if page == "🔍  Risk Analyser":

    st.markdown("""
    <div class="hero">
        <div class="hero-sub">AI-Powered Mental Health Screening</div>
        <div class="hero-title">MindScan Analyser</div>
        <div class="hero-desc">
            Enter student profile details below and let the AI model assess mental health 
            risk factors using a trained Gradient Boosting classifier.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Form ──────────────────────────────
    with st.form("prediction_form"):

        # Personal Info
        st.markdown('<div class="sec-header">👤 Personal Information</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c2:
            age = st.slider("Age", 18, 60, 22)
        with c3:
            city = st.selectbox("City", CITIES, index=CITIES.index('Delhi'))

        c4, c5 = st.columns(2)
        with c4:
            profession = st.selectbox("Profession", PROFESSIONS)
        with c5:
            degree = st.selectbox("Degree", DEGREES, index=DEGREES.index('B.Tech'))

        # Academic / Work
        st.markdown('<div class="sec-header">📚 Academic & Work Factors</div>',
                    unsafe_allow_html=True)
        c6, c7, c8 = st.columns(3)
        with c6:
            academic_pressure = st.slider("Academic Pressure", 0, 5, 3,
                                          help="0 = None, 5 = Extreme")
        with c7:
            work_pressure = st.slider("Work Pressure", 0, 5, 0,
                                      help="0 = None, 5 = Extreme")
        with c8:
            work_study_hours = st.slider("Work/Study Hours per Day", 0, 12, 7)

        c9, c10, c11 = st.columns(3)
        with c9:
            cgpa = st.number_input("CGPA", 0.0, 10.0, 7.5, step=0.01,
                                   format="%.2f")
        with c10:
            study_satisfaction = st.slider("Study Satisfaction", 0, 5, 3,
                                           help="0 = Very Low, 5 = Very High")
        with c11:
            job_satisfaction = st.slider("Job Satisfaction", 0, 5, 0,
                                         help="0 = N/A or Very Low, 5 = Very High")

        # Lifestyle
        st.markdown('<div class="sec-header">🌙 Lifestyle & Wellbeing</div>',
                    unsafe_allow_html=True)
        c12, c13, c14 = st.columns(3)
        with c12:
            sleep_duration = st.selectbox("Sleep Duration",
                                          list(SLEEP_MAP.keys()), index=1)
        with c13:
            dietary_habits = st.selectbox("Dietary Habits", DIETARY)
        with c14:
            financial_stress = st.slider("Financial Stress", 1, 5, 2,
                                         help="1 = Low, 5 = Extreme")

        # Risk Indicators
        st.markdown('<div class="sec-header">🚨 Risk Indicators</div>',
                    unsafe_allow_html=True)
        c15, c16 = st.columns(2)
        with c15:
            suicidal_thoughts = st.selectbox("History of Suicidal Thoughts?",
                                             ["No", "Yes"])
        with c16:
            family_history = st.selectbox("Family History of Mental Illness?",
                                          ["No", "Yes"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍  ANALYSE MENTAL HEALTH RISK")

    # ── Results ──────────────────────────────
    if submitted and models_loaded:
        inputs = {
            'Gender': gender, 'Age': age, 'City': city,
            'Profession': profession, 'Degree': degree,
            'Academic Pressure': academic_pressure,
            'Work Pressure': work_pressure,
            'Work/Study Hours': work_study_hours,
            'CGPA': cgpa,
            'Study Satisfaction': study_satisfaction,
            'Job Satisfaction': job_satisfaction,
            'Sleep Duration': sleep_duration,
            'Dietary Habits': dietary_habits,
            'Financial Stress': financial_stress,
            'Have you ever had suicidal thoughts ?': suicidal_thoughts,
            'Family History of Mental Illness': family_history,
        }

        try:
            X = build_input(inputs)
            prob = clf.predict_proba(X)[0][1]
            label, color, res_class, badge_class, icon = risk_level(prob)

            st.markdown("---")
            st.markdown('<div class="sec-header">📋 Analysis Results</div>',
                        unsafe_allow_html=True)

            col_gauge, col_result = st.columns([1, 1.8])

            with col_gauge:
                st.plotly_chart(make_gauge(prob, color),
                                use_container_width=True, config={'displayModeBar': False})

            with col_result:
                if label == "HIGH":
                    desc = "Significant risk indicators detected. Immediate professional support is strongly recommended."
                elif label == "MODERATE":
                    desc = "Some risk factors present. Consider speaking with a counsellor or mental health professional."
                else:
                    desc = "Low risk indicators detected. Continue healthy habits and monitor wellbeing regularly."

                st.markdown(f"""
                <div class="result-wrap {res_class}">
                    <div style="font-size:2.5rem; margin-bottom:0.4rem;">{icon}</div>
                    <div class="result-title" style="color:{color};">
                        {label} RISK
                    </div>
                    <div style="font-size:2rem; font-weight:800; color:{color};
                                font-family:'DM Mono',monospace; margin:0.4rem 0;">
                        {round(prob*100,1)}%
                    </div>
                    <div class="result-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

            # Charts row
            st.markdown('<div class="sec-header">📊 Risk Factor Breakdown</div>',
                        unsafe_allow_html=True)
            col_bar, col_radar = st.columns(2)
            with col_bar:
                st.markdown("**Contributing Factors**")
                st.plotly_chart(make_factor_bar(inputs),
                                use_container_width=True,
                                config={'displayModeBar': False})
            with col_radar:
                st.markdown("**Risk Profile Radar**")
                st.plotly_chart(make_radar(inputs),
                                use_container_width=True,
                                config={'displayModeBar': False})

            # Key flags
            st.markdown('<div class="sec-header">🏷️ Detected Risk Flags</div>',
                        unsafe_allow_html=True)
            flags = []
            if academic_pressure >= 4: flags.append("High Academic Pressure")
            if financial_stress  >= 4: flags.append("High Financial Stress")
            if work_study_hours  >= 10: flags.append("Excessive Study/Work Hours")
            if suicidal_thoughts == "Yes": flags.append("Suicidal Thoughts History")
            if family_history    == "Yes": flags.append("Family Mental Illness History")
            if SLEEP_MAP[sleep_duration] == 1: flags.append("Severe Sleep Deprivation")
            if dietary_habits == "Unhealthy": flags.append("Unhealthy Diet")
            if study_satisfaction <= 1: flags.append("Very Low Study Satisfaction")
            if cgpa < 5.5: flags.append("Low Academic Performance")

            if flags:
                tags_html = "".join(f'<span class="factor-tag">⚠️ {f}</span>' for f in flags)
            else:
                tags_html = '<span class="factor-tag">✅ No major risk flags detected</span>'
            st.markdown(tags_html, unsafe_allow_html=True)

            # Recommendations
            st.markdown('<div class="sec-header">💡 Recommendations</div>',
                        unsafe_allow_html=True)
            recs = []
            if academic_pressure >= 4:
                recs.append(("📚", "Academic Pressure",
                              "Talk to an academic advisor or counsellor about managing your workload."))
            if SLEEP_MAP[sleep_duration] <= 2:
                recs.append(("😴", "Sleep",
                              "Aim for 7–8 hours of quality sleep. Poor sleep significantly worsens mental health."))
            if financial_stress >= 4:
                recs.append(("💰", "Financial Stress",
                              "Explore scholarship options, part-time work, or speak to a financial advisor."))
            if dietary_habits in ['Unhealthy', 'Others']:
                recs.append(("🥗", "Nutrition",
                              "Balanced meals can meaningfully improve mood and cognitive performance."))
            if work_study_hours >= 10:
                recs.append(("⏰", "Work-Life Balance",
                              "Schedule regular breaks. The Pomodoro technique and structured rest improve productivity."))
            if suicidal_thoughts == "Yes":
                recs.append(("🆘", "Immediate Support",
                              "Please reach out to iCall (9152987821) or Vandrevala Foundation (1860-2662-345) immediately."))

            if not recs:
                recs.append(("✅", "Keep it up",
                              "Your profile shows low risk. Maintain healthy habits, exercise regularly, and stay connected."))

            cols = st.columns(min(len(recs), 3))
            for i, (icon2, title, text) in enumerate(recs):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="stat-card" style="min-height:110px;">
                        <div style="font-size:1.4rem; margin-bottom:0.3rem;">{icon2}</div>
                        <div style="font-weight:700; color:#00c8ff; font-size:0.85rem;
                                    margin-bottom:0.3rem;">{title}</div>
                        <div style="font-size:0.78rem; color:#5a7a9a; line-height:1.5;">{text}</div>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ─────────────────────────────────────────────
#  PAGE: DATA INSIGHTS
# ─────────────────────────────────────────────
elif page == "📊  Data Insights":

    st.markdown("""
    <div class="hero">
        <div class="hero-sub">Exploratory Data Analysis</div>
        <div class="hero-title">Dataset Insights</div>
        <div class="hero-desc">
            Visual exploration of the Student Depression Dataset used to train the model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        df = pd.read_csv("Student Depression Dataset.csv")
        # Drop rows with critical NaN
        df = df.dropna(subset=['Financial Stress'])
        return df

    try:
        df = load_data()

        # Stat row
        dep_rate = round(df['Depression'].mean() * 100, 1)
        avg_age  = round(df['Age'].mean(), 1)
        avg_cgpa = round(df['CGPA'].mean(), 2)
        avg_sleep_num = df['Sleep Duration'].map(
            {'Less than 5 hours': 1,'5-6 hours': 2,'7-8 hours': 3,
             'More than 8 hours': 4}).mean()
        sleep_label = ['< 5 hrs','5-6 hrs','7-8 hrs','> 8 hrs'][round(avg_sleep_num)-1]

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Total Records</div>
                <div class="stat-value">{len(df):,}</div>
                <div class="stat-sub">students surveyed</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Depression Rate</div>
                <div class="stat-value" style="color:#ff6060;">{dep_rate}%</div>
                <div class="stat-sub">of dataset positive</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average Age</div>
                <div class="stat-value">{avg_age}</div>
                <div class="stat-sub">years</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average CGPA</div>
                <div class="stat-value">{avg_cgpa}</div>
                <div class="stat-sub">out of 10</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Sleep</div>
                <div class="stat-value" style="color:#a855f7;">{sleep_label}</div>
                <div class="stat-sub">most common</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📈  Distributions", "🔗  Correlations", "🧩  Breakdowns"])

        CHART_BG = 'rgba(0,0,0,0)'
        GRID_COL = '#12233a'
        TICK_COL = '#3a5570'
        TEXT_COL = '#dde6f0'

        def style_fig(fig, h=350):
            fig.update_layout(
                height=h,
                paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                font=dict(family='Syne', color=TEXT_COL),
                margin=dict(t=30, b=20, l=10, r=10),
                xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL,
                           tickfont=dict(color=TICK_COL, family='DM Mono', size=10)),
                yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL,
                           tickfont=dict(color=TICK_COL, family='DM Mono', size=10)),
            )
            return fig

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(df, x='Age', color='Depression',
                                   color_discrete_map={0:'#00c8ff', 1:'#ff4d4d'},
                                   barmode='overlay', opacity=0.75,
                                   title='Age Distribution by Depression Status',
                                   labels={'Depression':'Depression (1=Yes)'})
                st.plotly_chart(style_fig(fig), use_container_width=True,
                                config={'displayModeBar': False})
            with c2:
                fig = px.histogram(df, x='CGPA', color='Depression',
                                   color_discrete_map={0:'#00c8ff', 1:'#ff4d4d'},
                                   barmode='overlay', opacity=0.75,
                                   title='CGPA Distribution by Depression Status',
                                   labels={'Depression':'Depression (1=Yes)'})
                st.plotly_chart(style_fig(fig), use_container_width=True,
                                config={'displayModeBar': False})

            c3, c4 = st.columns(2)
            with c3:
                sleep_dep = df.groupby(['Sleep Duration','Depression']).size().reset_index(name='Count')
                fig = px.bar(sleep_dep, x='Sleep Duration', y='Count', color='Depression',
                             color_discrete_map={0:'#00c8ff', 1:'#ff4d4d'},
                             barmode='group', title='Sleep Duration vs Depression',
                             category_orders={'Sleep Duration':list(SLEEP_MAP.keys())})
                st.plotly_chart(style_fig(fig), use_container_width=True,
                                config={'displayModeBar': False})
            with c4:
                diet_dep = df.groupby(['Dietary Habits','Depression']).size().reset_index(name='Count')
                fig = px.bar(diet_dep, x='Dietary Habits', y='Count', color='Depression',
                             color_discrete_map={0:'#00c8ff', 1:'#ff4d4d'},
                             barmode='group', title='Dietary Habits vs Depression')
                st.plotly_chart(style_fig(fig), use_container_width=True,
                                config={'displayModeBar': False})

        with tab2:
            num_features = ['Age','Academic Pressure','Work Pressure','CGPA',
                            'Study Satisfaction','Job Satisfaction',
                            'Work/Study Hours','Financial Stress','Depression']
            corr = df[num_features].corr()
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                colorscale=[[0,'#0c1929'],[0.5,'#162840'],[1,'#00c8ff']],
                text=corr.round(2).values,
                texttemplate='%{text}',
                textfont=dict(size=9, color='#dde6f0', family='DM Mono'),
                hoverongaps=False,
                showscale=True,
            ))
            fig.update_layout(
                title='Feature Correlation Heatmap',
                height=480,
                paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                font=dict(family='Syne', color=TEXT_COL),
                margin=dict(t=40, b=10, l=10, r=10),
                xaxis=dict(tickfont=dict(color=TICK_COL,family='DM Mono',size=9)),
                yaxis=dict(tickfont=dict(color=TICK_COL,family='DM Mono',size=9)),
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={'displayModeBar': False})

        with tab3:
            c5, c6 = st.columns(2)
            with c5:
                gen_dep = df.groupby('Gender')['Depression'].mean().reset_index()
                gen_dep['Rate %'] = (gen_dep['Depression']*100).round(1)
                fig = px.bar(gen_dep, x='Gender', y='Rate %',
                             color='Gender',
                             color_discrete_map={'Male':'#00c8ff','Female':'#a855f7'},
                             title='Depression Rate by Gender',
                             text='Rate %')
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                st.plotly_chart(style_fig(fig), use_container_width=True,
                                config={'displayModeBar': False})
            with c6:
                sui_dep = df.groupby('Have you ever had suicidal thoughts ?')['Depression'].mean().reset_index()
                sui_dep.columns = ['Suicidal Thoughts','Rate']
                sui_dep['Rate %'] = (sui_dep['Rate']*100).round(1)
                fig = px.bar(sui_dep, x='Suicidal Thoughts', y='Rate %',
                             color='Suicidal Thoughts',
                             color_discrete_map={'Yes':'#ff4d4d','No':'#00c8ff'},
                             title='Depression Rate by Suicidal Thoughts History',
                             text='Rate %')
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                st.plotly_chart(style_fig(fig), use_container_width=True,
                                config={'displayModeBar': False})

            # Academic pressure scatter
            fig = px.scatter(df.sample(2000, random_state=42),
                             x='Academic Pressure', y='CGPA',
                             color='Depression',
                             color_discrete_map={0:'#00c8ff', 1:'#ff4d4d'},
                             opacity=0.5, size_max=5,
                             title='Academic Pressure vs CGPA (sample of 2000)',
                             labels={'Depression':'Depression (1=Yes)'})
            fig.update_layout(height=370,
                              paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                              font=dict(family='Syne', color=TEXT_COL),
                              margin=dict(t=40,b=10,l=10,r=10),
                              xaxis=dict(gridcolor=GRID_COL,
                                         tickfont=dict(color=TICK_COL,family='DM Mono')),
                              yaxis=dict(gridcolor=GRID_COL,
                                         tickfont=dict(color=TICK_COL,family='DM Mono')))
            st.plotly_chart(fig, use_container_width=True,
                            config={'displayModeBar': False})

    except FileNotFoundError:
        st.warning("Place `Student Depression Dataset.csv` in the same folder as app.py to view insights.")

# ─────────────────────────────────────────────
#  PAGE: ABOUT
# ─────────────────────────────────────────────
else:
    st.markdown("""
    <div class="hero">
        <div class="hero-sub">Project Information</div>
        <div class="hero-title">About MindScan</div>
        <div class="hero-desc">
            An AI-powered mental health risk screening tool built as a data visualisation project.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-header">🛠️ Tech Stack</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <b style="color:#00c8ff;">Frontend / UI</b><br>
            Streamlit · Custom CSS · Plotly<br><br>
            <b style="color:#00c8ff;">ML Backend</b><br>
            scikit-learn · Gradient Boosting Classifier<br><br>
            <b style="color:#00c8ff;">Data Processing</b><br>
            Pandas · NumPy · OneHotEncoder · SHAP<br><br>
            <b style="color:#00c8ff;">Dataset</b><br>
            Student Depression Dataset (27,901 records, 18 features)
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="sec-header">📐 Model Pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <b style="color:#a855f7;">1. Data Cleaning</b><br>
            Null imputation · Outlier removal · Feature engineering<br><br>
            <b style="color:#a855f7;">2. Encoding</b><br>
            OneHotEncoder on 7 categorical columns → 104 binary features<br><br>
            <b style="color:#a855f7;">3. Training</b><br>
            80/20 train-test split · GradientBoostingClassifier<br><br>
            <b style="color:#a855f7;">4. Evaluation</b><br>
            Accuracy: 84.6% · SHAP explainability for feature importance
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-header">⚠️ Disclaimer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box" style="border-color:#ff4d4d33; background:linear-gradient(135deg,#1a060622,#0c040422);">
        This application is developed <b>purely for educational and academic purposes</b> 
        as part of a Data Visualisation Project. It is <b>not a clinical diagnostic tool</b> 
        and should not be used as a substitute for professional mental health evaluation. 
        If you or someone you know is struggling, please contact a qualified mental health 
        professional or a crisis helpline such as <b style="color:#ff6060;">iCall: 9152987821</b>.
    </div>
    """, unsafe_allow_html=True)


