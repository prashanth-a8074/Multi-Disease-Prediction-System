import streamlit as st
import pandas as pd
import numpy as np
import os, json, sqlite3, datetime, io

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

import sys
import importlib

# Force reload helper modules to bypass Streamlit cache locks
for module_name in ['src.predictor', 'src.ai_assistant', 'src.recommendation']:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

from src.predictor import predict_disease
from src.ai_assistant import generate_ai_response
from src.recommendation import HOSPITAL_DATABASE

# =============================================================================
# 1. PAGE & THEME CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Multi-Disease Prediction System", page_icon="🩺", layout="wide")

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;600&display=swap');
    html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; background: #020818 !important; color: #cdd6f4; }
    h1 { font-family: 'Space Grotesk', sans-serif; background: linear-gradient(135deg, #93c5fd, #a78bfa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; border-bottom: 2px solid rgba(147,197,253,0.2); padding-bottom: 8px; }
    h2, h3 { color: #60a5fa !important; }
    .med-card { background: linear-gradient(135deg, rgba(15,23,50,0.8), rgba(8,12,30,0.85)); border: 1px solid rgba(99,179,237,0.18); border-radius: 14px; padding: 18px; margin-bottom: 12px; }
    .result-display { border-radius: 14px; padding: 20px; text-align: center; margin: 15px 0; }
    .result-positive { background: rgba(239,68,68,0.2); border: 1px solid #ef4444; color: #fca5a5; }
    .result-negative { background: rgba(16,185,129,0.2); border: 1px solid #10b981; color: #6ee7b7; }
    [data-testid="stSidebar"] { background: #050c1a !important; border-right: 1px solid rgba(99,179,237,0.12) !important; }
    [data-testid="stSidebar"] .stRadio label:has(input[type="radio"]:checked) { background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2)) !important; color: #38bdf8 !important; font-weight: 700 !important; border-radius: 8px; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label > div:first-child { display: none !important; }
    div[data-baseweb="input"] input { background: rgba(8,15,36,0.8) !important; color: #e2e8f0 !important; }
    label { color: #93c5fd !important; font-weight: 600 !important; }
    .stButton > button { background: linear-gradient(135deg, #2563eb, #4f46e5) !important; color: white !important; font-weight: 600 !important; border-radius: 8px !important; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# 2. DATABASE LAYER
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "prediction_history.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, disease TEXT, prediction TEXT, confidence REAL, inputs TEXT, timestamp TEXT)")
        conn.execute("INSERT OR IGNORE INTO users VALUES ('admin', 'admin123', 'admin'), ('user', 'user123', 'user')")

def auth(u, p):
    with get_db() as conn:
        res = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?", (u, p)).fetchone()
        return dict(res) if res else None

def register(u, p, r="user"):
    try:
        with get_db() as conn:
            conn.execute("INSERT INTO users VALUES (?, ?, ?)", (u, p, r))
            return True
    except sqlite3.IntegrityError:
        return False

def save_log(u, disease, pred, conf, inputs):
    with get_db() as conn:
        conn.execute("INSERT INTO history (username, disease, prediction, confidence, inputs, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                     (u, disease, pred, conf, json.dumps(inputs), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

DOCTORS = {
    "Diabetes": ("Endocrinologist / Diabetologist", "Specializes in hormone disorders and blood sugar management."),
    "Heart Disease": ("Cardiologist", "Manages coronary disorders and cardiovascular health."),
    "Liver Disease": ("Hepatologist / Gastroenterologist", "Focuses on liver dysfunction and enzyme metrics."),
    "Breast Cancer": ("Oncologist / Breast Surgeon", "Specializes in tissue pathology and oncology therapy."),
    "Parkinson's Disease": ("Neurologist", "Manages movement disorders and vocal tremor analysis.")
}

# =============================================================================
# 3. PDF REPORT GENERATOR
# =============================================================================
def pdf_report(username, disease, pred, conf, inputs, rec):
    if not REPORTLAB_AVAILABLE: return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=35, leftMargin=35, topMargin=35, bottomMargin=35)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("CLINICAL DIAGNOSTIC REPORT", ParagraphStyle('T', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1d4ed8'), alignment=1)),
        Spacer(1, 10),
        Table([
            [Paragraph("<b>Patient Name:</b>", styles['Normal']), Paragraph(inputs.get("Patient Name", username), styles['Normal']), Paragraph("<b>Date:</b>", styles['Normal']), Paragraph(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), styles['Normal'])],
            [Paragraph("<b>Test:</b>", styles['Normal']), Paragraph(disease, styles['Normal']), Paragraph("<b>Age/Gender:</b>", styles['Normal']), Paragraph(f"{inputs.get('Patient Age', 'N/A')} / {inputs.get('Patient Gender', 'N/A')}", styles['Normal'])]
        ], colWidths=[100, 160, 80, 160], style=[('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')), ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc'))]),
        Spacer(1, 12),
        Table([[Paragraph("<b>Result:</b>", styles['Normal']), Paragraph(f"<b>{pred}</b> ({conf:.1f}%)", styles['Normal'])]], colWidths=[100, 400], style=[('GRID', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1'))]),
        Spacer(1, 12),
        Paragraph(f"<b>Consultation Advice:</b> {rec}", styles['Normal'])
    ]
    doc.build(story)
    return buf.getvalue()

# =============================================================================
# 4. INFERENCE & RESULT RENDERER
# =============================================================================
def run_and_show(disease_key, display_name, raw_inputs, patient_name):
    pred_val, conf = predict_disease(disease_key, raw_inputs)
    outcome = "High Risk (Positive)" if pred_val == 1 else "Low Risk (Negative)"
    if "Breast Cancer" in display_name: outcome = "Malignant (High Risk)" if pred_val == 1 else "Benign (Low Risk)"
    elif "Parkinson" in display_name: outcome = "Affected (High Risk)" if pred_val == 1 else "Not Affected (Low Risk)"

    st.markdown(f"<div class='result-display {'result-positive' if pred_val==1 else 'result-negative'}'><h3>Outcome: {outcome}</h3><p>Confidence: <b>{conf:.2f}%</b></p></div>", unsafe_allow_html=True)
    save_log(st.session_state.username, display_name, outcome, conf, raw_inputs)
    
    spec, desc = DOCTORS.get(display_name, ("Specialist", "Consult physician."))
    rec_msg = f"Outcome: {outcome} ({conf:.1f}%). Recommended Specialist: {spec}. {desc}"
    st.info(f"👨‍⚕️ **Specialist Advice:** {rec_msg}")

    if REPORTLAB_AVAILABLE:
        pdf = pdf_report(st.session_state.username, display_name, outcome, conf, raw_inputs, rec_msg)
        st.download_button(f"📥 Download {display_name} PDF Report", pdf, f"{display_name.replace(' ','_')}_Report_{patient_name}.pdf", "application/pdf")

# Nav Callback Helper
def nav_to_predict():
    st.session_state.sb_nav = "🩺 Predict Diseases"

# =============================================================================
# 5. MAIN CONTROLLER & APPLICATION ROUTING
# =============================================================================
def main():
    init_db()
    inject_custom_css()

    for k, v in [("logged_in", False), ("username", None), ("role", None)]:
        if k not in st.session_state: st.session_state[k] = v

    page = "Home"

    with st.sidebar:
        st.markdown("<h2 style='text-align:center;'>MDPS</h2>", unsafe_allow_html=True)
        if st.session_state.logged_in:
            st.markdown(f"<div class='med-card'><p style='margin:0; font-size:11px;'>USER</p><b>{st.session_state.username}</b> ({st.session_state.role})</div>", unsafe_allow_html=True)
            opts = ["🏠 Home & Overview", "🩺 Predict Diseases", "🕒 Prediction History", "👨‍⚕️ Doctor Recommendations", "🤖 AI Health Assistant"]
            if st.session_state.role == "admin": opts.append("⚙️ Admin Dashboard")
            
            sel = st.radio("Navigation", opts, label_visibility="collapsed", key="sb_nav")
            
            if "History" in sel: page = "History"
            elif "Predict" in sel: page = "Predict"
            elif "Doctor" in sel: page = "Doctor"
            elif "AI" in sel: page = "AI"
            elif "Admin" in sel: page = "Admin"
            else: page = "Home"
            
            if st.button("🚪 Logout", key="btn_logout"):
                st.session_state.logged_in = False; st.rerun()

    if not st.session_state.logged_in:
        st.title("🔐 Diagnostics Portal Sign In")
        t1, t2 = st.tabs(["🔑 Login", "📝 Register"])
        with t1:
            with st.form("l_form"):
                u, p = st.text_input("Username", key="l_user"), st.text_input("Password", type="password", key="l_pass")
                if st.form_submit_button("Sign In"):
                    user = auth(u, p)
                    if user:
                        st.session_state.logged_in, st.session_state.username, st.session_state.role = True, user["username"], user["role"]
                        st.rerun()
                    else: st.error("Invalid credentials (user/user123, admin/admin123)")
        with t2:
            with st.form("r_form"):
                nu, np, nr = st.text_input("Choose Username", key="r_user"), st.text_input("Choose Password", type="password", key="r_pass"), st.selectbox("Role", ["user", "admin"], key="r_role")
                if st.form_submit_button("Register"):
                    if register(nu, np, nr): st.success("Account created! Please login.")
                    else: st.error("Username taken.")
        return

    # HOME PAGE
    if page == "Home":
        st.title("🏠 Multi-Disease Predictive Portal")
        st.markdown("""
        <div class='med-card'>
            <h3>🩺 Welcome to Clinical Machine Learning Diagnostics</h3>
            <p>Select <b>Predict Diseases</b> from the sidebar menu to launch predictive risk modeling across 5 clinical categories:</p>
            <ul>
                <li><b>Diabetes</b> (Logistic Regression)</li>
                <li><b>Heart Disease</b> (Logistic Regression)</li>
                <li><b>Liver Disease</b> (Random Forest)</li>
                <li><b>Breast Cancer Pathology</b> (Logistic Regression)</li>
                <li><b>Parkinson's Vocal Tremor Analysis</b> (Random Forest)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.button("🩺 Launch Disease Predictor Now", key="btn_home_launch", on_click=nav_to_predict)

    # PREDICT DISEASES PAGE
    elif page == "Predict":
        st.title("🩺 Multi-Disease Risk Prediction")
        st.markdown("<div class='med-card'><b>👤 Patient Profile Information</b></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        p_name = c1.text_input("Patient Name", value=st.session_state.username.capitalize() if st.session_state.username else "Patient", key="prof_name")
        p_age = c2.number_input("Age", 1, 120, 35, key="prof_age")
        p_gen = c3.selectbox("Gender", ["Male", "Female"], key="prof_gen")

        with st.expander("💡 View Sample Test Case Input Values"):
            st.write("- **Diabetes**: Preg 6, Glu 148, BP 72, Skin 35, Ins 0, BMI 33.6, DPF 0.627, Age 50\n- **Heart**: Age 63, Sex Male, CP Typical Angina, BP 145, Chol 233, FastingBS Yes, RestECG LVH, MaxHR 150, ExAng No, Oldpeak 2.3, Slope Downsloping, CA 0, Thal Normal\n- **Liver**: Age 65, Gender Male, TotalBili 0.7, DirectBili 0.1, AlkPhos 187, ALT 16, AST 18, Proteins 6.8, Albumin 3.3, AG_Ratio 0.9\n- **Breast Cancer**: Rad 17.99, Tex 10.38, Per 122.8, Area 1001, Smooth 0.1184, Comp 0.2776, Conc 0.3001, Pts 0.1471, Symm 0.2419, Frac 0.0787\n- **Parkinson's**: Fo 119.99, Fhi 157.30, Flo 75.00, Jitter 0.0078, Shimmer 0.0437, NHR 0.0221, HNR 21.03, RPDE 0.415, DFA 0.815, PPE 0.285")

        t1, t2, t3, t4, t5 = st.tabs(["🩺 Diabetes", "❤️ Heart", "🧪 Liver", "🎀 Breast Cancer", "🧠 Parkinson's"])

        with t1:
            c1, c2 = st.columns(2)
            preg = c1.number_input("Pregnancies", 0, 20, 1, key="db_preg")
            glu = c1.number_input("Glucose", 0, 300, 120, key="db_glu")
            bp = c1.number_input("BP", 0, 150, 70, key="db_bp")
            skin = c1.number_input("Skin Thickness", 0, 100, 20, key="db_skin")
            ins = c2.number_input("Insulin", 0, 900, 80, key="db_ins")
            bmi = c2.number_input("BMI", 0.0, 70.0, 27.5, key="db_bmi")
            dpf = c2.number_input("DPF", 0.0, 3.0, 0.47, format="%.3f", key="db_dpf")
            age = c2.number_input("Age", 1, 120, int(p_age), key="db_age")
            if st.button("Predict Diabetes", key="btn_db"):
                raw = {"Patient Name": p_name, "Patient Age": p_age, "Patient Gender": p_gen, "Pregnancies": preg, "Glucose": glu, "BloodPressure": bp, "SkinThickness": skin, "Insulin": ins, "BMI": bmi, "DiabetesPedigreeFunction": dpf, "Age": age}
                run_and_show("diabetes", "Diabetes", raw, p_name)

        with t2:
            c1, c2 = st.columns(2)
            h_age = c1.number_input("Age ", 1, 120, int(p_age), key="hd_age")
            h_sex = c1.selectbox("Gender ", ["Male", "Female"], index=0 if p_gen=="Male" else 1, key="hd_sex")
            h_cp = c1.selectbox("Chest Pain", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], key="hd_cp")
            h_bp = c1.number_input("Resting BP", 50, 220, 130, key="hd_bp")
            h_chol = c1.number_input("Cholesterol", 50, 600, 240, key="hd_chol")
            h_fbs = c1.selectbox("Fasting BS > 120", ["No (<= 120)", "Yes (> 120)"], key="hd_fbs")
            h_ecg = c1.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "LVH"], key="hd_ecg")
            h_hr = c2.number_input("Max HR", 50, 250, 150, key="hd_hr")
            h_ang = c2.selectbox("Exercise Angina", ["No", "Yes"], key="hd_ang")
            h_old = c2.number_input("Oldpeak", 0.0, 10.0, 1.0, key="hd_old")
            h_slope = c2.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"], key="hd_slope")
            h_ca = c2.selectbox("CA Vessels", [0, 1, 2, 3, 4], key="hd_ca")
            h_thal = c2.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"], key="hd_thal")
            if st.button("Predict Heart Disease", key="btn_hd"):
                raw = {"Patient Name": p_name, "Patient Age": p_age, "Patient Gender": p_gen, "Age": h_age, "Sex": h_sex, "Chest_Pain_Type": h_cp, "Resting_BP": h_bp, "Cholesterol": h_chol, "Fasting_BS": h_fbs, "Resting_ECG": h_ecg, "Max_Heart_Rate": h_hr, "Exercise_Angina": h_ang, "Oldpeak": h_old, "ST_Slope": h_slope, "CA": h_ca, "Thal": h_thal}
                run_and_show("heart", "Heart Disease", raw, p_name)

        with t3:
            c1, c2 = st.columns(2)
            l_age = c1.number_input("Age  ", 1, 120, int(p_age), key="ld_age")
            l_gen = c1.selectbox("Gender  ", ["Male", "Female"], index=0 if p_gen=="Male" else 1, key="ld_gen")
            l_tb = c1.number_input("Total Bilirubin", 0.1, 40.0, 1.0, key="ld_tb")
            l_db = c1.number_input("Direct Bilirubin", 0.1, 20.0, 0.3, key="ld_db")
            l_ap = c1.number_input("Alk Phos", 10, 3000, 150, key="ld_ap")
            l_sgpt = c2.number_input("ALT/SGPT", 5, 2000, 35, key="ld_sgpt")
            l_sgot = c2.number_input("AST/SGOT", 5, 2000, 40, key="ld_sgot")
            l_tp = c2.number_input("Total Proteins", 1.0, 15.0, 6.5, key="ld_tp")
            l_alb = c2.number_input("Albumin", 0.5, 10.0, 3.2, key="ld_alb")
            l_ag = c2.number_input("A/G Ratio", 0.1, 5.0, 0.9, key="ld_ag")
            if st.button("Predict Liver Disease", key="btn_ld"):
                raw = {"Patient Name": p_name, "Patient Age": p_age, "Patient Gender": p_gen, "Age": l_age, "Gender": l_gen, "Total_Bilirubin": l_tb, "Direct_Bilirubin": l_db, "Alkaline_Phosphotase": l_ap, "ALT": l_sgpt, "AST": l_sgot, "Total_Proteins": l_tp, "Albumin": l_alb, "AG_Ratio": l_ag}
                run_and_show("liver", "Liver Disease", raw, p_name)

        with t4:
            c1, c2 = st.columns(2)
            b_rad = c1.number_input("Radius Mean", 5.0, 40.0, 14.1, key="bc_rad")
            b_tex = c1.number_input("Texture Mean", 5.0, 50.0, 19.2, key="bc_tex")
            b_per = c1.number_input("Perimeter Mean", 30.0, 250.0, 92.0, key="bc_per")
            b_area = c1.number_input("Area Mean", 100.0, 3000.0, 654.0, key="bc_area")
            b_sm = c1.number_input("Smoothness Mean", 0.01, 0.3, 0.096, format="%.3f", key="bc_sm")
            b_co = c2.number_input("Compactness Mean", 0.01, 0.5, 0.104, format="%.3f", key="bc_co")
            b_cn = c2.number_input("Concavity Mean", 0.0, 0.5, 0.088, format="%.3f", key="bc_cn")
            b_pt = c2.number_input("Concave Points", 0.0, 0.3, 0.048, format="%.3f", key="bc_pt")
            b_sy = c2.number_input("Symmetry Mean", 0.05, 0.5, 0.181, format="%.3f", key="bc_sy")
            b_fr = c2.number_input("Fractal Dimension", 0.01, 0.15, 0.062, format="%.3f", key="bc_fr")
            if st.button("Predict Breast Cancer", key="btn_bc"):
                raw = {"Patient Name": p_name, "Patient Age": p_age, "Patient Gender": p_gen, "Radius_Mean": b_rad, "Texture_Mean": b_tex, "Perimeter_Mean": b_per, "Area_Mean": b_area, "Smoothness_Mean": b_sm, "Compactness_Mean": b_co, "Concavity_Mean": b_cn, "Concave_Points_Mean": b_pt, "Symmetry_Mean": b_sy, "Fractal_Dimension_Mean": b_fr}
                run_and_show("breast_cancer", "Breast Cancer", raw, p_name)

        with t5:
            c1, c2 = st.columns(2)
            p_fo = c1.number_input("Fo (Hz)", 50.0, 300.0, 197.27, key="pk_fo")
            p_fhi = c1.number_input("Fhi (Hz)", 50.0, 600.0, 244.74, key="pk_fhi")
            p_flo = c1.number_input("Flo (Hz)", 30.0, 300.0, 113.12, key="pk_flo")
            p_jit = c1.number_input("Jitter (%)", 0.0, 5.0, 0.006, format="%.5f", key="pk_jit")
            p_shim = c1.number_input("Shimmer", 0.0, 0.5, 0.03, format="%.5f", key="pk_shim")
            p_nhr = c2.number_input("NHR", 0.0, 2.0, 0.024, format="%.3f", key="pk_nhr")
            p_hnr = c2.number_input("HNR", 0.0, 50.0, 21.8, key="pk_hnr")
            p_rp = c2.number_input("RPDE", 0.0, 1.0, 0.48, key="pk_rp")
            p_dfa = c2.number_input("DFA", 0.0, 1.5, 0.71, key="pk_dfa")
            p_ppe = c2.number_input("PPE", 0.0, 1.0, 0.20, key="pk_ppe")
            if st.button("Predict Parkinson's", key="btn_pk"):
                raw = {"Patient Name": p_name, "Patient Age": p_age, "Patient Gender": p_gen, "MDVP:Fo(Hz)": p_fo, "MDVP:Fhi(Hz)": p_fhi, "MDVP:Flo(Hz)": p_flo, "MDVP:Jitter(%)": p_jit, "MDVP:Shimmer": p_shim, "NHR": p_nhr, "HNR": p_hnr, "RPDE": p_rp, "DFA": p_dfa, "PPE": p_ppe}
                run_and_show("parkinsons", "Parkinson's Disease", raw, p_name)

    # PREDICTION HISTORY PAGE
    elif page == "History":
        st.title("🕒 Historical Patient Predictions")
        with get_db() as conn:
            q = "SELECT * FROM history ORDER BY id DESC" if st.session_state.role=="admin" else "SELECT * FROM history WHERE username=? ORDER BY id DESC"
            df = pd.read_sql_query(q, conn, params=() if st.session_state.role=="admin" else (st.session_state.username,))
        
        if df.empty: st.info("No history logs found.")
        else:
            def get_detail(r, k, default):
                try: return json.loads(r["inputs"]).get(k, default)
                except: return default

            df["Patient Name"] = df.apply(lambda r: get_detail(r, "Patient Name", r["username"].capitalize()), axis=1)
            df["Patient Gender"] = df.apply(lambda r: get_detail(r, "Patient Gender", "N/A"), axis=1)
            df_clean = df.drop(columns=["inputs", "username"]).rename(columns={"id": "Log ID", "disease": "Target Disease", "prediction": "Diagnostic Outcome", "confidence": "Confidence Score (%)", "timestamp": "Timestamp"})
            st.dataframe(df_clean[["Log ID", "Patient Name", "Patient Gender", "Target Disease", "Diagnostic Outcome", "Confidence Score (%)", "Timestamp"]], use_container_width=True)

            log_id = st.selectbox("Select Log ID to download PDF report:", df["id"].tolist(), key="sb_hist_pdf")
            row = df[df["id"] == log_id].iloc[0]
            inputs_data = json.loads(row["inputs"])
            spec, _ = DOCTORS.get(row["disease"], ("Specialist", ""))
            rec_msg = f"Logged outcome: {row['prediction']} ({row['confidence']:.1f}%). Recommended Specialist: {spec}."
            if REPORTLAB_AVAILABLE:
                pdf = pdf_report(row["username"], row["disease"], row["prediction"], row["confidence"], inputs_data, rec_msg)
                st.download_button(f"📥 Download PDF Report (Log #{log_id})", pdf, f"Report_Log_{log_id}.pdf", "application/pdf", key=f"btn_hist_pdf_{log_id}")

    # DOCTOR RECOMMENDATIONS PAGE
    elif page == "Doctor":
        st.title("👨‍⚕️ Medical Specialist Finder")
        for disease, details in HOSPITAL_DATABASE.items():
            st.markdown(f"<div class='med-card'><h3>👨‍⚕️ {details['specialization']}</h3></div>", unsafe_allow_html=True)
            cols = st.columns(min(len(details["hospitals"]), 3))
            for idx, doc in enumerate(details["hospitals"][:3]):
                with cols[idx]:
                    st.markdown(f"<div style='background:rgba(15,23,42,0.6); padding:12px; border-radius:8px;'><h5 style='color:#38bdf8; margin:0;'>{doc['name']}</h5><p style='color:#cbd5e1; font-size:12px;'>🏥 {doc.get('hospital', doc.get('address'))}<br>⭐ Rating: {doc['rating']} | 📞 {doc['contact']}</p></div>", unsafe_allow_html=True)

    # AI HEALTH ASSISTANT PAGE
    elif page == "AI":
        st.title("🤖 AI Health Assistant")
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "👋 Hello! Ask me any questions about symptoms, prevention guidelines, or metrics for Diabetes, Heart Disease, Liver Disease, Breast Cancer, or Parkinson's."}]
        for m in st.session_state.messages: st.chat_message(m["role"]).write(m["content"])
        if prompt := st.chat_input("Ask a medical question...", key="chat_in"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            reply = generate_ai_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.chat_message("assistant").write(reply)

    # ADMIN DASHBOARD
    elif page == "Admin" and st.session_state.role == "admin":
        st.title("⚙️ System Administration")
        with get_db() as conn:
            u_cnt = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            h_cnt = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
        c1, c2 = st.columns(2)
        c1.metric("Users", u_cnt); c2.metric("History Logs", h_cnt)

if __name__ == "__main__":
    main()
