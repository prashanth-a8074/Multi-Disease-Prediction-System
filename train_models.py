import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Ensure directories exist
def create_directories():
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("models", exist_ok=True)

# 1. Diabetes Dataset Generator
def generate_diabetes_data():
    np.random.seed(42)
    n = 768
    preg = np.random.poisson(lam=3.8, size=n)
    glu = np.random.normal(loc=120.8, scale=32.0, size=n).astype(int)
    bp = np.random.normal(loc=69.1, scale=19.3, size=n).astype(int)
    skin = np.random.normal(loc=20.5, scale=15.9, size=n).astype(int)
    ins = np.random.normal(loc=79.8, scale=115.2, size=n).astype(int)
    
    preg[preg < 0] = 0
    glu[glu < 0] = 0
    bp[bp < 0] = 0
    skin[skin < 0] = 0
    ins[ins < 0] = 0
    
    bmi = np.random.normal(loc=32.0, scale=7.8, size=n)
    bmi[bmi < 0.0] = 0.0
    dpf = np.random.exponential(scale=0.37, size=n) + 0.078
    age = np.random.normal(loc=33.2, scale=11.7, size=n).astype(int)
    age[age < 21] = 21
    
    # Logistic model for Outcome probability
    z = -8.2 + 0.12 * preg + 0.038 * glu + 0.008 * bp + 0.005 * skin + 0.0005 * ins + 0.082 * bmi + 0.94 * dpf + 0.015 * age
    p = 1 / (1 + np.exp(-z))
    outcome = np.random.binomial(n=1, p=p)
    
    df = pd.DataFrame({
        "Pregnancies": preg,
        "Glucose": glu,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": ins,
        "BMI": np.round(bmi, 1),
        "DiabetesPedigreeFunction": np.round(dpf, 3),
        "Age": age,
        "Outcome": outcome
    })
    return df

# 2. Heart Disease Dataset Generator
def generate_heart_data():
    np.random.seed(42)
    n = 303
    age = np.random.normal(loc=54.3, scale=9.0, size=n).astype(int)
    sex = np.random.binomial(n=1, p=0.68, size=n)
    cp = np.random.choice([0, 1, 2, 3], size=n, p=[0.47, 0.16, 0.29, 0.08])
    bp = np.random.normal(loc=131.6, scale=17.5, size=n).astype(int)
    chol = np.random.normal(loc=246.2, scale=51.8, size=n).astype(int)
    fbs = np.random.binomial(n=1, p=0.148, size=n)
    ecg = np.random.choice([0, 1, 2], size=n, p=[0.48, 0.50, 0.02])
    hr = np.random.normal(loc=149.6, scale=22.9, size=n).astype(int)
    hr = np.minimum(hr, 220 - age)  # Biological constraint
    ang = np.random.binomial(n=1, p=0.326, size=n)
    old = np.random.exponential(scale=1.03, size=n)
    slope = np.random.choice([1, 2, 3], size=n, p=[0.24, 0.46, 0.30])
    ca = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.58, 0.21, 0.12, 0.06, 0.03])
    thal = np.random.choice([1, 2, 3], size=n, p=[0.06, 0.54, 0.40])
    
    # Heart disease mathematical risk probability
    z = -4.5 + 0.02 * age + 0.45 * sex + 0.55 * cp + 0.012 * bp + 0.004 * chol + 0.3 * fbs - 0.018 * hr + 1.15 * ang + 0.58 * old - 0.35 * slope + 0.75 * ca + 0.65 * thal
    p = 1 / (1 + np.exp(-z))
    outcome = np.random.binomial(n=1, p=p)
    
    df = pd.DataFrame({
        "Age": age, "Sex": sex, "ChestPainType": cp, "RestingBP": bp, "Cholesterol": chol,
        "FastingBS": fbs, "RestingECG": ecg, "MaxHR": hr, "ExerciseAngina": ang,
        "Oldpeak": np.round(old, 1), "ST_Slope": slope, "CA": ca, "Thal": thal, "Outcome": outcome
    })
    return df

# 3. Liver Disease Dataset Generator
def generate_liver_data():
    np.random.seed(42)
    n = 583
    age = np.random.normal(loc=44.7, scale=16.1, size=n).astype(int)
    age[age < 4] = 4
    gender = np.random.binomial(n=1, p=0.75, size=n) # 1=Male, 0=Female
    tb = np.random.exponential(scale=3.3, size=n) + 0.1
    db = tb * np.random.uniform(0.35, 0.55, size=n)
    ap = np.random.normal(loc=290, scale=180, size=n).astype(int)
    ap[ap < 60] = 60
    sgpt = np.random.normal(loc=80, scale=120, size=n).astype(int)
    sgpt[sgpt < 10] = 10
    sgot = sgpt * np.random.uniform(1.2, 1.8, size=n)
    sgot = sgot.astype(int)
    tp = np.random.normal(loc=6.4, scale=0.8, size=n)
    alb = np.random.normal(loc=3.1, scale=0.6, size=n)
    alb = np.minimum(alb, tp - 0.4) # logical check
    ag = alb / (tp - alb)
    
    # Risk calculation
    z = -3.8 + 0.015 * age + 0.4 * gender + 0.6 * tb + 1.0 * db + 0.0015 * ap + 0.008 * sgpt + 0.004 * sgot - 0.25 * tp - 0.65 * alb - 0.8 * ag
    p = 1 / (1 + np.exp(-z))
    outcome = np.random.binomial(n=1, p=p)
    
    df = pd.DataFrame({
        "Age": age, "Gender": gender, "Total_Bilirubin": np.round(tb, 1),
        "Direct_Bilirubin": np.round(db, 1), "Alkaline_Phosphotase": ap,
        "Alamine_Aminotransferase": sgpt, "Aspartate_Aminotransferase": sgot,
        "Total_Protiens": np.round(tp, 1), "Albumin": np.round(alb, 1),
        "Albumin_and_Globulin_Ratio": np.round(ag, 2), "Outcome": outcome
    })
    return df

# 4. Breast Cancer Dataset Generator
def generate_breast_cancer_data():
    np.random.seed(42)
    n = 569
    rad = np.random.normal(loc=14.1, scale=3.5, size=n)
    rad[rad < 6.0] = 6.0
    text = np.random.normal(loc=19.2, scale=4.3, size=n)
    text[text < 9.0] = 9.0
    perm = rad * np.random.uniform(6.1, 6.5, size=n)
    area = np.pi * (rad ** 2) * np.random.uniform(0.9, 1.1, size=n)
    smooth = np.random.normal(loc=0.096, scale=0.014, size=n)
    comp = np.random.normal(loc=0.104, scale=0.052, size=n)
    comp[comp < 0.01] = 0.01
    conc = np.random.normal(loc=0.088, scale=0.079, size=n)
    conc[conc < 0.0] = 0.0
    points = conc * np.random.uniform(0.42, 0.58, size=n)
    symm = np.random.normal(loc=0.181, scale=0.027, size=n)
    fract = np.random.normal(loc=0.062, scale=0.007, size=n)
    
    # Target Outcome: 1 = Malignant, 0 = Benign
    z = -13.5 + 0.42 * rad + 0.12 * text + 0.08 * perm + 0.0048 * area + 11.2 * smooth + 5.5 * comp + 13.0 * conc + 28.0 * points + 4.2 * symm
    p = 1 / (1 + np.exp(-z))
    outcome = np.random.binomial(n=1, p=p)
    
    df = pd.DataFrame({
        "Radius_Mean": np.round(rad, 2), "Texture_Mean": np.round(text, 2),
        "Perimeter_Mean": np.round(perm, 2), "Area_Mean": np.round(area, 1),
        "Smoothness_Mean": np.round(smooth, 4), "Compactness_Mean": np.round(comp, 4),
        "Concavity_Mean": np.round(conc, 4), "Concave_Points_Mean": np.round(points, 4),
        "Symmetry_Mean": np.round(symm, 4), "Fractal_Dimension_Mean": np.round(fract, 4),
        "Outcome": outcome
    })
    return df

# 5. Parkinson's Disease Dataset Generator
def generate_parkinsons_data():
    np.random.seed(42)
    n = 195
    fo = np.random.normal(loc=154.2, scale=41.3, size=n)
    fhi = fo * np.random.uniform(1.15, 1.75, size=n)
    flo = fo * np.random.uniform(0.62, 0.88, size=n)
    jit = np.random.exponential(scale=0.0062, size=n) + 0.0009
    shim = np.random.exponential(scale=0.029, size=n) + 0.009
    nhr = np.random.exponential(scale=0.024, size=n) + 0.0015
    hnr = np.random.normal(loc=21.88, scale=4.42, size=n)
    hnr[hnr < 8.0] = 8.0
    rpde = np.random.normal(loc=0.489, scale=0.103, size=n)
    dfa = np.random.normal(loc=0.718, scale=0.055, size=n)
    ppe = np.random.normal(loc=0.206, scale=0.090, size=n)
    
    # Target Outcome: 1 = Parkinson's, 0 = Healthy
    z = 4.8 - 0.028 * fo - 0.008 * fhi - 0.015 * flo + 240 * jit + 35 * shim + 18 * nhr - 0.18 * hnr + 2.8 * rpde + 1.2 * dfa + 4.8 * ppe
    p = 1 / (1 + np.exp(-z))
    outcome = np.random.binomial(n=1, p=p)
    
    df = pd.DataFrame({
        "MDVP:Fo(Hz)": np.round(fo, 3), "MDVP:Fhi(Hz)": np.round(fhi, 3),
        "MDVP:Flo(Hz)": np.round(flo, 3), "Jitter(%)": np.round(jit, 6),
        "Shimmer": np.round(shim, 6), "NHR": np.round(nhr, 6), "HNR": np.round(hnr, 3),
        "RPDE": np.round(rpde, 4), "DFA": np.round(dfa, 4), "PPE": np.round(ppe, 4),
        "Outcome": outcome
    })
    return df

# Main retraining execution
def run_retrain():
    create_directories()
    
    log = []
    def log_print(msg):
        log.append(msg)
        print(msg)
        
    metrics_summary = {}

    log_print("="*60)
    log_print(" STARTING CLINICAL DIAGNOSTIC MODEL TRAINING SUITE ")
    log_print("="*60)

    # List of configurations
    training_configs = [
        {
            "name": "diabetes",
            "file": "diabetes.csv",
            "generator": generate_diabetes_data,
            "algorithm": "Logistic Regression",
            "model_class": LogisticRegression(max_iter=1000, random_state=42),
            "scale": True
        },
        {
            "name": "heart",
            "file": "heart.csv",
            "generator": generate_heart_data,
            "algorithm": "Logistic Regression",
            "model_class": LogisticRegression(max_iter=1000, random_state=42),
            "scale": True
        },
        {
            "name": "liver",
            "file": "liver.csv",
            "generator": generate_liver_data,
            "algorithm": "Random Forest",
            "model_class": RandomForestClassifier(n_estimators=100, random_state=42),
            "scale": False
        },
        {
            "name": "breast_cancer",
            "file": "breast_cancer.csv",
            "generator": generate_breast_cancer_data,
            "algorithm": "Logistic Regression",
            "model_class": LogisticRegression(max_iter=1000, random_state=42),
            "scale": True
        },
        {
            "name": "parkinsons",
            "file": "parkinsons.csv",
            "generator": generate_parkinsons_data,
            "algorithm": "Logistic Regression",
            "model_class": LogisticRegression(max_iter=1000, random_state=42),
            "scale": True
        }
    ]

    for config in training_configs:
        name = config["name"]
        csv_path = os.path.join("datasets", config["file"])
        
        # 1. Generate/Load Dataset
        if not os.path.exists(csv_path):
            log_print(f"Generating synthetic dataset for {name.upper()}...")
            df = config["generator"]()
            df.to_csv(csv_path, index=False)
        else:
            log_print(f"Loading existing dataset for {name.upper()}...")
            df = pd.read_csv(csv_path)
            
        # 2. Split features and target
        X = df.drop(columns=["Outcome"])
        y = df["Outcome"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 3. Handle scaling
        scaler = None
        if config["scale"]:
            log_print(f"Fitting StandardScaler for {name.upper()} features...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            
        # 4. Train Model
        log_print(f"Training {config['algorithm']} classifier for {name.upper()}...")
        model = config["model_class"]
        model.fit(X_train_scaled, y_train)
        
        # 5. Evaluate Model
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        rep = classification_report(y_test, y_pred)
        
        log_print(f"Accuracy on {name.upper()} Testing Split: {acc*100:.2f}%")
        log_print(f"Evaluation Report for {name.upper()}:\n{rep}")
        
        # 6. Save resources to models/ folder
        m_filename = os.path.join("models", f"{name}_model.pkl")
        with open(m_filename, "wb") as f:
            pickle.dump(model, f)
            
        if scaler:
            s_filename = os.path.join("models", f"{name}_scaler.pkl")
            with open(s_filename, "wb") as f:
                pickle.dump(scaler, f)
                
        # Cache metrics metadata
        metrics_summary[name] = {
            "accuracy": float(acc),
            "test_samples": int(len(y_test)),
            "report": rep
        }
        log_print("-"*60)
        
    # Save global metrics JSON
    metrics_path = os.path.join("models", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    log_print("Saved global diagnostic accuracies to models/metrics.json")
    log_print("ML training suite executed and serialized models successfully!")
    log_print("="*60)
    
    return "\n".join(log)

if __name__ == "__main__":
    run_retrain()
