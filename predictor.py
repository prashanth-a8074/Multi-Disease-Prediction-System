import os
import pickle
import pandas as pd
import numpy as np

# Cache loaders to avoid reading disk on every single prediction
_MODEL_CACHE = {}

def load_prediction_artifacts(disease):
    """Loads the model, scaler, feature names, and encoders for a disease."""
    if disease in _MODEL_CACHE:
        return _MODEL_CACHE[disease]
    
    model_path = f'models/model_{disease}.pkl'
    scaler_path = f'models/scaler_{disease}.pkl'
    features_path = f'models/features_{disease}.pkl'
    encoders_path = f'models/encoders_{disease}.pkl'
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
        raise FileNotFoundError(f"Model artifacts not found for {disease}. Run train_models.py first.")
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
        
    encoders = None
    if os.path.exists(encoders_path):
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
            
    artifacts = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'encoders': encoders
    }
    
    _MODEL_CACHE[disease] = artifacts
    return artifacts

def predict_disease(disease, raw_inputs):
    """
    Predicts status and risk percentage for a given disease.
    
    Parameters:
    - disease (str): Name of the disease ('diabetes', 'heart', 'breast_cancer', 'liver', 'parkinsons')
    - raw_inputs (dict): Raw key-value inputs from user interface
    
    Returns:
    - result (int): 0 (Negative/Healthy) or 1 (Positive/At Risk)
    - risk_percentage (float): Risk probability percentage [0.0, 100.0]
    """
    artifacts = load_prediction_artifacts(disease)
    model = artifacts['model']
    scaler = artifacts['scaler']
    features = artifacts['features']
    encoders = artifacts['encoders']
    
    # Map raw_inputs to the expected model features
    mapped_inputs = {}
    
    # Pre-clean keys for comparison
    clean_inputs = {}
    for k, v in raw_inputs.items():
        clean_k = k.lower().replace(" ", "_").replace("-", "_").replace(":", "").replace("(", "").replace(")", "").replace("%", "")
        clean_inputs[clean_k] = v

    # Common abbreviation mapping
    abbrev_map = {
        "cp": ["chest_pain", "cp"],
        "trestbps": ["resting_bp", "restingbp", "bp", "blood_pressure"],
        "fbs": ["fasting_bs", "fastingbs"],
        "restecg": ["resting_ecg", "restingecg"],
        "thalach": ["max_heart_rate", "max_hr", "maxhr", "heart_rate"],
        "exang": ["exercise_angina", "exerciseangina", "exang"],
        "slope": ["st_slope", "stslope", "slope"],
        "ca": ["ca_vessels", "ca"],
        "thal": ["thalassemia", "thal"],
        "alamine_aminotransferase": ["alt", "sgpt"],
        "aspartate_aminotransferase": ["ast", "sgot"],
        "total_protiens": ["total_proteins", "total_protiens", "tp"],
        "albumin_and_globulin_ratio": ["ag_ratio", "ag_ratio_ratio", "ag"],
        "average_fundamental_freq": ["fo"],
        "max_fundamental_freq": ["fhi"],
        "min_fundamental_freq": ["flo"],
        "jitter_percent": ["jitter"],
        "mdvpfohz": ["fo"],
        "mdvpfhihz": ["fhi"],
        "mdvpflohz": ["flo"],
        "jitter": ["jitter"],
        "shimmer": ["shimmer"],
        "nhr": ["nhr"],
        "hnr": ["hnr"],
        "rpde": ["rpde"],
        "dfa": ["dfa"],
        "ppe": ["ppe"]
    }

    for feat in features:
        feat_clean = feat.lower().replace(" ", "_").replace("-", "_").replace(":", "").replace("(", "").replace(")", "").replace("%", "")
        
        # 1. Exact case-insensitive / clean key match
        if feat_clean in clean_inputs:
            mapped_inputs[feat] = clean_inputs[feat_clean]
            continue
            
        # 2. Check abbreviation map
        matched_val = None
        for abbrev_key, aliases in abbrev_map.items():
            if feat_clean == abbrev_key or abbrev_key in feat_clean:
                for alias in aliases:
                    if alias in clean_inputs:
                        matched_val = clean_inputs[alias]
                        break
                if matched_val is not None:
                    break
        
        if matched_val is not None:
            mapped_inputs[feat] = matched_val
            continue

        # 3. Breast Cancer "worst" fallback to "mean"
        if "worst" in feat_clean:
            mean_version = feat_clean.replace("worst", "mean")
            if mean_version in clean_inputs:
                mapped_inputs[feat] = clean_inputs[mean_version]
                continue
            # Try also checking without "_mean"
            mean_version_short = mean_version.replace("_mean", "")
            if mean_version_short in clean_inputs:
                mapped_inputs[feat] = clean_inputs[mean_version_short]
                continue

        # 4. Try partial substring match
        found = False
        for k, v in clean_inputs.items():
            if k in feat_clean or feat_clean in k:
                mapped_inputs[feat] = v
                found = True
                break
        if found:
            continue

    # 5. Default fallback
        if "gender" in feat_clean or "sex" in feat_clean:
            mapped_inputs[feat] = "Male" if (encoders and feat in encoders) else 1.0
        else:
            mapped_inputs[feat] = 0.0

    # Map raw categorical inputs to numeric values
    for feat in mapped_inputs:
        val = mapped_inputs[feat]
        if isinstance(val, str):
            val_clean = val.lower().strip()
            if feat.lower() in ["sex", "gender"]:
                if encoders and feat in encoders:
                    pass
                else:
                    mapped_inputs[feat] = 0.0 if "female" in val_clean else 1.0
            elif feat.lower() in ["cp", "chestpaintype"]:
                cp_list = ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"]
                if val_clean in cp_list:
                    mapped_inputs[feat] = float(cp_list.index(val_clean))
                else:
                    mapped_inputs[feat] = float(next((i for i, x in enumerate(cp_list) if x in val_clean), 0))
            elif feat.lower() in ["fbs", "fastingbs"]:
                mapped_inputs[feat] = 1.0 if ("yes" in val_clean or "true" in val_clean or "> 120" in val_clean) else 0.0
            elif feat.lower() in ["restecg", "restingecg"]:
                ecg_list = ["normal", "st-t wave abnormality", "lvh", "left ventricular hypertrophy"]
                idx = ecg_list.index(val_clean) if val_clean in ecg_list else next((i for i, x in enumerate(ecg_list) if x in val_clean), 0)
                mapped_inputs[feat] = float(2 if idx == 3 else idx)
            elif feat.lower() in ["exang", "exerciseangina"]:
                mapped_inputs[feat] = 1.0 if ("yes" in val_clean or "true" in val_clean) else 0.0
            elif feat.lower() in ["slope", "st_slope"]:
                slope_list = ["upsloping", "flat", "downsloping"]
                idx = slope_list.index(val_clean) if val_clean in slope_list else next((i for i, x in enumerate(slope_list) if x in val_clean), 1)
                mapped_inputs[feat] = float(idx + 1)
            elif feat.lower() in ["thal", "thalassemia"]:
                thal_list = ["normal", "fixed defect", "reversible defect"]
                idx = thal_list.index(val_clean) if val_clean in thal_list else next((i for i, x in enumerate(thal_list) if x in val_clean), 1)
                mapped_inputs[feat] = float(idx + 1)

    # 1. Apply categorical label encoding if applicable (in dict space to avoid pandas dtype conflicts)
    if encoders:
        for col, encoder in encoders.items():
            if col in mapped_inputs:
                try:
                    val = str(mapped_inputs[col])
                    if val not in encoder.classes_:
                        mapped_inputs[col] = int(encoder.transform([encoder.classes_[0]])[0])
                    else:
                        mapped_inputs[col] = int(encoder.transform([val])[0])
                except Exception as e:
                    print(f"Error encoding categorical column {col}: {e}")
                    mapped_inputs[col] = 0

    # 2. Prepare input DataFrame
    input_df = pd.DataFrame([mapped_inputs])
    
    # 3. Order columns exactly as during model training
    input_df = input_df[features]
    
    # 4. Scale inputs if scaler is present
    if scaler:
        scaled_inputs = scaler.transform(input_df)
    else:
        scaled_inputs = input_df.values
        
    # 5. Classify & Compute Risk
    prediction = int(model.predict(scaled_inputs)[0])
    
    # Calculate probability/risk percentage
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(scaled_inputs)[0][1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(scaled_inputs)[0]
        # Map decision score to standard sigmoid to get mock probability
        prob = 1 / (1 + np.exp(-decision))
    else:
        prob = float(prediction)
        
    risk_percentage = round(float(prob) * 100, 2)
    
    return prediction, risk_percentage
