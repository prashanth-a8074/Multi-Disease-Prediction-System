import re

# Clinical Guidelines Knowledge Base
HEALTH_KNOWLEDGE = {
    'diabetes': {
        'symptoms': "Common symptoms of diabetes include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, irritability, and blurry vision.",
        'prevention': "Reduce refined sugar intake, exercise at least 150 minutes per week, monitor carbohydrates, consume fiber-rich food, and maintain a BMI under 25.",
        'metrics': "Key metrics include Fasting Blood Sugar (Normal < 100 mg/dL, Diabetic > 125 mg/dL) and HbA1c (Normal < 5.7%, Diabetic >= 6.5%)."
    },
    'heart': {
        'symptoms': "Heart disease warning signs include chest pain (angina), shortness of breath, pain in neck/jaw/throat/back, numbness or coldness in limbs, and palpitations.",
        'prevention': "Adopt a Mediterranean diet, limit saturated fats, exercise regularly, manage blood pressure (< 120/80 mmHg), quit smoking, and manage stress.",
        'metrics': "Important factors: Systolic/Diastolic blood pressure, Serum Cholesterol (Desirable < 200 mg/dL), and Max Heart Rate (roughly 220 minus age)."
    },
    'breast_cancer': {
        'symptoms': "Symptoms include a new lump in the breast or underarm, thickening or swelling of part of the breast, irritation or dimpling of breast skin, redness, or nipple discharge.",
        'prevention': "Maintain a healthy weight, stay physically active, limit alcohol, breastfeed if possible, and discuss risks with your doctor regarding hormone replacement therapy.",
        'metrics': "Clinical screening includes regular self-exams, clinical breast exams, and mammograms starting at age 40 (or earlier if high family risk)."
    },
    'liver': {
        'symptoms': "Liver disease indicators include yellowish skin and eyes (jaundice), abdominal pain and swelling, chronic fatigue, dark urine, pale stool, and itchy skin.",
        'prevention': "Limit alcohol consumption, get vaccinated against Hepatitis A & B, avoid sharing needles, maintain a healthy weight, and use medications only as directed.",
        'metrics': "Key indicators: Total Bilirubin (Normal 0.1 - 1.2 mg/dL), SGPT/ALT (Normal 7 - 56 U/L), SGOT/AST (Normal 10 - 40 U/L), and Albumin-Globulin ratio (Normal > 1.0)."
    },
    'parkinsons': {
        'symptoms': "Early signs of Parkinson's include tremors (especially at rest), slow movement (bradykinesia), rigid muscles, impaired posture/balance, and loss of automatic movements.",
        'prevention': "Engage in regular aerobic exercise, eat a nutrient-rich diet (especially containing antioxidants, Omega-3 fatty acids, and green tea), and avoid exposure to toxins/pesticides.",
        'metrics': "Diagnosed clinically through neurological examination, acoustic analysis of vocal tremors (including jitter, shimmer, pitch fluctuations like PPE, spread1, spread2)."
    }
}

def generate_ai_response(user_message, prediction_context=None):
    """
    Generates a helpful, structured response to medical queries.
    
    Parameters:
    - user_message (str): Text entered by the user
    - prediction_context (dict, optional): Details of current session prediction
      e.g. {'disease': 'heart', 'risk_percentage': 72.5, 'patient_name': 'John'}
    
    Returns:
    - str: Markdown formatted response
    """
    msg = user_message.lower().strip()
    
    # 1. Check if user is asking about their own recent prediction results
    if prediction_context and any(kw in msg for kw in ['my result', 'my prediction', 'my risk', 'diagnosis', 'am i safe', 'am i sick']):
        dis = prediction_context['disease']
        risk = prediction_context['risk_percentage']
        name = prediction_context['patient_name']
        res_label = "Elevated Risk" if risk >= 50 else "Low Risk"
        
        response = f"Hello {name}, based on your input parameters, our diagnostic engine analyzed indicators for **{dis.replace('_', ' ').title()}**.\n\n"
        response += f"- **Result Category:** {res_label}\n"
        response += f"- **Risk Probability:** `{risk}%` \n\n"
        
        if risk >= 50:
            response += f"Because your risk is elevated ({risk}%), I strongly advise you to contact a specialist in **{HEALTH_KNOWLEDGE.get(dis, {}).get('specialization', 'General Medicine')}**. "
            response += "In the meantime, make sure to monitor your daily vital signs, avoid stress, and review our Hospital Recommendation section for nearest centers."
        else:
            response += "Your diagnostic probability shows a low risk profile. That is great! Please continue maintaining a balanced lifestyle. "
            response += f"Here are some preventive guidelines for {dis.replace('_', ' ').title()}:\n\n"
            response += f"👉 {HEALTH_KNOWLEDGE.get(dis, {}).get('prevention', '')}"
        return response

    # 2. General Greetings
    if any(kw in msg for kw in ['hello', 'hi', 'hey', 'greetings', 'hola']):
        return ("👋 Hello! I am your AI Health Assistant.\n\n"
                "I can help explain clinical metrics, outline symptoms or prevention strategies for Diabetes, Heart Disease, Breast Cancer, Liver Disease, and Parkinson's, or analyze your dashboard prediction logs. "
                "What health topic can I assist you with today?")

    # 3. Handle specific disease questions
    for disease_key, data in HEALTH_KNOWLEDGE.items():
        name_matched = disease_key.replace('_', ' ')
        if name_matched in msg or (disease_key == 'breast_cancer' and 'cancer' in msg) or (disease_key == 'heart' and 'cardio' in msg):
            if 'symptom' in msg or 'sign' in msg or 'feel' in msg:
                return f"### Symptoms of {name_matched.title()}\n\n{data['symptoms']}"
            elif 'prevent' in msg or 'avoid' in msg or 'diet' in msg or 'lifestyle' in msg:
                return f"### Prevention & Lifestyle Guidelines for {name_matched.title()}\n\n{data['prevention']}"
            elif 'metric' in msg or 'test' in msg or 'biomarker' in msg or 'range' in msg:
                return f"### Diagnostics & Clinical Metrics for {name_matched.title()}\n\n{data['metrics']}"
            else:
                return (f"### {name_matched.title()} Information\n\n"
                        f"**Symptoms:** {data['symptoms']}\n\n"
                        f"**Prevention:** {data['prevention']}\n\n"
                        f"**Clinical Metrics:** {data['metrics']}")

    # 4. Explanations for specific complex metrics
    if 'bmi' in msg:
        return ("### Body Mass Index (BMI)\n\n"
                "BMI is a measure of body fat based on height and weight. "
                "- **Underweight:** BMI < 18.5\n"
                "- **Normal Weight:** BMI 18.5 - 24.9\n"
                "- **Overweight:** BMI 25.0 - 29.9\n"
                "- **Obese:** BMI >= 30.0\n"
                "High BMI is a major risk factor for Diabetes, Heart Disease, and Liver disease.")
                
    if 'bilirubin' in msg or 'liver function' in msg:
        return ("### Bilirubin (Total & Direct)\n\n"
                "Bilirubin is a yellowish substance in your blood, formed after red blood cells break down. It passes through your liver.\n"
                "- **Total Bilirubin:** Sum of direct and indirect bilirubin. Normal is 0.1 to 1.2 mg/dL.\n"
                "- **Direct (Conjugated) Bilirubin:** Bilirubin processed by the liver. Normal is < 0.3 mg/dL.\n"
                "Elevated bilirubin levels indicate potential liver damage, hepatitis, bile duct obstruction, or gallbladder issues.")

    if 'sgpt' in msg or 'sgot' in msg or 'alt' in msg or 'ast' in msg:
        return ("### SGPT (ALT) and SGOT (AST) Enzymes\n\n"
                "- **SGPT / ALT (Alanine Aminotransferase):** Enzyme found mostly in liver cells. When liver is injured, ALT is released into blood. Normal is 7 - 56 U/L.\n"
                "- **SGOT / AST (Aspartate Aminotransferase):** Enzyme found in liver, heart, muscles. Normal is 10 - 40 U/L.\n"
                "High levels of these enzymes in blood signify liver inflammation or injury.")

    if 'oldpeak' in msg or 'st depression' in msg:
        return ("### ST Depression (Oldpeak)\n\n"
                "In electrocardiography (ECG), **Oldpeak** refers to the ST depression induced by exercise relative to rest. "
                "It is a key indicator of coronary artery stenosis (blockage). Higher values (e.g., > 1.5) indicate potential myocardial ischemia (heart disease).")

    if 'ca' in msg or 'major vessels' in msg:
        return ("### Number of Major Vessels (ca)\n\n"
                "In heart diagnostics, **ca** refers to the number of major coronary vessels (0-3) colored by fluoroscopy. "
                "It indicates how many primary arteries are unblocked. Higher values (1, 2, or 3) indicate significant vessel narrowing or coronary blockages.")

    if 'thal' in msg or 'thalassemia' in msg:
        return ("### Thalassemia Type (thal)\n\n"
                "In heart datasets, **thal** represents a blood flow scan:\n"
                "- **3 = Normal** (adequate blood flow)\n"
                "- **6 = Fixed Defect** (no blood flow in some parts, representing prior heart tissue death)\n"
                "- **7 = Reversible Defect** (restricted blood flow during stress/exercise, indicating active ischemia)")

    if 'jitter' in msg or 'shimmer' in msg or 'ppe' in msg or 'fo' in msg:
        return ("### Vocal/Acoustic Metrics in Parkinson's Detection\n\n"
                "Voice impairment is an early sign of Parkinson's disease:\n"
                "- **MDVP:Fo (Hz):** Average vocal fundamental frequency (frequency drops under vocal muscle rigidity).\n"
                "- **Jitter (%):** Frequency variation in vocal cord vibration. High jitter means uneven pitch.\n"
                "- **Shimmer:** Amplitude variation in vocal cord vibration. High shimmer means uneven volume.\n"
                "- **PPE (Pitch Period Entropy):** Measure of progressive fundamental frequency variations. Higher values indicate voice tremor severity.")

    if 'help' in msg or 'what can you do' in msg:
        return ("### How to use this Assistant\n\n"
                "You can ask me questions like:\n"
                "1. *'What are the early signs of Parkinson's?'*\n"
                "2. *'How can I prevent Type-2 Diabetes?'*\n"
                "3. *'What does Bilirubin mean in my liver report?'*\n"
                "4. *'Explain oldpeak in my heart disease results.'*\n"
                "5. *'Can you explain my last prediction results?'* (If you have run a prediction in the current session)")

    # 5. Default Fallback
    return ("Thank you for reaching out. I'm trained specifically on clinical metrics and prevention guidelines for "
            "Diabetes, Heart Disease, Breast Cancer, Liver Disease, and Parkinson's.\n\n"
            "Could you please specify which of these conditions or related metrics (such as BMI, Bilirubin, oldpeak, Jitter) you would like to know more about? "
            "Alternatively, type **help** to see example queries.")
