import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import google.generativeai as gemini

gemini.configure(api_key="AIzaSyBg18tLTIeKEWL9mH2k-XH20apKiKoJWSk")

# Load the dataset to get the feature names and scaler
cardio_data = pd.read_csv("HEART_MODEL/CAD/Coronary_Artery_Disease.csv")
x = cardio_data.drop(columns="Cath", axis=1)

# Load the scaler
scaler = StandardScaler()
scaler.fit(x)  # Fit scaler on the original data

# Load the trained model from the pickle file
with open('HEART_MODEL/CAD/cad_prediction.pkl', 'rb') as model_file:
    ensemble_model = pickle.load(model_file)

print("Model loaded successfully.")

def predict_coronary_artery_disease(input_data):
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    input_data_df = pd.DataFrame(input_data_as_numpy_array, columns=x.columns)
    input_data_scaled = scaler.transform(input_data_df)
    
    prediction = ensemble_model.predict(input_data_scaled)[0]
    probability = ensemble_model.predict_proba(input_data_scaled)[0][1]
    risk_percentage = probability * 100

    if prediction == 0:
        risk = "The person is not having coronary artery disease"
        disease_type = "No Coronary Artery Disease"
    else:
        risk = "The person is having coronary artery disease"
        disease_type = "Coronary Artery Disease"

    return risk, risk_percentage, disease_type

# Collect user input
age = float(input("Enter Age: "))
weight = float(input("Enter Weight (in kg): "))
length = float(input("Enter Length/Height (in cm): "))
sex = int(input("Enter Sex (0 for female, 1 for male): "))
bmi = float(input("Enter BMI: "))
dm = int(input("Diabetes Mellitus (0 for No, 1 for Yes): "))
htn = int(input("Hypertension (0 for No, 1 for Yes): "))
current_smoker = int(input("Current Smoker (0 for No, 1 for Yes): "))
ex_smoker = int(input("Ex-Smoker (0 for No, 1 for Yes): "))
fh = int(input("Family History of CAD (0 for No, 1 for Yes): "))
obesity = int(input("Obesity (0 for No, 1 for Yes): "))
crf = int(input("Chronic Renal Failure (0 for No, 1 for Yes): "))
cva = int(input("Cerebrovascular Accident (0 for No, 1 for Yes): "))
airway_disease = int(input("Airway Disease (0 for No, 1 for Yes): "))
thyroid_disease = int(input("Thyroid Disease (0 for No, 1 for Yes): "))
chf = int(input("Congestive Heart Failure (0 for No, 1 for Yes): "))
dlp = int(input("Dyslipidemia (0 for No, 1 for Yes): "))
bp = float(input("Blood Pressure (systolic): "))
pr = int(input("Pulse Rate: "))
edema = int(input("Edema (0 for No, 1 for Yes): "))
weak_peripheral_pulse = int(input("Weak Peripheral Pulse (0 for No, 1 for Yes): "))
lung_rales = int(input("Lung Rales (0 for No, 1 for Yes): "))
systolic_murmur = int(input("Systolic Murmur (0 for No, 1 for Yes): "))
diastolic_murmur = int(input("Diastolic Murmur (0 for No, 1 for Yes): "))
typical_chest_pain = int(input("Typical Chest Pain (0 for No, 1 for Yes): "))
dyspnea = int(input("Dyspnea (0 for No, 1 for Yes): "))
functional_class = int(input("Functional Class: "))
atypical = int(input("Atypical presentation (0 for No, 1 for Yes): "))
nonanginal = int(input("Nonanginal (0 for No, 1 for Yes): "))
exertional_cp = int(input("Exertional Chest Pain (0 for No, 1 for Yes): "))
lowth_ang = int(input("Low Threshold Angina (0 for No, 1 for Yes): "))
q_wave = int(input("Q Wave (0 for No, 1 for Yes): "))
st_elevation = int(input("ST Elevation (0 for No, 1 for Yes): "))
st_depression = int(input("ST Depression (0 for No, 1 for Yes): "))
t_inversion = int(input("T Inversion (0 for No, 1 for Yes): "))
lvh = int(input("Left Ventricular Hypertrophy (0 for No, 1 for Yes): "))
poor_r_progression = int(input("Poor R Progression (0 for No, 1 for Yes): "))
fbs = float(input("Fasting Blood Sugar: "))
cr = float(input("Creatinine Level: "))
tg = float(input("Triglycerides Level: "))
ldl = float(input("LDL Level: "))
hdl = float(input("HDL Level: "))
bun = float(input("Blood Urea Nitrogen: "))
esr = float(input("Erythrocyte Sedimentation Rate: "))
hb = float(input("Hemoglobin Level: "))
k = float(input("Potassium Level: "))
na = float(input("Sodium Level: "))
wbc = float(input("White Blood Cell Count: "))
lymph = float(input("Lymphocyte Count: "))
neut = float(input("Neutrophil Count: "))
plt = float(input("Platelet Count: "))
ef_tte = float(input("Ejection Fraction by TTE: "))
regional_rwma = int(input("Regional Wall Motion Abnormality (0 for No, 1 for Yes): "))
vhd = int(input("Valvular Heart Disease (0 for No, 1 for Yes): "))

# Combine the inputs into a tuple for prediction
input_data = (
    age, weight, length, sex, bmi, dm, htn, current_smoker, ex_smoker, fh,
    obesity, crf, cva, airway_disease, thyroid_disease, chf, dlp, bp, pr, edema,
    weak_peripheral_pulse, lung_rales, systolic_murmur, diastolic_murmur, typical_chest_pain,
    dyspnea, functional_class, atypical, nonanginal, exertional_cp, lowth_ang, q_wave,
    st_elevation, st_depression, t_inversion, lvh, poor_r_progression, fbs, cr, tg, ldl,
    hdl, bun, esr, hb, k, na, wbc, lymph, neut, plt, ef_tte, regional_rwma, vhd
)

# Predict coronary artery disease and provide only necessary information
risk, risk_percentage, disease_type = predict_coronary_artery_disease(input_data)

# Print the results for debugging
print(f"Debug - Risk: {risk}")
print(f"Debug - Risk Percentage: {risk_percentage:.2f}%")
print(f"Debug - Disease Type: {disease_type}")

# Function to generate a prevention report
def generate_prevention_report(risk, risk_percentage, disease_type, age):
    prompt = f"""
    Provide a general wellness report with the following sections:

    1. **Introduction**
        -Purpose of the Report: Clearly state why this report is being generated, including its relevance to the individual’s health.
        -Overview of Health & Wellness: Briefly describe the importance of understanding and managing health risks, with a focus on proactive wellness and disease prevention.
        -Personalized Context: Include the user's specific details such as age, gender, and any relevant medical history that can be linked to the risk factor and disease.
    
    2. **Risk Description**
        -Detailed Explanation of Risk: Describe the identified risk factor in detail, including how it impacts the body and its potential consequences if left unaddressed.
        -Associated Conditions: Mention any other health conditions commonly associated with this risk factor.
        -Prevalence and Statistics: Provide some general statistics or prevalence rates to contextualize the risk (e.g., how common it is in the general population or specific age groups).
    
    3. **Stage of Risk**
        -Risk Level Analysis: Provide a more granular breakdown of the risk stages (e.g., low, medium, high), explaining what each stage means in terms of potential health outcomes.
        -Progression: Discuss how the risk may progress over time if not managed, and what signs to watch for that indicate worsening or improvement.
    
    4. **Risk Assessment**
        -Impact on Health: Explore how this specific risk factor might affect various aspects of health (e.g., cardiovascular, metabolic, etc.).
        -Modifiable vs. Non-Modifiable Risks: Distinguish between risks that can be changed (e.g., lifestyle factors) and those that cannot (e.g., genetic predisposition).
        -Comparative Risk: Compare the individual's risk to average levels in the general population or among peers.
        
    5. **Findings**
        -In-Depth Health Observations: Summarize the key findings from the assessment, explaining any critical areas of concern.
        -Diagnostic Insights: Provide insights into how the disease was identified, including the symptoms, biomarkers, or other diagnostic criteria used.
        -Data Interpretation: Offer a more detailed interpretation of the user's health data, explaining what specific values or results indicate.
    
    6. **Recommendations**
        -Personalized Action Plan: Suggest specific, actionable steps the individual can take to mitigate the risk or manage the disease (e.g., dietary changes, exercise plans, medical treatments).
        -Lifestyle Modifications: Tailor suggestions to the individual’s lifestyle, providing practical tips for integrating these changes.
        -Monitoring and Follow-up: Recommend how the user should monitor their health and when to seek follow-up care.
        
    7. **Way Forward**
        -Next Steps: Provide a clear path forward, including short-term and long-term goals for managing the identified risk or disease.
        -Preventive Measures: Highlight preventive strategies to avoid worsening the condition or preventing its recurrence.
        -Health Resources: Suggest additional resources, such as apps, websites, or support groups, that could help the individual manage their health.
        
    8. **Conclusion**
        -Summary of Key Points: Recap the most important points from the report, focusing on what the individual should remember and prioritize.
        -Encouragement: Offer positive reinforcement and encouragement for taking proactive steps toward better health.
    
    9. **Contact Information**
        -Professional Guidance: Include information on how to get in touch with healthcare providers for more personalized advice or follow-up.
        -Support Services: List any available support services, such as nutritionists, fitness coaches, or mental health professionals, that could assist in managing the risk.
    
    10. **References**
        -Scientific Sources: Provide references to the scientific literature or authoritative health guidelines that support the information and recommendations given in the report.
        -Further Reading: Suggest articles, books, or other educational materials for the individual to learn more about their condition and how to manage it.

    **Details:**
    Risk: {risk}
    Disease: {disease_type}
    Age: {age}

    Note: This information is for general wellness purposes. For specific health concerns, consult a healthcare professional.
    """
    
    return prompt

# Generate the report only if CAD is predicted
if disease_type == "Coronary Artery Disease":
    report = generate_prevention_report(risk, risk_percentage, disease_type, age)
    print("\nGenerated Wellness Report:")
    print(report)
else:
    print(f"\n{risk} with a risk percentage of 0%.")
