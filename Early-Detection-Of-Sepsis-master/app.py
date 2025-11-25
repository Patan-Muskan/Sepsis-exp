#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/static')
model = pickle.load(open('model.pkl', 'rb'))

# Feature names (27 features)
FEATURE_NAMES = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp',
    'BaseExcess', 'HCO3', 'FiO2', 'PaCO2', 'SaO2', 'Creatinine',
    'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
    'Bilirubin_total', 'Hgb', 'WBC', 'Fibrinogen', 'Platelets',
    'Age', 'Gender', 'HospAdmTime', 'ICULOS'
]

# Clinical reference ranges for warning indicators
CLINICAL_RANGES = {
    'HR': (60, 100, 'beats/min'),
    'O2Sat': (95, 100, '%'),
    'Temp': (36.5, 37.5, '°C'),
    'SBP': (90, 120, 'mm Hg'),
    'MAP': (70, 100, 'mm Hg'),
    'DBP': (60, 80, 'mm Hg'),
    'Resp': (12, 20, 'breaths/min'),
    'Lactate': (0.5, 2.0, 'mmol/L'),
    'Glucose': (70, 100, 'mg/dL'),
    'Creatinine': (0.7, 1.3, 'mg/dL'),
    'WBC': (4.5, 11, 'K/µL'),
    'Hgb': (13.5, 17.5, 'g/dL'),
}

@app.route('/')
def home():
    return render_template('index.html')

def get_abnormal_features(features_dict):
    """
    Identify which features are outside normal ranges
    """
    abnormal = []
    for feature_name, value in features_dict.items():
        if feature_name in CLINICAL_RANGES and value != '':
            try:
                val = float(value)
                min_val, max_val, unit = CLINICAL_RANGES[feature_name]
                if val < min_val or val > max_val:
                    abnormal.append({
                        'feature': feature_name,
                        'value': val,
                        'normal_range': f"{min_val}-{max_val}",
                        'unit': unit,
                        'direction': 'HIGH' if val > max_val else 'LOW'
                    })
            except:
                pass
    
    # Sort by severity (furthest from normal range)
    abnormal.sort(key=lambda x: abs(x['value'] - (CLINICAL_RANGES[x['feature']][1] + CLINICAL_RANGES[x['feature']][0]) / 2), reverse=True)
    return abnormal

def generate_explanation(features_dict, prediction, confidence):
    """
    Generate a simple explanation based on abnormal values
    """
    abnormal_features = get_abnormal_features(features_dict)
    
    html = '<div style="margin-top: 20px;">'
    
    if abnormal_features:
        html += '''
        <div style="background: rgba(255, 215, 0, 0.08); border: 1px solid rgba(255, 215, 0, 0.2); 
                    border-radius: 10px; padding: 15px; margin-bottom: 15px;">
            <h5 style="color: #ffd700;">⚠️ Abnormal Clinical Values Detected</h5>
            <p style="color: #b0b0b0; margin-bottom: 10px;">The following values are outside normal ranges and may indicate sepsis risk:</p>
            <ul style="color: #b0b0b0; margin-left: 20px;">
        '''
        
        for item in abnormal_features[:10]:  # Show top 10 abnormal values
            color = '#ff6b6b' if item['direction'] == 'HIGH' else '#facc15'
            html += f'''
            <li style="color: {color}; margin-bottom: 8px;">
                <strong>{item['feature']}</strong>: {item['value']:.2f} {item['unit']} 
                <span style="color: #b0b0b0;">({item['direction']} - Normal: {item['normal_range']} {item['unit']})</span>
            </li>
            '''
        
        html += '</ul></div>'
    else:
        html += '''
        <div style="background: rgba(74, 222, 128, 0.08); border: 1px solid rgba(74, 222, 128, 0.2); 
                    border-radius: 10px; padding: 15px; margin-bottom: 15px;">
            <h5 style="color: #4ade80;">✓ All Clinical Values Within Normal Ranges</h5>
            <p style="color: #b0b0b0;">All monitored parameters are within normal clinical ranges, which is a positive indicator.</p>
        </div>
        '''
    
    # Prediction confidence explanation
    if prediction == 1:
        html += f'''
        <div style="background: rgba(255, 107, 107, 0.08); border: 1px solid rgba(255, 107, 107, 0.2); 
                    border-radius: 10px; padding: 15px;">
            <h5 style="color: #ff6b6b;">🚨 High Risk Assessment</h5>
            <p style="color: #b0b0b0;">The model predicts a <strong>{confidence:.1f}%</strong> probability of sepsis risk based on the clinical data provided.</p>
            <p style="color: #b0b0b0;"><strong>Recommendation:</strong> Consider immediate clinical evaluation and monitoring.</p>
        </div>
        '''
    else:
        html += f'''
        <div style="background: rgba(74, 222, 128, 0.08); border: 1px solid rgba(74, 222, 128, 0.2); 
                    border-radius: 10px; padding: 15px;">
            <h5 style="color: #4ade80;">✓ Low Risk Assessment</h5>
            <p style="color: #b0b0b0;">The model predicts a <strong>{100-confidence:.1f}%</strong> probability of low sepsis risk based on the clinical data provided.</p>
            <p style="color: #b0b0b0;"><strong>Recommendation:</strong> Continue routine monitoring and clinical assessment.</p>
        </div>
        '''
    
    html += '</div>'
    return html

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI with clinical explanation
    '''
    try:
        # Get form data
        form_data = request.form.to_dict()
        features = []
        
        # Convert to float, handle empty values with 0
        for feature_name in FEATURE_NAMES:
            try:
                val = float(form_data.get(feature_name, 0))
            except:
                val = 0
            features.append(val)
        
        final_features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(final_features)
        probability = model.predict_proba(final_features)[0]
        
        # Determine prediction text
        prediction_text = "High Risk of Sepsis" if prediction[0] == 1 else "Low Risk of Sepsis"
        confidence = max(probability) * 100
        
        # Generate explanation
        explanation_html = generate_explanation(form_data, prediction[0], confidence)
        
        return render_template(
            'index.html',
            prediction_text=prediction_text,
            confidence=f"{confidence:.2f}%",
            explanation=explanation_html,
            risk_level='High Risk' if prediction[0] == 1 else 'Low Risk'
        )
    
    except Exception as e:
        error_msg = f"Error in prediction: {str(e)}"
        return render_template('index.html', prediction_text=error_msg)


if __name__ == '__main__':
    app.run(debug=True)

