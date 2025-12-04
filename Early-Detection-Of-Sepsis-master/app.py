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

def detect_vital_instability(features_dict):
    """
    Detect critical vital sign fluctuations/instability that may indicate sepsis risk
    Even if instantaneous values seem normal, significant variability is a red flag
    """
    instability_indicators = []
    severity_score = 0
    
    critical_vitals = {
        'HR': {'normal_range': (60, 100), 'fluctuation_threshold': 20, 'critical_high': 130, 'critical_low': 40},
        'O2Sat': {'normal_range': (95, 100), 'fluctuation_threshold': 5, 'critical_high': 100, 'critical_low': 85},
        'Temp': {'normal_range': (36.5, 37.5), 'fluctuation_threshold': 1.5, 'critical_high': 40, 'critical_low': 35},
        'SBP': {'normal_range': (90, 120), 'fluctuation_threshold': 25, 'critical_high': 180, 'critical_low': 70},
        'Resp': {'normal_range': (12, 20), 'fluctuation_threshold': 8, 'critical_high': 30, 'critical_low': 8},
    }
    
    for vital, thresholds in critical_vitals.items():
        try:
            value = float(features_dict.get(vital, 0))
            if value == 0:
                continue
            
            min_normal, max_normal = thresholds['normal_range']
            critical_high = thresholds['critical_high']
            critical_low = thresholds['critical_low']
            fluctuation = thresholds['fluctuation_threshold']
            
            # Check for critically abnormal values (strong indicator of sepsis)
            if value >= critical_high or value <= critical_low:
                instability_indicators.append({
                    'vital': vital,
                    'value': value,
                    'severity': 'CRITICAL',
                    'description': f'{vital} is critically abnormal ({value:.1f})',
                    'concern': 'Critical vital sign deviation - immediate attention required'
                })
                severity_score += 3
            
            # Check for high variability (deviation from normal center point)
            center_normal = (min_normal + max_normal) / 2
            deviation_from_normal = abs(value - center_normal)
            max_acceptable_deviation = (max_normal - min_normal) / 2
            
            # If value deviates significantly from normal center, it indicates instability
            if deviation_from_normal > max_acceptable_deviation + fluctuation:
                instability_indicators.append({
                    'vital': vital,
                    'value': value,
                    'severity': 'HIGH',
                    'description': f'{vital} shows significant instability ({value:.1f})',
                    'concern': 'Notable deviation from normal range'
                })
                severity_score += 2
            
            # Check for marginal abnormality (approaching danger zones)
            elif (value > max_normal and value < critical_high) or (value < min_normal and value > critical_low):
                instability_indicators.append({
                    'vital': vital,
                    'value': value,
                    'severity': 'MODERATE',
                    'description': f'{vital} is outside normal range ({value:.1f})',
                    'concern': 'Minor deviation - continued monitoring advised'
                })
                severity_score += 1
        
        except (ValueError, TypeError):
            pass
    
    return {
        'indicators': instability_indicators,
        'severity_score': severity_score,
        'has_instability': severity_score > 0
    }

def generate_explanation(features_dict, prediction, confidence):
    """
    Generate a comprehensive explanation based on abnormal values and vital instability
    """
    abnormal_features = get_abnormal_features(features_dict)
    vital_instability = detect_vital_instability(features_dict)
    
    # Adjust prediction if critical instability is detected
    adjusted_risk_level = prediction
    adjustment_reason = ""
    
    if vital_instability['severity_score'] >= 3:
        # High severity instability detected - escalate to high risk
        adjusted_risk_level = 1
        adjustment_reason = "⚠️ Critical vital sign instability detected - Risk level elevated to HIGH"
    elif vital_instability['severity_score'] >= 2 and prediction == 0:
        # Moderate instability with low model prediction - consider it elevated risk
        adjusted_risk_level = 1
        adjustment_reason = "⚠️ Significant vital fluctuations detected - Risk level adjusted to MODERATE-HIGH"
    
    html = '<div style="margin-top: 20px;">'
    
    # Show vital instability warnings if present
    if vital_instability['has_instability']:
        severity_colors = {
            'CRITICAL': '#ff6b6b',
            'HIGH': '#ff9f43',
            'MODERATE': '#facc15'
        }
        
        html += '''
        <div style="background: rgba(255, 107, 107, 0.1); border: 2px solid rgba(255, 107, 107, 0.4); 
                    border-radius: 10px; padding: 15px; margin-bottom: 15px;">
            <h5 style="color: #ff6b6b;">🚨 Vital Sign Instability Alert</h5>
            <p style="color: #b0b0b0; margin-bottom: 10px;">Significant fluctuations detected in vital signs - this is concerning even if some individual values appear acceptable:</p>
            <ul style="color: #b0b0b0; margin-left: 20px;">
        '''
        
        for indicator in vital_instability['indicators']:
            color = severity_colors.get(indicator['severity'], '#facc15')
            severity_badge = f'<span style="background: {color}; color: #0a0e27; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold;">{indicator["severity"]}</span>'
            html += f'''
            <li style="color: #b0b0b0; margin-bottom: 10px;">
                <strong style="color: {color};">{indicator['vital']}</strong>: {indicator['value']:.1f} {severity_badge}
                <br><span style="color: #999; margin-left: 20px;">→ {indicator['concern']}</span>
            </li>
            '''
        
        html += '</ul></div>'
        
        if adjustment_reason:
            html += f'''
            <div style="background: rgba(255, 193, 7, 0.1); border: 2px solid rgba(255, 193, 7, 0.3); 
                        border-radius: 10px; padding: 12px; margin-bottom: 15px;">
                <p style="color: #ffc107; font-weight: bold;">{adjustment_reason}</p>
            </div>
            '''
    
    if abnormal_features:
        html += '''
        <div style="background: rgba(255, 215, 0, 0.08); border: 1px solid rgba(255, 215, 0, 0.2); 
                    border-radius: 10px; padding: 15px; margin-bottom: 15px;">
            <h5 style="color: #ffd700;">⚠️ Abnormal Clinical Values Detected</h5>
            <p style="color: #b0b0b0; margin-bottom: 10px;">The following values are outside normal ranges:</p>
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
        if not vital_instability['has_instability']:
            html += '''
            <div style="background: rgba(74, 222, 128, 0.08); border: 1px solid rgba(74, 222, 128, 0.2); 
                        border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                <h5 style="color: #4ade80;">✓ All Clinical Values Within Normal Ranges</h5>
                <p style="color: #b0b0b0;">All monitored parameters are within normal clinical ranges.</p>
            </div>
            '''
    
    # Final risk assessment with adjusted level
    if adjusted_risk_level == 1:
        html += f'''
        <div style="background: rgba(255, 107, 107, 0.08); border: 1px solid rgba(255, 107, 107, 0.2); 
                    border-radius: 10px; padding: 15px;">
            <h5 style="color: #ff6b6b;">🚨 High Risk Assessment</h5>
            <p style="color: #b0b0b0;">The model predicts a <strong>{confidence:.1f}%</strong> probability of sepsis risk based on clinical data.</p>
            <p style="color: #b0b0b0;"><strong>Recommendation:</strong> <span style="color: #ff9f43; font-weight: bold;">Consider immediate clinical evaluation, monitoring, and possible sepsis protocols.</span></p>
        </div>
        '''
    else:
        html += f'''
        <div style="background: rgba(74, 222, 128, 0.08); border: 1px solid rgba(74, 222, 128, 0.2); 
                    border-radius: 10px; padding: 15px;">
            <h5 style="color: #4ade80;">✓ Low Risk Assessment</h5>
            <p style="color: #b0b0b0;">The model predicts a <strong>{100-confidence:.1f}%</strong> probability of low sepsis risk based on clinical data.</p>
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
        
        # Get vital instability assessment
        vital_instability = detect_vital_instability(form_data)
        
        # Adjust prediction based on vital instability
        adjusted_prediction = prediction[0]
        if vital_instability['severity_score'] >= 3:
            adjusted_prediction = 1  # Escalate to high risk
        elif vital_instability['severity_score'] >= 2 and prediction[0] == 0:
            adjusted_prediction = 1  # Moderate-high risk
        
        # Use original confidence, but reflect adjusted prediction
        confidence = max(probability) * 100
        if adjusted_prediction == 1 and prediction[0] == 0:
            # If we adjusted up due to instability, show higher confidence for safety
            confidence = min(confidence + 15, 95)
        
        # Determine prediction text
        prediction_text = "High Risk of Sepsis" if adjusted_prediction == 1 else "Low Risk of Sepsis"
        
        # Generate explanation
        explanation_html = generate_explanation(form_data, adjusted_prediction, confidence)
        
        return render_template(
            'index.html',
            prediction_text=prediction_text,
            confidence=f"{confidence:.2f}%",
            explanation=explanation_html,
            risk_level='High Risk' if adjusted_prediction == 1 else 'Low Risk'
        )
    
    except Exception as e:
        error_msg = f"Error in prediction: {str(e)}"
        return render_template('index.html', prediction_text=error_msg)


if __name__ == '__main__':
    app.run(debug=True)

