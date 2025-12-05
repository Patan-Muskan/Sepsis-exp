#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/static')

# Try to load calibrated model (Random Forest with probability scaling)
scaling_params = None
try:
    model = pickle.load(open('model_calibrated.pkl', 'rb'))
    scaler = pickle.load(open('scaler_calibrated.pkl', 'rb'))
    try:
        scaling_params = pickle.load(open('scaling_params.pkl', 'rb'))
    except:
        scaling_params = None
    print("[INFO] Using Random Forest with linear probability scaling")
except:
    # Fallback to Phase 1 model
    try:
        model = pickle.load(open('model_phase2.pkl', 'rb'))
        if model.n_features_in_ == 43:
            print("[INFO] Phase 2 model requires trend features - switching to Phase 1")
            model = pickle.load(open('model.pkl', 'rb'))
        else:
            print("[INFO] Using Phase 2 model")
    except:
        model = pickle.load(open('model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        print("[INFO] Using Phase 1 model")

# Load Phase 2 threshold info if available (for reference only)
try:
    threshold_info = pickle.load(open('threshold_info.pkl', 'rb'))
    optimal_threshold = threshold_info.get('optimal_threshold', 0.5)
except:
    threshold_info = None
    optimal_threshold = 0.5

# Base features (27 features)
FEATURE_NAMES = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp',
    'BaseExcess', 'HCO3', 'FiO2', 'PaCO2', 'SaO2', 'Creatinine',
    'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
    'Bilirubin_total', 'Hgb', 'WBC', 'Fibrinogen', 'Platelets',
    'Age', 'Gender', 'HospAdmTime', 'ICULOS'
]

# Phase 2 trend features (optional)
TREND_FEATURES = [
    'HR_trend_1h', 'HR_volatility', 'O2Sat_trend_1h', 'O2Sat_volatility',
    'Temp_trend_1h', 'Temp_volatility', 'Lactate_trend_1h', 'Lactate_volatility',
    'SBP_trend_1h', 'SBP_volatility', 'Creatinine_trend_1h', 'Creatinine_volatility',
    'WBC_trend_1h', 'WBC_volatility', 'Glucose_trend_1h', 'Glucose_volatility'
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
    
    # Only escalate prediction if CRITICAL vital issues exist (very high thresholds)
    # Don't override model prediction for mild abnormalities
    if vital_instability['severity_score'] >= 5:
        # Only escalate if extremely critical (e.g., cardiac shock, severe respiratory distress)
        adjusted_risk_level = 1
        adjustment_reason = "⚠️ CRITICAL vital sign abnormalities detected - Risk escalated to HIGH"
    # NOTE: Removed the aggressive escalation that was converting all abnormal values to sepsis risk
    
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
    Supports both Phase 1 (base features) and Phase 2 (base + trend features)
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
        
        # Apply scaler if available (Phase 1 optimization)
        if scaler is not None:
            final_features = scaler.transform(final_features)
        
        # Make prediction using model
        prediction = model.predict(final_features)
        probability = model.predict_proba(final_features)[0]
        
        # Get raw sepsis probability
        prob_sepsis_raw = probability[1]
        
        # Apply probability scaling if available (full 0-100% range)
        if scaling_params is not None:
            prob_min = scaling_params['prob_min']
            prob_max = scaling_params['prob_max']
            # Scale to full [0, 1] range for complete 0-100% spread
            prob_sepsis = (prob_sepsis_raw - prob_min) / (prob_max - prob_min)
            prob_sepsis = max(0.0, min(1.0, prob_sepsis))  # Clip to [0, 1]
        else:
            prob_sepsis = prob_sepsis_raw
        
        # Ensure probability is in valid range [0, 1]
        prob_sepsis = max(0.0, min(1.0, prob_sepsis))
        prob_no_sepsis = 1 - prob_sepsis
        
        # Use 0.5 threshold for binary prediction
        prediction_tuned = 1 if prob_sepsis >= 0.5 else 0
        
        # Display confidence for the predicted class
        if prediction_tuned == 1:
            confidence = prob_sepsis * 100
        else:
            confidence = prob_no_sepsis * 100
        model_version = "Calibrated Logistic Regression"
        
        # Get vital instability assessment
        vital_instability = detect_vital_instability(form_data)
        
        # Adjust prediction based on vital instability
        adjusted_prediction = prediction_tuned
        if vital_instability['severity_score'] >= 5:
            adjusted_prediction = 1  # Escalate to high risk only for critical cases
        
        # Determine prediction text
        if adjusted_prediction == 1:
            prediction_text = f"High Risk of Sepsis ({confidence:.1f}%)"
        else:
            prediction_text = f"Low Risk of Sepsis ({confidence:.1f}% probability of no sepsis)"
        
        # Generate explanation
        explanation_html = generate_explanation(form_data, adjusted_prediction, confidence)
        
        return render_template(
            'index.html',
            prediction_text=prediction_text,
            confidence=f"{confidence:.2f}%",
            explanation=explanation_html,
            risk_level='High Risk' if adjusted_prediction == 1 else 'Low Risk',
            model_version=model_version
        )
    
    except Exception as e:
        error_msg = f"Error in prediction: {str(e)}"
        return render_template('index.html', prediction_text=error_msg)


if __name__ == '__main__':
    app.run(debug=True)

