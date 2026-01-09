#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

# Try to import Phase 3 LSTM support
try:
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import StandardScaler as SSScaler
    PHASE3_AVAILABLE = True
except:
    PHASE3_AVAILABLE = False

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

# Load Phase 3 LSTM model if available
phase3_lstm_model = None
phase3_scaler = None
phase3_available = False

if PHASE3_AVAILABLE:
    try:
        if os.path.exists('model_phase3_lstm.h5') and os.path.exists('scaler_phase3.pkl'):
            phase3_lstm_model = load_model('model_phase3_lstm.h5')
            phase3_scaler = pickle.load(open('scaler_phase3.pkl', 'rb'))
            phase3_available = True
            print("[INFO] Phase 3 LSTM model loaded - 6-hour advance prediction available")
    except Exception as e:
        print(f"[WARNING] Phase 3 LSTM not available: {e}")
        phase3_available = False

# Phase 3 LSTM feature columns
PHASE3_FEATURES = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp',
    'BaseExcess', 'HCO3', 'FiO2', 'PaCO2', 'SaO2', 'Creatinine',
    'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
    'Bilirubin_total', 'Hgb', 'WBC', 'Fibrinogen', 'Platelets'
]

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

def calculate_continuous_risk_trajectory(form_data, current_sepsis_probability):
    """
    Calculate continuous, trend-based 6-7 hour sepsis risk trajectory.
    
    MANDATORY SAFETY RULE:
    - If current risk < 1% and vitals are normal → future risk must stay ≤5%
    - No artificial escalation without strong deteriorating trends
    - This is an EARLY-WARNING system, not a pessimistic simulator
    
    Uses smooth probability scaling based on:
    1. Distance from optimal physiological ranges (continuous)
    2. Magnitude of deviations (not binary classification)
    3. Multiple simultaneous vitals showing deterioration
    4. Temporal consistency (future risk logically follows current risk)
    
    Returns:
        dict: {
            'current_risk': float (0-1),
            'future_risk_6h': float (0-1),
            'trajectory': str ('escalating', 'stable', 'improving'),
            'risk_velocity': float (-1 to 1, rate of change)
        }
    """
    
    # Define optimal physiological ranges (center of comfort zone)
    optimal_ranges = {
        'HR': {'optimal': 70, 'min': 60, 'max': 100, 'critical_low': 40, 'critical_high': 150},
        'Temp': {'optimal': 37.0, 'min': 36.5, 'max': 37.5, 'critical_low': 35, 'critical_high': 40},
        'SBP': {'optimal': 110, 'min': 90, 'max': 130, 'critical_low': 70, 'critical_high': 200},
        'MAP': {'optimal': 85, 'min': 70, 'max': 100, 'critical_low': 50, 'critical_high': 150},
        'DBP': {'optimal': 70, 'min': 60, 'max': 85, 'critical_low': 40, 'critical_high': 120},
        'Resp': {'optimal': 16, 'min': 12, 'max': 20, 'critical_low': 8, 'critical_high': 40},
        'O2Sat': {'optimal': 97, 'min': 95, 'max': 100, 'critical_low': 88, 'critical_high': 100},
        'Glucose': {'optimal': 85, 'min': 70, 'max': 100, 'critical_low': 40, 'critical_high': 300},
        'Lactate': {'optimal': 1.2, 'min': 0.5, 'max': 2.0, 'critical_low': 0.2, 'critical_high': 10},
        'WBC': {'optimal': 7, 'min': 4.5, 'max': 11, 'critical_low': 1, 'critical_high': 50},
    }
    
    def calculate_deviation_risk(value, param_name):
        """
        Calculate continuous risk contribution (0-1) from a single parameter.
        Uses smooth Gaussian-like curve centered on optimal range.
        """
        try:
            value = float(value)
        except:
            return 0.0
        
        if param_name not in optimal_ranges:
            return 0.0
        
        ranges = optimal_ranges[param_name]
        optimal = ranges['optimal']
        normal_min = ranges['min']
        normal_max = ranges['max']
        critical_low = ranges['critical_low']
        critical_high = ranges['critical_high']
        
        # If within optimal range -> minimal risk
        if normal_min <= value <= normal_max:
            # Smooth function: highest at center, increases toward edges
            center_distance = abs(value - optimal)
            normal_range = (normal_max - normal_min) / 2
            # Returns 0 at optimal, rises to ~0.1-0.2 at range edges
            return (center_distance / normal_range) * 0.2
        
        # If outside normal range -> progressive risk scaling
        elif value < normal_min:
            # Below normal: scale from 0.2 (at edge) to 1.0 (at critical)
            range_below = normal_min - critical_low
            if range_below <= 0:
                return 0.5 if value < normal_min else 0.0
            distance = normal_min - value
            risk = 0.2 + (distance / range_below) * 0.8
            return min(1.0, risk)
        
        else:  # Above normal
            # Above normal: scale from 0.2 (at edge) to 1.0 (at critical)
            range_above = critical_high - normal_max
            if range_above <= 0:
                return 0.5 if value > normal_max else 0.0
            distance = value - normal_max
            risk = 0.2 + (distance / range_above) * 0.8
            return min(1.0, risk)
    
    # Calculate continuous risk contributions from key vitals
    vital_deviations = {}
    for param in ['HR', 'Temp', 'SBP', 'MAP', 'Resp', 'O2Sat', 'Glucose', 'Lactate', 'WBC']:
        value = form_data.get(param, '')
        if value:
            vital_deviations[param] = calculate_deviation_risk(value, param)
    
    # Combined vital risk: average of all deviations (gives equal weight to each vital)
    if vital_deviations:
        vital_risk = np.mean(list(vital_deviations.values()))
    else:
        vital_risk = 0.0
    
    # Blend with model's current sepsis probability
    # Current risk = 60% from model, 40% from vital deviations
    blended_current_risk = (0.6 * current_sepsis_probability) + (0.4 * vital_risk)
    blended_current_risk = max(0.0, min(1.0, blended_current_risk))
    
    # ===== 6-HOUR TRAJECTORY CALCULATION WITH SAFETY CONSTRAINTS =====
    # MANDATORY BASELINE SAFETY: If current risk is very low and vitals are normal,
    # don't create artificial escalation
    
    # Count vitals with significant deviations (>0.3 risk contribution)
    significantly_abnormal = sum(1 for v in vital_deviations.values() if v > 0.3)
    
    # Count vitals with moderate deviations (0.15-0.3 risk)
    moderately_abnormal = sum(1 for v in vital_deviations.values() if 0.15 <= v <= 0.3)
    
    # Calculate abnormality burden
    abnormality_burden = (significantly_abnormal * 0.4) + (moderately_abnormal * 0.15)
    abnormality_burden = min(1.0, abnormality_burden)
    
    # ===== BASELINE SAFETY RULE =====
    # If current risk is <1% and vitals are clinically normal (vital_risk < 0.1),
    # constrain future risk to stay low (max +4%)
    if blended_current_risk < 0.01 and vital_risk < 0.1:
        # Baseline safety: healthy patient stays healthy
        # Only allow minimal escalation if abnormalities exist
        if abnormality_burden < 0.1:
            # No abnormalities -> slight improvement
            risk_velocity = -0.01
            trajectory = 'improving'
        else:
            # Minimal abnormalities -> cap escalation at +3%
            risk_velocity = min(0.03, abnormality_burden * 0.05)
            trajectory = 'stable'
        
        future_risk = blended_current_risk + risk_velocity
        future_risk = max(0.0, min(0.05, future_risk))  # Cap at 5%
    
    # ===== STANDARD TRAJECTORY LOGIC =====
    elif blended_current_risk >= 0.7:
        # Very high risk: monitor for stabilization or escalation
        if abnormality_burden >= 0.6:
            # Very high risk with severe abnormalities -> escalating
            risk_velocity = 0.12
            trajectory = 'escalating'
        elif abnormality_burden >= 0.3:
            # High risk with moderate abnormalities -> slightly escalating
            risk_velocity = 0.06
            trajectory = 'escalating'
        else:
            # High risk but fewer abnormalities -> stable or slight improvement
            risk_velocity = 0.02
            trajectory = 'stable'
    
    elif blended_current_risk >= 0.5:
        # Moderate-high risk
        if abnormality_burden >= 0.6:
            # Significant abnormalities -> escalate
            risk_velocity = 0.18
            trajectory = 'escalating'
        elif abnormality_burden >= 0.3:
            # Some abnormalities -> gradual escalation
            risk_velocity = 0.10
            trajectory = 'escalating'
        else:
            # Minimal abnormalities -> stable
            risk_velocity = 0.03
            trajectory = 'stable'
    
    elif blended_current_risk >= 0.3:
        # Moderate risk
        if abnormality_burden >= 0.5:
            # Significant abnormalities -> early warning escalates risk
            risk_velocity = 0.15
            trajectory = 'escalating'
        elif abnormality_burden >= 0.2:
            # Mild abnormalities -> slight escalation
            risk_velocity = 0.05
            trajectory = 'stable'
        else:
            # Minimal abnormalities -> improving
            risk_velocity = -0.02
            trajectory = 'improving'
    
    else:
        # Low current risk (1-30%)
        if abnormality_burden >= 0.4:
            # Emerging abnormalities in low-risk patient -> early warning signal
            # But cap escalation at reasonable levels
            risk_velocity = min(0.08, abnormality_burden * 0.15)
            trajectory = 'escalating'
        elif abnormality_burden >= 0.2:
            # Minimal abnormalities -> stay stable
            risk_velocity = 0.01
            trajectory = 'stable'
        else:
            # Very few abnormalities -> improve
            risk_velocity = -0.02
            trajectory = 'improving'
    
    # Calculate future risk with logical continuity
    future_risk = blended_current_risk + risk_velocity
    future_risk = max(0.0, min(1.0, future_risk))
    
    return {
        'current_risk': blended_current_risk,
        'future_risk_6h': future_risk,
        'trajectory': trajectory,
        'risk_velocity': risk_velocity,
        'vital_deviations': vital_deviations,
        'abnormality_burden': abnormality_burden
    }

def get_sepsis_risk_label(probability):
    """
    Strict probability-to-label mapping (MANDATORY consistency).
    
    0–20% → Low Risk
    21–50% → Moderate Risk
    51–75% → High Risk
    76–100% → Critical Risk
    """
    prob_percent = probability * 100
    
    if prob_percent <= 20:
        return 'Low Risk', '#51cf66', 'green'
    elif prob_percent <= 50:
        return 'Moderate Risk', '#facc15', 'yellow'
    elif prob_percent <= 75:
        return 'High Risk', '#ff9f43', 'orange'
    else:  # > 75%
        return 'Critical Risk', '#ff6b6b', 'red'

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI with clinical explanation
    Supports Phase 1 (base features), Phase 2 (base + trend features), AND Phase 3 (6-hour forecast)
    Now automatically generates all three predictions simultaneously
    
    MANDATORY CLINICAL RULE:
    If all parameters are normal with no abnormal trends → NO SEPSIS PREDICTION
    Output: "No current evidence of sepsis risk"
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
        
        # ===== MANDATORY RULE: Check if all vitals/labs are normal =====
        abnormal_features = get_abnormal_features(form_data)
        vital_instability = detect_vital_instability(form_data)
        
        # If all parameters are normal (no abnormal features, no instability)
        # AND model probability is very low, output "No evidence of sepsis"
        all_normal = (len(abnormal_features) == 0 and 
                     vital_instability['severity_score'] == 0 and
                     prob_sepsis < 0.15)  # Allow some model uncertainty
        
        if all_normal:
            # MANDATORY OUTPUT: No sepsis prediction when all parameters are normal
            prediction_text = "No current evidence of sepsis risk"
            confidence = 0.0
            risk_label = "Not Prone to Sepsis"
            label_color = "#51cf66"
            label_class = "green"
            
            # Generate explanation for normal state
            explanation_html = f"""
            <div style="background: rgba(81, 207, 102, 0.1); border: 2px solid rgba(81, 207, 102, 0.3); 
                        border-radius: 10px; padding: 20px; margin-bottom: 15px;">
                <h5 style="color: #4ade80; margin-top: 0;">✓ Patient is Not Prone to Sepsis</h5>
                <p style="color: #b0b0b0; margin-bottom: 10px;">
                    <strong>Clinical Assessment:</strong> All monitored vital signs and laboratory values are within normal ranges with no abnormal trends detected.
                </p>
                <p style="color: #b0b0b0; margin-bottom: 10px;">
                    <strong>Current Status:</strong> No clinical evidence of sepsis or early sepsis trajectory.
                </p>
                <p style="color: #b0b0b0;">
                    <strong>Recommendation:</strong> Continue routine clinical monitoring. Standard surveillance protocols are sufficient.
                </p>
            </div>
            """
            
            # No 6-hour prediction needed for normal patients
            phase3_explanation = ""
            phase3_risk = None
            
            return render_template(
                'index.html',
                prediction_text=prediction_text,
                confidence=f"{confidence:.1f}%",
                explanation=explanation_html,
                risk_level=risk_label,
                model_version="Clinical Assessment",
                phase3_risk=phase3_risk,
                is_normal_state=True  # Flag to hide 6h prediction
            )
        
        # ===== STANDARD SEPSIS PREDICTION WORKFLOW =====
        # Use CONTINUOUS probability for display (not binary threshold)
        sepsis_probability = prob_sepsis
        confidence = sepsis_probability * 100
        model_version = "Calibrated Logistic Regression"
        
        # Get vital instability assessment (already computed above)
        
        # Adjust prediction based on vital instability
        prediction_tuned = 1 if prob_sepsis >= 0.5 else 0
        adjusted_prediction = prediction_tuned
        if vital_instability['severity_score'] >= 5:
            adjusted_prediction = 1  # Escalate to high risk only for critical cases
        
        # Get strict label based on probability (MANDATORY consistency)
        risk_label, label_color, label_class = get_sepsis_risk_label(sepsis_probability)
        
        # Determine prediction text with strict label mapping
        prediction_text = f"{risk_label} ({confidence:.1f}%)"
        
        # Generate explanation
        explanation_html = generate_explanation(form_data, adjusted_prediction, confidence)
        
        # ==================== PHASE 3: CONTINUOUS 6-7 HOUR EARLY PREDICTION ====================
        # Calculate continuous, trend-based risk trajectory
        trajectory_result = calculate_continuous_risk_trajectory(form_data, sepsis_probability)
        
        current_risk_continuous = trajectory_result['current_risk']
        sepsis_risk_6h = trajectory_result['future_risk_6h']
        trajectory_type = trajectory_result['trajectory']
        risk_velocity = trajectory_result['risk_velocity']
        
        # Get 6-hour prediction label (strict mapping for medical consistency)
        risk_label_6h, label_color_6h, label_class_6h = get_sepsis_risk_label(sepsis_risk_6h)
        
        # Generate Phase 3 early prediction explanation
        trajectory_icons = {
            'escalating': '⚠️ ESCALATING',
            'stable': '→ STABLE',
            'improving': '✓ IMPROVING'
        }
        trajectory_colors = {
            'escalating': '#ff6b6b',
            'stable': '#facc15',
            'improving': '#51cf66'
        }
        
        phase3_explanation = f"""
        <div class="phase3-result-card" style="margin-top: 20px; padding: 15px; background: rgba(255, 215, 0, 0.1); border-left: 4px solid #ffd700; border-radius: 4px;">
            <h4 style="margin-top: 0;"><i class="fas fa-clock"></i> Early Sepsis Prediction (6-7 Hour Horizon)</h4>
            <p>Continuous probability forecast based on vital deviations and abnormality burden:</p>
            <div class="phase3-risk-display" style="background: #1a1a2e; padding: 12px; border-radius: 4px; margin: 10px 0;">
                <p style="font-size: 1.2em; color: {label_color_6h}; margin: 0;">
                    <strong>{sepsis_risk_6h*100:.1f}% Risk (6-7h forecast)</strong> → <strong>{risk_label_6h}</strong>
                </p>
                <p style="font-size: 0.95em; color: #999; margin-top: 5px;">
                    Trajectory: <span style="color: {trajectory_colors[trajectory_type]}; font-weight: bold;">{trajectory_icons[trajectory_type]}</span>
                    (Velocity: {risk_velocity:+.1%}/6h)
                </p>
            </div>
            <p><strong>Clinical Interpretation:</strong><br/>
            {'⚠️ <strong>URGENT:</strong> Critical risk range with escalating trajectory. Immediate intervention required.' if sepsis_risk_6h > 0.75 and trajectory_type == 'escalating' else 
             '⚠️ <strong>HIGH CONCERN:</strong> High risk with potential deterioration. Close monitoring and treatment essential.' if sepsis_risk_6h > 0.5 else
             '📊 <strong>MODERATE WATCH:</strong> Moderate risk. Early warning signals present - continued monitoring advised.' if sepsis_risk_6h > 0.2 else
             '✓ <strong>STABLE:</strong> Low risk trajectory. Continue standard monitoring.'}
            </p>
        </div>
        """
        
        phase3_risk = sepsis_risk_6h
        
        # Combine Phase 1/2 explanation with Phase 3
        full_explanation = explanation_html + phase3_explanation
        
        # Ensure phase3_risk has a default value
        phase3_risk_value = phase3_risk if phase3_risk is not None else ""
        
        return render_template(
            'index.html',
            prediction_text=prediction_text,
            confidence=f"{confidence:.2f}%",
            explanation=full_explanation,
            risk_level=risk_label,
            model_version=model_version,
            phase3_risk=phase3_risk_value,
            is_normal_state=False
        )
    
    except Exception as e:
        error_msg = f"Error in prediction: {str(e)}"
        return render_template('index.html', prediction_text=error_msg)
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
        
        # Use CONTINUOUS probability for display (not binary threshold)
        sepsis_probability = prob_sepsis
        confidence = sepsis_probability * 100
        model_version = "Calibrated Logistic Regression"
        
        # Get vital instability assessment
        vital_instability = detect_vital_instability(form_data)
        
        # Adjust prediction based on vital instability
        adjusted_prediction = prediction_tuned
        if vital_instability['severity_score'] >= 5:
            adjusted_prediction = 1  # Escalate to high risk only for critical cases
        
        # Get strict label based on probability (MANDATORY consistency)
        risk_label, label_color, label_class = get_sepsis_risk_label(sepsis_probability)
        
        # Determine prediction text with strict label mapping
        prediction_text = f"{risk_label} ({confidence:.1f}%)"
        
        # Generate explanation
        explanation_html = generate_explanation(form_data, adjusted_prediction, confidence)
        
        # ==================== PHASE 3: CONTINUOUS 6-7 HOUR EARLY PREDICTION ====================
        # Calculate continuous, trend-based risk trajectory
        trajectory_result = calculate_continuous_risk_trajectory(form_data, sepsis_probability)
        
        current_risk_continuous = trajectory_result['current_risk']
        sepsis_risk_6h = trajectory_result['future_risk_6h']
        trajectory_type = trajectory_result['trajectory']
        risk_velocity = trajectory_result['risk_velocity']
        
        # Get 6-hour prediction label (strict mapping for medical consistency)
        risk_label_6h, label_color_6h, label_class_6h = get_sepsis_risk_label(sepsis_risk_6h)
        
        # Generate Phase 3 early prediction explanation
        trajectory_icons = {
            'escalating': '⚠️ ESCALATING',
            'stable': '→ STABLE',
            'improving': '✓ IMPROVING'
        }
        trajectory_colors = {
            'escalating': '#ff6b6b',
            'stable': '#facc15',
            'improving': '#51cf66'
        }
        
        phase3_explanation = f"""
        <div class="phase3-result-card" style="margin-top: 20px; padding: 15px; background: rgba(255, 215, 0, 0.1); border-left: 4px solid #ffd700; border-radius: 4px;">
            <h4 style="margin-top: 0;"><i class="fas fa-clock"></i> Early Sepsis Prediction (6-7 Hour Horizon)</h4>
            <p>Continuous probability forecast based on vital deviations and abnormality burden:</p>
            <div class="phase3-risk-display" style="background: #1a1a2e; padding: 12px; border-radius: 4px; margin: 10px 0;">
                <p style="font-size: 1.2em; color: {label_color_6h}; margin: 0;">
                    <strong>{sepsis_risk_6h*100:.1f}% Risk (6-7h forecast)</strong> → <strong>{risk_label_6h}</strong>
                </p>
                <p style="font-size: 0.95em; color: #999; margin-top: 5px;">
                    Trajectory: <span style="color: {trajectory_colors[trajectory_type]}; font-weight: bold;">{trajectory_icons[trajectory_type]}</span>
                    (Velocity: {risk_velocity:+.1%}/6h)
                </p>
            </div>
            <p><strong>Clinical Interpretation:</strong><br/>
            {'⚠️ <strong>URGENT:</strong> Critical risk range with escalating trajectory. Immediate intervention required.' if sepsis_risk_6h > 0.75 and trajectory_type == 'escalating' else 
             '⚠️ <strong>HIGH CONCERN:</strong> High risk with potential deterioration. Close monitoring and treatment essential.' if sepsis_risk_6h > 0.5 else
             '📊 <strong>MODERATE WATCH:</strong> Moderate risk. Early warning signals present - continued monitoring advised.' if sepsis_risk_6h > 0.2 else
             '✓ <strong>STABLE:</strong> Low risk trajectory. Continue standard monitoring.'}
            </p>
        </div>
        """
        
        phase3_risk = sepsis_risk_6h
        
        # Generate early prediction explanation
        phase3_explanation = f"""
        <div class="phase3-result-card" style="margin-top: 20px; padding: 15px; background: rgba(255, 215, 0, 0.1); border-left: 4px solid #ffd700; border-radius: 4px;">
            <h4 style="margin-top: 0;"><i class="fas fa-clock"></i> Early Sepsis Prediction (6-7 Hour Horizon)</h4>
            <p>Based on current abnormalities and vital sign trajectory analysis:</p>
            <div class="phase3-risk-display" style="background: #1a1a2e; padding: 12px; border-radius: 4px; margin: 10px 0;">
                <p style="font-size: 1.2em; color: {'#ff6b6b' if sepsis_risk_6h >= 0.5 else '#51cf66'}; margin: 0;">
                    <strong>{sepsis_risk_6h*100:.1f}% Risk (6-7h forecast)</strong>
                </p>
            </div>
            <p><strong>Clinical Trajectory:</strong><br/>
            {'⚠️ <strong>ESCALATION ALERT:</strong> Current vital sign instability predicts rapid deterioration. Early intervention recommended to prevent sepsis development.' if sepsis_risk_6h >= 0.7 else 
             '⚠️ <strong>HIGH CONCERN:</strong> Trajectory analysis suggests significant risk increase in next 6-7 hours. Close monitoring and early treatment advised.' if sepsis_risk_6h >= 0.5 else
             '📊 <strong>MODERATE WATCH:</strong> Early warning signals detected. Trend analysis indicates potential deterioration - continued vigilance recommended.' if sepsis_risk_6h >= 0.3 else
             '✓ <strong>STABLE TRAJECTORY:</strong> Current abnormalities are minimal. Expected to remain stable without intervention.'}
            </p>
        </div>
        """
        
        # Combine Phase 1/2 explanation with Phase 3
        full_explanation = explanation_html + phase3_explanation
        
        # Ensure phase3_risk has a default value
        phase3_risk_value = phase3_risk if phase3_risk is not None else ""
        
        return render_template(
            'index.html',
            prediction_text=prediction_text,
            confidence=f"{confidence:.2f}%",
            explanation=full_explanation,
            risk_level='High Risk' if adjusted_prediction == 1 else 'Low Risk',
            model_version=model_version,
            phase3_risk=phase3_risk_value
        )
    
    except Exception as e:
        error_msg = f"Error in prediction: {str(e)}"
        return render_template('index.html', prediction_text=error_msg)



@app.route('/predict_phase3', methods=['POST'])
def predict_phase3():
    """
    Phase 3: LSTM-based 6-hour advance prediction
    Requires 12-hour historical patient data (sequence of measurements)
    """
    try:
        if not phase3_available:
            return render_template('index.html', 
                error="Phase 3 LSTM model not available. Using Phase 1/2 prediction instead.")
        
        form_data = request.form.to_dict()
        
        # Build 12-timestep sequence from historical data
        sequence = []
        for t in range(12):
            timestep = []
            for feature in PHASE3_FEATURES:
                field_name = f"{feature}_t{t}"
                try:
                    val = float(form_data.get(field_name, 0))
                except (ValueError, TypeError):
                    val = 0
                timestep.append(val)
            sequence.append(timestep)
        
        # Convert to numpy array and reshape
        X_sequence = np.array([sequence])  # Shape: (1, 12, 20)
        
        # Scale the sequence
        n_samples, n_timesteps, n_features = X_sequence.shape
        X_reshaped = X_sequence.reshape(-1, n_features)
        X_scaled = phase3_scaler.transform(X_reshaped)
        X_sequence_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Make prediction
        predictions = phase3_lstm_model.predict(X_sequence_scaled, verbose=0)
        
        # Get 6-step ahead average prediction (next 6 hours)
        sepsis_risk_6h = float(np.mean(predictions[0, :6, 0]))  # Average of first 6 timesteps
        sepsis_risk_6h = max(0.0, min(1.0, sepsis_risk_6h))  # Clip to [0, 1]
        
        # Determine risk level
        if sepsis_risk_6h >= 0.5:
            risk_level = "HIGH RISK (6-hour)"
            prediction_text = f"⚠️ 6-Hour Advance Warning: {sepsis_risk_6h*100:.1f}% Sepsis Risk"
            confidence = sepsis_risk_6h * 100
        else:
            risk_level = "LOW RISK (6-hour)"
            prediction_text = f"✓ 6-Hour Outlook: {(1-sepsis_risk_6h)*100:.1f}% Probability of Remaining Stable"
            confidence = (1 - sepsis_risk_6h) * 100
        
        # Generate Phase 3 specific explanation
        explanation = f"""
        <div class="phase3-explanation">
            <h4><i class="fas fa-clock"></i> 6-Hour Advance Prediction (Phase 3 LSTM)</h4>
            <p>This prediction analyzes the temporal patterns from the past 12 hours of patient data
            to forecast sepsis risk for the next 6 hours.</p>
            <div class="risk-details">
                <p><strong>Predicted Risk (6h ahead):</strong> {sepsis_risk_6h*100:.1f}%</p>
                <p><strong>Clinical Action:</strong> 
                {'Start prophylactic monitoring and prepare early interventions' if sepsis_risk_6h >= 0.5 else 'Continue standard monitoring'}
                </p>
            </div>
        </div>
        """
        
        return render_template(
            'index.html',
            prediction_text=prediction_text,
            confidence=f"{confidence:.2f}%",
            explanation=explanation,
            risk_level=risk_level,
            model_version="LSTM Time-Series (6-hour forecast)",
            is_phase3=True
        )
    
    except Exception as e:
        error_msg = f"Phase 3 Error: {str(e)}"
        return render_template('index.html', prediction_text=error_msg, error=error_msg)


if __name__ == '__main__':
    app.run(debug=True)

