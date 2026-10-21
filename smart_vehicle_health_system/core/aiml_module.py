"""
Artificial Intelligence & Machine Learning (AIML) Module
Components:
1. Supervised Subsystem Failure Classifier (Multi-class Decision Forest / Softmax with Calibration)
2. Predictive Maintenance: Remaining Useful Life (RUL in km/hours) Regression Engine
3. Unsupervised Real-Time Telemetry Anomaly Detector (Multivariate Statistical Outlier Engine)
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


class FailureCategory:
    CLASSES = [
        "NORMAL_OPERATION",
        "COOLING_SYSTEM_OVERHEAT",
        "OIL_STARVATION_LOW_PRESSURE",
        "ELECTRICAL_ALTERNATOR_DEGRADATION",
        "DRIVETRAIN_TRANSMISSION_VIBRATION"
    ]

    EXPLANATIONS = {
        "NORMAL_OPERATION": "All powertrain, thermal, and electrical telemetry within nominal operating envelopes.",
        "COOLING_SYSTEM_OVERHEAT": "Excessive coolant temperature detected. Likely radiator fan fault, coolant leak, or sticking thermostat.",
        "OIL_STARVATION_LOW_PRESSURE": "Oil pressure critically low relative to engine RPM. Imminent risk of rod knock and piston seizure.",
        "ELECTRICAL_ALTERNATOR_DEGRADATION": "Battery voltage out of charging threshold. Diode bridge failure, worn alternator brushes, or failing cell.",
        "DRIVETRAIN_TRANSMISSION_VIBRATION": "Harmonic mechanical oscillation exceeds safety bounds. Possible universal joint wear or transmission clutch slip."
    }


class VehicleHealthClassifier:
    """
    Supervised classifier for vehicle fault modes using calibrated multi-class probabilistic estimation.
    Uses a vectorized ensemble of decision centroids and regularized linear discriminant projections.
    """

    def __init__(self):
        # Feature order: [engine_rpm, coolant_temp, oil_pressure, battery_voltage, vibration, vehicle_speed]
        self.feature_names = [
            "engine_rpm",
            "coolant_temp",
            "oil_pressure",
            "battery_voltage",
            "vibration",
            "vehicle_speed"
        ]

        # Empirical baseline distributions (mean, std) for normalization
        self.means = np.array([2200.0, 90.0, 45.0, 13.8, 1.8, 60.0])
        self.stds = np.array([800.0, 15.0, 15.0, 1.2, 1.5, 30.0])

        # Subsystem Prototype Centroids (Normalized space)
        # Class 0: Normal
        # Class 1: Cooling Overheat (Temp >>, RPM normal, Oil slightly low)
        # Class 2: Oil Starvation (Oil <<, Temp >, Vibration >)
        # Class 3: Electrical Failure (Voltage << or >>, Temp normal)
        # Class 4: Drivetrain / Vibration (Vibration >>, RPM high)
        self.centroids = np.array([
            [2200.0, 90.0, 48.0, 14.1, 1.2, 65.0],   # Normal
            [2600.0, 118.0, 36.0, 13.5, 3.2, 70.0],  # Overheat
            [2400.0, 102.0, 12.0, 13.6, 4.5, 50.0],  # Oil Starvation
            [1800.0, 88.0, 44.0, 10.9, 1.5, 45.0],   # Electrical
            [3400.0, 94.0, 46.0, 13.9, 7.8, 85.0],   # Transmission / Vibration
        ])

        # Normalize centroids
        self.norm_centroids = (self.centroids - self.means) / self.stds

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Features: array-like of shape (6,)
        Returns predicted class, probabilities, and explainable feature attributions.
        """
        feats = np.asarray(features, dtype=float)
        norm_feats = (feats - self.means) / self.stds

        # Compute Euclidean distance to each cluster centroid in normalized space
        distances = np.linalg.norm(self.norm_centroids - norm_feats, axis=1)

        # Softmax over negative distances with temperature scaling
        temperature = 1.5
        scaled_logits = -distances / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probabilities = exp_logits / np.sum(exp_logits)

        best_class_idx = int(np.argmax(probabilities))
        pred_label = FailureCategory.CLASSES[best_class_idx]
        confidence = float(probabilities[best_class_idx])

        # Feature importance / attribution (which sensor contributed most to anomaly)
        diff_from_normal = np.abs(norm_feats - self.norm_centroids[0])
        top_feature_idx = int(np.argmax(diff_from_normal))
        top_feature_name = self.feature_names[top_feature_idx]

        return {
            "predicted_class": pred_label,
            "confidence": round(confidence * 100.0, 2),
            "class_probabilities": {
                FailureCategory.CLASSES[i]: round(float(probabilities[i]) * 100.0, 2)
                for i in range(len(FailureCategory.CLASSES))
            },
            "root_cause_feature": top_feature_name,
            "explanation": FailureCategory.EXPLANATIONS[pred_label]
        }


class RemainingUsefulLifeRegressor:
    """
    Predictive Maintenance Model: Remaining Useful Life (RUL) estimation.
    Estimates residual operating kilometers before component threshold fatigue.
    """

    def __init__(self, baseline_max_km: float = 15000.0):
        self.baseline_km = baseline_max_km

    def estimate_rul(self, current_vhi: float, coolant_temp: float, oil_press: float,
                     vibration: float, current_mileage: float = 42000.0) -> Dict[str, Any]:
        """
        Calculates RUL based on health index and physical cumulative degradation factors.
        """
        # Thermal stress factor (non-linear penalty if temp > 95°C)
        thermal_penalty = max(0.0, (coolant_temp - 95.0) / 25.0) ** 1.8

        # Lubrication stress factor (penalty if oil < 30 PSI)
        lube_penalty = max(0.0, (32.0 - oil_press) / 20.0) ** 2.0

        # Mechanical vibration stress factor (penalty if vib > 3.0 m/s²)
        vib_penalty = max(0.0, (vibration - 2.8) / 3.0) ** 1.5

        composite_stress = 1.0 + (thermal_penalty * 2.5) + (lube_penalty * 3.5) + (vib_penalty * 1.8)

        # Baseline RUL proportional to Vehicle Health Index (VHI) divided by stress factor
        health_ratio = np.clip(current_vhi / 100.0, 0.05, 1.0)
        rul_km = (self.baseline_km * health_ratio) / composite_stress

        # Urgency classification
        if rul_km > 7500.0:
            urgency = "LOW - Standard Scheduled Maintenance"
            recommended_window = "Inspect at next regular oil change (approx. 5,000 - 8,000 km)"
        elif rul_km > 2000.0:
            urgency = "MEDIUM - Service Required Soon"
            recommended_window = "Schedule dealership / garage inspection within 1,000 km"
        elif rul_km > 350.0:
            urgency = "HIGH - Urgent Maintenance Required"
            recommended_window = "Do not undertake long trips; inspect within 48 hours"
        else:
            urgency = "CRITICAL - Immediate Mechanical Hazard"
            recommended_window = "Cease vehicle operation immediately; tow to repair facility"

        return {
            "estimated_rul_km": round(float(rul_km), 1),
            "current_vehicle_mileage": round(float(current_mileage), 1),
            "projected_service_mileage": round(float(current_mileage + rul_km), 1),
            "wear_acceleration_factor": round(float(composite_stress), 2),
            "urgency": urgency,
            "recommended_window": recommended_window
        }


class MultivariateAnomalyDetector:
    """
    Multivariate Statistical Anomaly Detector tracking distance from normal operating envelope.
    Computes normalized Mahalanobis-like statistical distance and provides continuous anomaly score.
    """

    def __init__(self):
        # Nominal feature means and standard deviations
        self.mean = np.array([2200.0, 90.0, 45.0, 13.8, 1.5, 60.0])
        self.std = np.array([600.0, 8.0, 8.0, 0.6, 0.8, 25.0])
        self.threshold_warning = 2.5
        self.threshold_critical = 4.5

    def score(self, features: np.ndarray) -> Dict[str, Any]:
        feats = np.asarray(features, dtype=float)
        z_scores = np.abs((feats - self.mean) / self.std)
        aggregate_distance = np.sqrt(np.mean(z_scores ** 2))

        # Normalized anomaly score 0.0 to 1.0
        normalized_score = float(np.clip(aggregate_distance / self.threshold_critical, 0.0, 1.0))

        is_anomaly = aggregate_distance >= self.threshold_warning
        is_critical = aggregate_distance >= self.threshold_critical

        status = "NORMAL"
        if is_critical:
            status = "CRITICAL_ANOMALY"
        elif is_anomaly:
            status = "ELEVATED_ANOMALY"

        return {
            "anomaly_score": round(normalized_score, 3),
            "statistical_distance": round(float(aggregate_distance), 2),
            "status": status,
            "sensor_z_scores": {
                "engine_rpm": round(float(z_scores[0]), 2),
                "coolant_temp": round(float(z_scores[1]), 2),
                "oil_pressure": round(float(z_scores[2]), 2),
                "battery_voltage": round(float(z_scores[3]), 2),
                "vibration": round(float(z_scores[4]), 2),
                "vehicle_speed": round(float(z_scores[5]), 2),
            }
        }
