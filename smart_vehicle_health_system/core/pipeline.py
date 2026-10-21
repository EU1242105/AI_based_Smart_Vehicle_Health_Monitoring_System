"""
Smart Vehicle Health Diagnostic Pipeline
Orchestrator combining TCS, Soft Computing, and AIML layers into a unified diagnostic pipeline.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from core.tcs_module import (
    VehicleStateAutomaton, TriggerEvent, VehicleOperationalState,
    DTCRegexValidator, TelemetryCFGParser, MooreMachineOutput
)
from core.soft_computing_module import VehicleHealthFuzzySystem, MultiLayerPerceptron
from core.aiml_module import (
    VehicleHealthClassifier, RemainingUsefulLifeRegressor,
    MultivariateAnomalyDetector, FailureCategory
)


class VehicleDiagnosticsPipeline:
    def __init__(self):
        # 1. Theoretical Computer Science Engine
        self.automaton = VehicleStateAutomaton()
        self.dtc_validator = DTCRegexValidator()
        self.cfg_parser = TelemetryCFGParser()

        # 2. Soft Computing Engine
        self.fuzzy_system = VehicleHealthFuzzySystem()

        # 3. Artificial Intelligence & Machine Learning Engine
        self.classifier = VehicleHealthClassifier()
        self.rul_regressor = RemainingUsefulLifeRegressor()
        self.anomaly_detector = MultivariateAnomalyDetector()

    def process_telemetry(self,
                          engine_rpm: float,
                          coolant_temp: float,
                          oil_pressure: float,
                          battery_voltage: float,
                          vibration: float,
                          vehicle_speed: float,
                          active_dtc: Optional[str] = None,
                          current_odometer: float = 42000.0) -> Dict[str, Any]:
        """
        Executes end-to-end multi-disciplinary diagnostic pipeline for one telemetry snapshot.
        """
        feature_vector = np.array([
            engine_rpm,
            coolant_temp,
            oil_pressure,
            battery_voltage,
            vibration,
            vehicle_speed
        ])

        # -------------------------------------------------------------
        # Step 1: Soft Computing Inference (Fuzzy Logic Layer)
        # -------------------------------------------------------------
        fuzzy_result = self.fuzzy_system.infer(
            temp=coolant_temp,
            oil=oil_pressure,
            volt=battery_voltage,
            vib=vibration
        )
        vhi = fuzzy_result["vehicle_health_index"]

        # -------------------------------------------------------------
        # Step 2: AIML Diagnostic Models
        # -------------------------------------------------------------
        ml_prediction = self.classifier.predict(feature_vector)
        anomaly_result = self.anomaly_detector.score(feature_vector)
        rul_result = self.rul_regressor.estimate_rul(
            current_vhi=vhi,
            coolant_temp=coolant_temp,
            oil_press=oil_pressure,
            vibration=vibration,
            current_mileage=current_odometer
        )

        # -------------------------------------------------------------
        # Step 3: TCS OBD-II DTC Validation
        # -------------------------------------------------------------
        dtc_info = None
        has_critical_dtc = False
        if active_dtc and active_dtc.strip():
            dtc_info = self.dtc_validator.validate_and_decode(active_dtc)
            if dtc_info.is_valid and dtc_info.severity in ["HIGH", "CRITICAL"]:
                has_critical_dtc = True

        # -------------------------------------------------------------
        # Step 4: TCS Finite State Automaton (DFA) Transition
        # -------------------------------------------------------------
        # Map conditions to formal alphabet Σ
        if coolant_temp >= 125.0 or oil_pressure <= 12.0 or has_critical_dtc and dtc_info.severity == "CRITICAL":
            trigger = TriggerEvent.SAFETY_LOCKOUT if (oil_pressure <= 8.0 or coolant_temp >= 135.0) else TriggerEvent.SEVERE_THRESHOLD_BREACH
        elif vhi < 35.0 or anomaly_result["status"] == "CRITICAL_ANOMALY":
            trigger = TriggerEvent.SEVERE_THRESHOLD_BREACH
        elif vhi < 65.0 or anomaly_result["status"] == "ELEVATED_ANOMALY":
            trigger = TriggerEvent.PERSISTENT_MODERATE
        elif vhi < 80.0:
            trigger = TriggerEvent.TRANSIENT_ANOMALY
        else:
            trigger = TriggerEvent.ALL_NOMINAL

        automaton_output = self.automaton.transition(trigger)

        # -------------------------------------------------------------
        # Step 5: Consolidated Response Package
        # -------------------------------------------------------------
        return {
            "telemetry_inputs": {
                "engine_rpm": engine_rpm,
                "coolant_temp_c": coolant_temp,
                "oil_pressure_psi": oil_pressure,
                "battery_voltage_v": battery_voltage,
                "vibration_ms2": vibration,
                "vehicle_speed_kmh": vehicle_speed,
                "active_dtc": active_dtc
            },
            "tcs_layer": {
                "current_state": automaton_output.state.value,
                "trigger_symbol": trigger.value,
                "moore_outputs": {
                    "max_throttle_percent": automaton_output.max_throttle_percent,
                    "check_engine_light": automaton_output.check_engine_light,
                    "buzzer_alert": automaton_output.buzzer_alert,
                    "ecu_fuel_cutoff": automaton_output.ecu_fuel_cutoff,
                    "action_advisory": automaton_output.diagnostic_message
                },
                "dtc_analysis": {
                    "is_valid": dtc_info.is_valid if dtc_info else None,
                    "category": dtc_info.category if dtc_info else None,
                    "subsystem": dtc_info.subsystem if dtc_info else None,
                    "description": dtc_info.description if dtc_info else "No DTC set",
                    "severity": dtc_info.severity if dtc_info else "NONE",
                    "recommended_action": dtc_info.recommended_action if dtc_info else "None"
                } if dtc_info else None,
                "transition_history": self.automaton.transition_history[-5:]
            },
            "soft_computing_layer": {
                "vehicle_health_index_pct": vhi,
                "health_grade": "Pristine" if vhi >= 85 else ("Normal" if vhi >= 70 else ("Degraded" if vhi >= 45 else "Critical Hazard")),
                "fuzzified_inputs": fuzzy_result["fuzzified_inputs"],
                "active_fuzzy_rules_count": len(fuzzy_result["fired_rules"]),
                "fired_rules": fuzzy_result["fired_rules"]
            },
            "aiml_layer": {
                "diagnostic_classification": ml_prediction["predicted_class"],
                "confidence_pct": ml_prediction["confidence"],
                "class_probabilities": ml_prediction["class_probabilities"],
                "root_cause_sensor": ml_prediction["root_cause_feature"],
                "ai_explanation": ml_prediction["explanation"],
                "anomaly_detection": anomaly_result,
                "predictive_maintenance_rul": rul_result
            }
        }
