"""
AI-Based Smart Vehicle Health Monitoring System
Capstone Engineering Project Entrypoint (TCS + Soft Computing + AIML)
"""

import sys
import os
import argparse
import json
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def load_local_module(name, rel_path):
    full_path = os.path.join(BASE_DIR, rel_path)
    spec = importlib.util.spec_from_file_location(name, full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Dynamically link modules
tcs = load_local_module("core.tcs_module", "core/tcs_module.py")
sc = load_local_module("core.soft_computing_module", "core/soft_computing_module.py")
aiml = load_local_module("core.aiml_module", "core/aiml_module.py")
pipe_mod = load_local_module("core.pipeline", "core/pipeline.py")
sim_mod = load_local_module("core.telemetry_simulator", "core/telemetry_simulator.py")


def run_demo():
    print("=" * 80)
    print(" AI-BASED SMART VEHICLE HEALTH MONITORING SYSTEM - CAPSTONE DEMO")
    print(" Integrated Theoretical Computer Science, Soft Computing, and AIML Pipeline")
    print("=" * 80)

    pipeline = pipe_mod.VehicleDiagnosticsPipeline()
    simulator = sim_mod.VehicleTelemetrySimulator()

    scenarios = [
        ("NORMAL_CRUISE", 0, "Scenario 1: Nominal Highway Cruise (Clear Highway)"),
        ("COOLANT_LEAK_OVERHEAT", 18, "Scenario 2: Thermal Runaway & Radiator Boil (Coolant Loss)"),
        ("OIL_PUMP_FAILURE", 14, "Scenario 3: Hydraulic Pressure Loss (Oil Starvation Hazard)"),
        ("ALTERNATOR_FAILURE", 22, "Scenario 4: Electrical Decay / Alternator Failure"),
        ("TRANSMISSION_SLIP", 16, "Scenario 5: Drivetrain Slip & Harmonic Vibration Shudder")
    ]

    for name, step, desc in scenarios:
        print(f"\n--- {desc} ---")
        snap = simulator.generate_snapshot(name, step)
        print(f" Raw Telemetry: RPM={snap['engine_rpm']} | Temp={snap['coolant_temp_c']}°C | Oil={snap['oil_pressure_psi']} PSI | Volt={snap['battery_voltage_v']}V | Vib={snap['vibration_ms2']} m/s² | DTC={snap.get('active_dtc') or 'NONE'}")

        diag = pipeline.process_telemetry(
            engine_rpm=snap['engine_rpm'],
            coolant_temp=snap['coolant_temp_c'],
            oil_pressure=snap['oil_pressure_psi'],
            battery_voltage=snap['battery_voltage_v'],
            vibration=snap['vibration_ms2'],
            vehicle_speed=snap['vehicle_speed_kmh'],
            active_dtc=snap.get('active_dtc')
        )

        tcs_res = diag['tcs_layer']
        sc_res = diag['soft_computing_layer']
        ai_res = diag['aiml_layer']

        print(f" [TCS Engine] State: {tcs_res['current_state']} | Trigger Σ: {tcs_res['trigger_symbol']} | Throttle Cap: {tcs_res['moore_outputs']['max_throttle_percent']}% | Lamp: {tcs_res['moore_outputs']['check_engine_light']}")
        print(f" [Soft Computing] Fuzzy Health Index (VHI): {sc_res['vehicle_health_index_pct']}% ({sc_res['health_grade']}) | Fired Rules: {sc_res['active_fuzzy_rules_count']}")
        print(f" [AIML Engine] Diagnosis: {ai_res['diagnostic_classification']} ({ai_res['confidence_pct']}% conf) | Root Cause: {ai_res['root_cause_sensor']}")
        print(f" [Predictive RUL] Est. Remaining Life: {ai_res['predictive_maintenance_rul']['estimated_rul_km']} km | Urgency: {ai_res['predictive_maintenance_rul']['urgency']}")


def run_tests():
    print("Running system validation tests across all 3 subject disciplines...")
    pipeline = pipe_mod.VehicleDiagnosticsPipeline()

    # Test 1: TCS Regex
    res_valid = tcs.DTCRegexValidator.validate_and_decode("P0217")
    assert res_valid.is_valid, "P0217 should be valid"
    assert res_valid.severity == "CRITICAL", "P0217 should be critical"
    res_invalid = tcs.DTCRegexValidator.validate_and_decode("Z9999")
    assert not res_invalid.is_valid, "Z9999 should be invalid"
    print(" [+] TCS Layer: SAE J2012 OBD-II Regex validation PASSED")

    # Test 2: TCS CFG
    cfg_test = tcs.TelemetryCFGParser.parse_frame("CAN:RPM=2400;TEMP=95.0;OIL=45.0#19")
    assert cfg_test["valid"], "CFG parsing should succeed"
    print(" [+] TCS Layer: Telematics Context-Free Grammar parser PASSED")

    # Test 3: Soft Computing Mamdani FIS
    fuzzy_out = pipeline.fuzzy_system.infer(90.0, 48.0, 14.1, 1.2)
    assert fuzzy_out["vehicle_health_index"] > 80.0, "Nominal VHI should be > 80%"
    print(f" [+] Soft Computing Layer: Mamdani FIS inference PASSED (Nominal VHI: {fuzzy_out['vehicle_health_index']}%)")

    # Test 4: AIML Diagnostic Model
    import numpy as np
    normal_vector = np.array([2200.0, 90.0, 48.0, 14.1, 1.2, 65.0])
    ml_out = pipeline.classifier.predict(normal_vector)
    assert ml_out["predicted_class"] == "NORMAL_OPERATION", "Should predict normal"
    print(f" [+] AIML Layer: Centroid/Softmax Failure Classifier PASSED ({ml_out['confidence']}% conf)")

    # Test 5: Predictive Maintenance RUL
    rul_out = pipeline.rul_regressor.estimate_rul(92.0, 90.0, 48.0, 1.2)
    assert rul_out["estimated_rul_km"] > 10000.0, "Healthy vehicle RUL should be high"
    print(f" [+] AIML Layer: Remaining Useful Life Regressor PASSED (Est. RUL: {rul_out['estimated_rul_km']} km)")

    print("\nALL MODULE TESTS PASSED (100% SUCCESS)!")


def export_dataset():
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    out_path = os.path.join(BASE_DIR, "data", "vehicle_telemetry_dataset.csv")
    csv_str = sim_mod.VehicleTelemetrySimulator.generate_dataset_csv(num_samples_per_scenario=150)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(csv_str)
    print(f"Exported 750-row synthetic telemetry dataset to {out_path}")


def start_web_server(port=8080):
    server_mod = load_local_module("web.app_server", "web/app_server.py")
    server_mod.run_server(port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Smart Vehicle Health Monitoring System")
    parser.add_argument("--demo", action="store_true", help="Run terminal simulation demo")
    parser.add_argument("--test", action="store_true", help="Run automated module test suite")
    parser.add_argument("--export-dataset", action="store_true", help="Generate synthetic CSV dataset")
    parser.add_argument("--server", "-s", action="store_true", help="Launch interactive web dashboard")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Web server port (default: 8080)")

    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.export_dataset:
        export_dataset()
    elif args.server:
        start_web_server(args.port)
    else:
        # Default behavior: run demo and suggest server
        run_demo()
        print("\n" + "=" * 80)
        print(f" To launch the interactive visual dashboard, run:")
        print(f"   python3 main.py --server")
        print(f" Then open your browser at: http://localhost:8080")
        print("=" * 80)
