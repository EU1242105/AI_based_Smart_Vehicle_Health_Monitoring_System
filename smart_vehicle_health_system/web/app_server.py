"""
Smart Vehicle Health Monitoring System - Web Application Server
Unified Driver Assistant & Technical Telemetry Hub
Powered by Python standard library (http.server) with zero external dependencies.
"""

import http.server
import socketserver
import json
import urllib.parse
import sys
import os
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


tcs_mod = load_module('core.tcs_module', os.path.join(PROJECT_DIR, 'core', 'tcs_module.py'))
sc_mod = load_module('core.soft_computing_module', os.path.join(PROJECT_DIR, 'core', 'soft_computing_module.py'))
aiml_mod = load_module('core.aiml_module', os.path.join(PROJECT_DIR, 'core', 'aiml_module.py'))
pipe_mod = load_module('core.pipeline', os.path.join(PROJECT_DIR, 'core', 'pipeline.py'))
sim_mod = load_module('core.telemetry_simulator', os.path.join(PROJECT_DIR, 'core', 'telemetry_simulator.py'))
advisor_mod = load_module('core.driver_advisor', os.path.join(PROJECT_DIR, 'core', 'driver_advisor.py'))
chatbot_mod = load_module('core.diagnostic_chatbot', os.path.join(PROJECT_DIR, 'core', 'diagnostic_chatbot.py'))
wizard_mod = load_module('core.diagnostic_wizard', os.path.join(PROJECT_DIR, 'core', 'diagnostic_wizard.py'))

pipeline = pipe_mod.VehicleDiagnosticsPipeline()
simulator = sim_mod.VehicleTelemetrySimulator()


class VehicleHealthAPIHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _set_headers(self, content_type="application/json", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(status=204)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            html_path = os.path.join(PROJECT_DIR, 'web', 'index.html')
            if os.path.exists(html_path):
                with open(html_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self._set_headers("text/html; charset=utf-8")
                self.wfile.write(content.encode('utf-8'))
            else:
                self._set_headers("text/plain", status=404)
                self.wfile.write(b"index.html not found")
            return

        elif path == "/api/scenario":
            scen_name = query.get("name", ["NORMAL_CRUISE"])[0]
            step = int(query.get("step", ["0"])[0])
            try:
                snap = simulator.generate_snapshot(scen_name, step)
                diag = pipeline.process_telemetry(
                    engine_rpm=snap["engine_rpm"],
                    coolant_temp=snap["coolant_temp_c"],
                    oil_pressure=snap["oil_pressure_psi"],
                    battery_voltage=snap["battery_voltage_v"],
                    vibration=snap["vibration_ms2"],
                    vehicle_speed=snap["vehicle_speed_kmh"],
                    active_dtc=snap.get("active_dtc")
                )
                self._set_headers()
                self.wfile.write(json.dumps({"snapshot": snap, "diagnosis": diag}).encode('utf-8'))
            except Exception as e:
                self._set_headers(status=500)
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            return

        elif path == "/api/reset":
            pipeline.automaton.reset()
            self._set_headers()
            self.wfile.write(json.dumps({"status": "reset_successful"}).encode('utf-8'))
            return

        elif path == "/api/symptoms_list":
            symptoms = [
                {"id": k, "name": v["name"], "category": v["category"], "severity": v["severity"]}
                for k, v in advisor_mod.DriverSymptomTroubleshooter.KNOWLEDGE_BASE.items()
            ]
            self._set_headers()
            self.wfile.write(json.dumps(symptoms).encode('utf-8'))
            return

        elif path == "/api/export_dataset":
            csv_data = simulator.generate_dataset_csv(num_samples_per_scenario=30)
            self._set_headers("text/csv")
            self.send_header("Content-Disposition", "attachment; filename=vehicle_telemetry_dataset.csv")
            self.wfile.write(csv_data.encode('utf-8'))
            return

        else:
            self._set_headers("text/plain", status=404)
            self.wfile.write(b"404 Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')

        if parsed.path == "/api/diagnose":
            try:
                payload = json.loads(post_data)
                diag = pipeline.process_telemetry(
                    engine_rpm=float(payload.get("engine_rpm", 2200.0)),
                    coolant_temp=float(payload.get("coolant_temp", 90.0)),
                    oil_pressure=float(payload.get("oil_pressure", 45.0)),
                    battery_voltage=float(payload.get("battery_voltage", 13.8)),
                    vibration=float(payload.get("vibration", 1.5)),
                    vehicle_speed=float(payload.get("vehicle_speed", 60.0)),
                    active_dtc=payload.get("active_dtc", None)
                )
                self._set_headers()
                self.wfile.write(json.dumps(diag).encode('utf-8'))
            except Exception as e:
                self._set_headers(status=400)
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

        elif parsed.path == "/api/wizard_step1":
            try:
                payload = json.loads(post_data)
                res = wizard_mod.DiagnosticWizardEngine.identify_preliminary_issue(payload)
                self._set_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))
            except Exception as e:
                self._set_headers(status=400)
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

        elif parsed.path == "/api/wizard_step4":
            try:
                payload = json.loads(post_data)
                issue_key = payload.get("issue_key", "BATTERY_ELECTRICAL")
                answers = payload.get("answers", {})
                vehicle_info = payload.get("vehicle_info", {})
                res = wizard_mod.DiagnosticWizardEngine.evaluate_final_decision(issue_key, answers, vehicle_info)
                self._set_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))
            except Exception as e:
                self._set_headers(status=400)
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

        elif parsed.path == "/api/chat":
            try:
                payload = json.loads(post_data)
                user_msg = payload.get("message", "")
                chat_res = chatbot_mod.DiagnosticChatbotEngine.process_query(user_msg)
                self._set_headers()
                self.wfile.write(json.dumps(chat_res).encode('utf-8'))
            except Exception as e:
                self._set_headers(status=400)
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

        elif parsed.path == "/api/diagnose_symptom":
            try:
                payload = json.loads(post_data)
                sym_key = payload.get("symptom_key", "SOUND_BRAKE_SQUEAL")
                diagnosis = advisor_mod.DriverSymptomTroubleshooter.diagnose(sym_key)
                self._set_headers()
                self.wfile.write(json.dumps(diagnosis).encode('utf-8'))
            except Exception as e:
                self._set_headers(status=400)
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

        elif parsed.path == "/api/service_status":
            try:
                payload = json.loads(post_data)
                curr_odo = float(payload.get("current_odometer", 42000.0))
                last_odo = float(payload.get("last_service_odometer", 35000.0))
                months_elapsed = float(payload.get("months_since_service", 7.0))
                parts_replaced = payload.get("parts_replaced", ["engine_oil", "air_filter"])
                symptom_severity = float(payload.get("symptom_severity_weight", 0.0))
                has_abnormal_sound = bool(payload.get("has_sound", False))

                km_driven = max(0.0, curr_odo - last_odo)

                # 1. TCS Service Lifecycle State Machine
                lifecycle_state = advisor_mod.ServiceLifecycleAutomaton.evaluate_state(
                    km_since_service=km_driven,
                    months_since_service=months_elapsed,
                    has_abnormal_sound=has_abnormal_sound
                )

                # 2. Soft Computing Fuzzy Maintenance Urgency Index
                fuzzy_urgency = advisor_mod.FuzzyServiceAdvisor.infer_urgency(
                    km_elapsed=km_driven,
                    months_elapsed=months_elapsed,
                    symptom_severity=symptom_severity
                )

                # 3. Component Wear Catalog
                wear_catalog = advisor_mod.ComponentWearCatalog.calculate_wear(
                    current_odometer=curr_odo,
                    last_service_odometer=last_odo,
                    months_elapsed=months_elapsed,
                    parts_replaced=parts_replaced
                )

                # 4. Service Recommendation
                km_to_next = max(0.0, 10000.0 - km_driven)
                days_to_next = max(0, int((12.0 - months_elapsed) * 30))

                response_data = {
                    "current_odometer": curr_odo,
                    "last_service_odometer": last_odo,
                    "km_driven_since_service": km_driven,
                    "months_since_service": months_elapsed,
                    "km_to_next_service": round(km_to_next, 0),
                    "estimated_days_to_service": days_to_next,
                    "tcs_lifecycle_automaton": lifecycle_state,
                    "soft_computing_fuzzy_urgency": fuzzy_urgency,
                    "component_wear_status": wear_catalog["parts_wear"]
                }
                self._set_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            except Exception as e:
                self._set_headers(status=400)
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

        elif parsed.path == "/api/validate_vin":
            try:
                payload = json.loads(post_data)
                vin = payload.get("vin", "")
                res = advisor_mod.VINValidator.validate(vin)
                self._set_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))
            except Exception as e:
                self._set_headers(status=400)
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

        elif parsed.path == "/api/validate_dtc":
            try:
                payload = json.loads(post_data)
                code = payload.get("code", "")
                res = tcs_mod.DTCRegexValidator.validate_and_decode(code)
                self._set_headers()
                self.wfile.write(json.dumps({
                    "is_valid": res.is_valid,
                    "raw_code": res.raw_code,
                    "category": res.category,
                    "standard": res.standard,
                    "subsystem": res.subsystem,
                    "fault_code": res.fault_code,
                    "description": res.description,
                    "recommended_action": res.recommended_action,
                    "severity": res.severity
                }).encode('utf-8'))
            except Exception as e:
                self._set_headers(status=400)
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

        else:
            self._set_headers("text/plain", status=404)
            self.wfile.write(b"404 Not Found")


def run_server(port=8080):
    handler = VehicleHealthAPIHandler
    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        print(f"[*] Smart Vehicle Health Monitoring System Dashboard running at http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("[!] Shutting down server.")
            httpd.server_close()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
