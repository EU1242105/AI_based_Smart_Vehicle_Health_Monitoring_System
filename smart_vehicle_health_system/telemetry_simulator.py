"""
Vehicle Telemetry Simulator
Generates realistic dynamic sensor streams and pre-programmed fault injection scenarios.
"""

import random
from typing import Dict, Any


class VehicleTelemetrySimulator:
    SCENARIOS = {
        "NORMAL_CRUISE": "Nominal highway cruise with healthy thermal, hydraulic, and electrical parameters.",
        "COOLANT_LEAK_OVERHEAT": "Gradual loss of radiator coolant resulting in thermal runaway and boiling.",
        "OIL_PUMP_FAILURE": "Sudden hydraulic pressure collapse in the oil galleried lubrication circuit.",
        "ALTERNATOR_FAILURE": "Electrical charging failure causing progressive battery drainage down to ECU brownout.",
        "TRANSMISSION_SLIP": "Severe mechanical shudder and harmonic drivetrain vibration at highway speeds."
    }

    @staticmethod
    def generate_snapshot(scenario: str = "NORMAL_CRUISE", step: int = 0) -> Dict[str, Any]:
        noise = lambda scale: random.gauss(0, scale)

        if scenario == "NORMAL_CRUISE":
            rpm = 2200 + noise(60)
            speed = 65 + noise(2.0)
            temp = 90.0 + noise(1.0)
            oil = 48.0 + noise(1.5)
            volt = 14.1 + noise(0.1)
            vib = 1.2 + abs(noise(0.2))
            dtc = ""

        elif scenario == "COOLANT_LEAK_OVERHEAT":
            progress = min(1.0, step / 20.0)
            rpm = 2400 + noise(80)
            speed = max(20.0, 70.0 - progress * 40.0)
            temp = 90.0 + progress * 38.0 + noise(1.2)
            oil = max(25.0, 48.0 - progress * 16.0 + noise(1.5))
            volt = 13.9 + noise(0.15)
            vib = 1.4 + progress * 2.2 + abs(noise(0.3))
            dtc = "P0217" if progress > 0.6 else ("P0118" if progress > 0.3 else "")

        elif scenario == "OIL_PUMP_FAILURE":
            progress = min(1.0, step / 15.0)
            rpm = 2500 + noise(90)
            speed = max(10.0, 60.0 - progress * 50.0)
            temp = 92.0 + progress * 24.0 + noise(1.5)
            oil = max(6.0, 48.0 - progress * 41.0 + noise(0.8))
            volt = 13.8 + noise(0.2)
            vib = 1.3 + progress * 4.5 + abs(noise(0.4))
            dtc = "P0524" if progress > 0.4 else ""

        elif scenario == "ALTERNATOR_FAILURE":
            progress = min(1.0, step / 25.0)
            rpm = 2100 + noise(50)
            speed = 55 + noise(3.0)
            temp = 89.0 + noise(1.0)
            oil = 45.0 + noise(1.2)
            volt = max(9.8, 14.1 - progress * 4.2 + noise(0.1))
            vib = 1.2 + abs(noise(0.15))
            dtc = "P0562" if progress > 0.5 else ""

        elif scenario == "TRANSMISSION_SLIP":
            progress = min(1.0, step / 18.0)
            rpm = 2200 + progress * 1600 + noise(150)
            speed = max(35.0, 75.0 - progress * 30.0 + noise(2.0))
            temp = 91.0 + progress * 14.0 + noise(1.2)
            oil = 44.0 + noise(1.5)
            volt = 13.9 + noise(0.1)
            vib = 1.4 + progress * 6.8 + abs(noise(0.6))
            dtc = "P0700" if progress > 0.4 else ""

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        return {
            "step": step,
            "scenario": scenario,
            "engine_rpm": round(float(rpm), 1),
            "vehicle_speed_kmh": round(float(speed), 1),
            "coolant_temp_c": round(float(temp), 1),
            "oil_pressure_psi": round(float(oil), 1),
            "battery_voltage_v": round(float(volt), 2),
            "vibration_ms2": round(float(vib), 2),
            "active_dtc": dtc
        }

    @classmethod
    def generate_dataset_csv(cls, num_samples_per_scenario: int = 100) -> str:
        import pandas as pd
        records = []
        for scen in cls.SCENARIOS.keys():
            for s in range(num_samples_per_scenario):
                row = cls.generate_snapshot(scen, step=s % 30)
                records.append(row)
        df = pd.DataFrame(records)
        return df.to_csv(index=False)
