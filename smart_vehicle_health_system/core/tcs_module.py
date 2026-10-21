"""
Theoretical Computer Science (TCS) Module
Course Code Alignment: 24CSPCC501C (SJCEM Computer Engineering)
Components:
1. Module I & II: Regular Expressions & Finite Automata for SAE J2012 OBD-II Diagnostic Trouble Codes (DTC)
2. Module III & IV: Context-Free Grammar (CFG) & Parsing of CAN-Bus Telemetry Frames
3. Module I: Deterministic Finite Automaton (DFA) / Moore Machine for Vehicle Health Operational Modes
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


# ==============================================================================
# 1. REGULAR EXPRESSIONS & FINITE AUTOMATA FOR DTC VALIDATION (Modules I & II)
# ==============================================================================

class DTCCategory(Enum):
    POWERTRAIN = "P"  # Engine, transmission, emissions
    CHASSIS = "C"     # Suspension, steering, ABS, stability control
    BODY = "B"        # Airbags, lighting, climate control, instrumentation
    NETWORK = "U"     # CAN-bus, wiring harnesses, ECU communication


class DTCStandard(Enum):
    SAE_STANDARDIZED = "0"
    MANUFACTURER_SPECIFIC = "1"
    SAE_OR_MANUFACTURER = "2"
    RESERVED = "3"


@dataclass
class DTCDecodeResult:
    is_valid: bool
    raw_code: str
    category: Optional[str] = None
    standard: Optional[str] = None
    subsystem: Optional[str] = None
    fault_code: Optional[str] = None
    description: Optional[str] = None
    recommended_action: Optional[str] = None
    severity: str = "UNKNOWN"
    validation_regex: str = r"^[PCBU][0-3][0-9A-F]{3}$"


class DTCRegexValidator:
    """
    Formal regular expression validator and lexical decoder for SAE J2012 OBD-II DTCs.
    Formal Regular Expression:
        L(DTC) = (P | C | B | U) . (0 | 1 | 2 | 3) . [0-9A-F] . [0-9A-F] . [0-9A-F]
    """
    DTC_REGEX = re.compile(r"^([PCBU])([0-3])([0-9A-F])([0-9A-F]{2})$", re.IGNORECASE)

    SUBSYSTEM_MAP = {
        "P": {
            "0": "Fuel and Air Metering and Auxiliary Emission Controls",
            "1": "Fuel and Air Metering",
            "2": "Fuel and Air Metering (Injector Circuit)",
            "3": "Ignition System or Misfire",
            "4": "Auxiliary Emission Controls",
            "5": "Vehicle Speed, Idle Control, and Auxiliary Inputs",
            "6": "Computer and Auxiliary Outputs",
            "7": "Transmission Control System",
            "8": "Transmission Control System",
        },
        "C": {
            "0": "Brake System & ABS",
            "1": "Traction Control & ESP",
            "2": "Steering Control",
            "3": "Suspension System",
        },
        "B": {
            "0": "Lighting and Signaling",
            "1": "Airbag & Restraint System",
            "2": "Door Locks and Windows",
            "3": "Climate Control / HVAC",
        },
        "U": {
            "0": "Network Communication (Standard)",
            "1": "Manufacturer Network Bus",
            "2": "Software & Firmware Handshake",
            "3": "Network Hardware Bus Failure",
        }
    }

    # Standard Common Fault Dictionary
    COMMON_DTC_DB = {
        "P0115": ("Engine Coolant Temperature Circuit Malfunction", "Inspect coolant temp sensor and wiring harness", "HIGH"),
        "P0118": ("Engine Coolant Temperature High Input / Overheating Hazard", "Immediate stop, check coolant level and thermostat", "CRITICAL"),
        "P0217": ("Engine Coolant Over-Temperature Condition", "Stop vehicle immediately; risk of cylinder head warp", "CRITICAL"),
        "P0300": ("Random/Multiple Cylinder Misfire Detected", "Check spark plugs, ignition coils, and fuel injectors", "HIGH"),
        "P0524": ("Engine Oil Pressure Too Low", "Shut down engine immediately to prevent engine seizure", "CRITICAL"),
        "P0562": ("System Voltage Low (Alternator / Battery Failure)", "Check alternator charging voltage and battery terminal corrosion", "MEDIUM"),
        "P0563": ("System Voltage High (Overcharging)", "Inspect voltage regulator; risk of damaging electronic modules", "HIGH"),
        "P0700": ("Transmission Control System Malfunction", "Scan transmission ECU; enter limp mode for safety", "HIGH"),
        "C0035": ("Left Front Wheel Speed Sensor Malfunction", "Inspect ABS sensor ring and wheel speed wiring", "MEDIUM"),
        "C0242": ("PCM Indicating Throttle Reduction Needed", "Check traction control system and throttle position sensor", "HIGH"),
        "B0001": ("Driver Airbag Squib Circuit Open", "Immediate safety check of steering wheel clock spring", "HIGH"),
        "U0100": ("Lost Communication With ECM / PCM 'A'", "Inspect high-speed CAN bus terminating resistors and harness", "CRITICAL"),
        "U0121": ("Lost Communication With Anti-Lock Brake System (ABS) Control Module", "Check ABS CAN gateway and fuse box", "HIGH"),
    }

    @classmethod
    def validate_and_decode(cls, code: str) -> DTCDecodeResult:
        code_clean = code.strip().upper()
        match = cls.DTC_REGEX.match(code_clean)
        if not match:
            return DTCDecodeResult(
                is_valid=False,
                raw_code=code,
                severity="INVALID"
            )

        cat_char, std_char, sub_char, fault_chars = match.groups()

        categories = {
            "P": "Powertrain (Engine, Transmission)",
            "C": "Chassis (Braking, Steering, Suspension)",
            "B": "Body (Safety, Airbags, Comfort)",
            "U": "Network (CAN-Bus, Communication)"
        }
        standards = {
            "0": "Standardized (SAE/ISO)",
            "1": "Manufacturer Specific",
            "2": "Standardized / Extended",
            "3": "Reserved / Special"
        }

        sub_desc = cls.SUBSYSTEM_MAP.get(cat_char, {}).get(sub_char, "General Subsystem Fault")
        known = cls.COMMON_DTC_DB.get(code_clean)

        if known:
            desc, action, severity = known
        else:
            desc = f"{categories[cat_char]} - {sub_desc} (Code #{fault_chars})"
            action = "Perform targeted electronic diagnostics via scan tool"
            severity = "MEDIUM" if cat_char in ["B", "C"] else "HIGH"

        return DTCDecodeResult(
            is_valid=True,
            raw_code=code_clean,
            category=categories[cat_char],
            standard=standards[std_char],
            subsystem=sub_desc,
            fault_code=fault_chars,
            description=desc,
            recommended_action=action,
            severity=severity
        )


# ==============================================================================
# 2. CONTEXT-FREE GRAMMAR (CFG) & TELEMATICS FRAME PARSER (Modules III & IV)
# ==============================================================================

class TelemetryCFGParser:
    """
    Context Free Grammar (CFG) parser for serialized vehicle telemetry frames.
    Grammar Specification G = (V, Σ, R, S):
        S          -> HEADER ':' PAYLOAD '#' CHECKSUM
        HEADER     -> 'CAN' | 'OBD2' | 'TEL'
        PAYLOAD    -> KV_PAIR (';' KV_PAIR)*
        KV_PAIR    -> KEY '=' VALUE
        KEY        -> [A-Z_]+
        VALUE      -> [0-9]+ ('.' [0-9]+)?
        CHECKSUM   -> [0-9A-F]{2,4}
    """
    FRAME_PATTERN = re.compile(r"^(CAN|OBD2|TEL):([A-Z_]+=[0-9]+(?:\.[0-9]+)?(?:;[A-Z_]+=[0-9]+(?:\.[0-9]+)?)*)#([0-9A-F]{2,4})$")

    @classmethod
    def parse_frame(cls, frame_str: str) -> Dict[str, Any]:
        frame_str = frame_str.strip()
        match = cls.FRAME_PATTERN.match(frame_str)
        if not match:
            return {
                "valid": False,
                "error": "Syntax Error: Frame violates Context-Free Grammar derivation S -> HEADER ':' PAYLOAD '#' CHECKSUM",
                "raw": frame_str
            }

        header, payload_str, checksum = match.groups()
        
        # Verify 8-bit XOR checksum
        computed_checksum = 0
        raw_to_check = f"{header}:{payload_str}"
        for char in raw_to_check:
            computed_checksum ^= ord(char)
        computed_hex = f"{computed_checksum:02X}"

        # Parse Key-Value pairs
        telemetry_dict = {}
        for pair in payload_str.split(";"):
            k, v = pair.split("=")
            telemetry_dict[k] = float(v)

        return {
            "valid": True,
            "header": header,
            "telemetry": telemetry_dict,
            "received_checksum": checksum,
            "computed_checksum": computed_hex,
            "checksum_verified": (checksum.upper() == computed_hex.upper()),
            "parse_tree": {
                "Root (S)": {
                    "HEADER": header,
                    "PAYLOAD_COUNT": len(telemetry_dict),
                    "CHECKSUM": checksum
                }
            }
        }


# ==============================================================================
# 3. DETERMINISTIC FINITE AUTOMATON (DFA) / MOORE MACHINE (Module I)
# ==============================================================================

class VehicleOperationalState(Enum):
    S0_NORMAL = "S0_NORMAL"                   # All systems nominal
    S1_ADVISORY = "S1_ADVISORY"               # Minor transient deviation
    S2_LIMP_MODE = "S2_LIMP_MODE"             # Derated engine power, protective operation
    S3_CRITICAL_FAULT = "S3_CRITICAL_FAULT"   # Severe hazard: stop imminent
    S4_EMERGENCY_HALT = "S4_EMERGENCY_HALT"   # Engine fuel cutoff / safe shutdown


class TriggerEvent(Enum):
    ALL_NOMINAL = "ALL_NOMINAL"
    TRANSIENT_ANOMALY = "TRANSIENT_ANOMALY"
    PERSISTENT_MODERATE = "PERSISTENT_MODERATE"
    SEVERE_THRESHOLD_BREACH = "SEVERE_THRESHOLD_BREACH"
    MANUAL_OVERRIDE_RESET = "MANUAL_OVERRIDE_RESET"
    SAFETY_LOCKOUT = "SAFETY_LOCKOUT"


@dataclass
class MooreMachineOutput:
    state: VehicleOperationalState
    max_throttle_percent: int
    check_engine_light: str      # OFF, STEADY_AMBER, FLASHING_AMBER, FLASHING_RED
    buzzer_alert: str            # OFF, INTERMITTENT, CONTINUOUS
    ecu_fuel_cutoff: bool
    diagnostic_message: str


class VehicleStateAutomaton:
    """
    Deterministic Finite Automaton (DFA) / Moore Machine M = (Q, Σ, δ, q0, F, Λ, λ)
    States Q: {S0_NORMAL, S1_ADVISORY, S2_LIMP_MODE, S3_CRITICAL_FAULT, S4_EMERGENCY_HALT}
    Input Alphabet Σ: {ALL_NOMINAL, TRANSIENT_ANOMALY, PERSISTENT_MODERATE, SEVERE_THRESHOLD_BREACH, MANUAL_OVERRIDE_RESET, SAFETY_LOCKOUT}
    Initial State q0: S0_NORMAL
    Output Function λ(q): Determines engine governor, warning lamps, and ECU actuation.
    """

    # Transition Function δ: Q × Σ -> Q
    TRANSITION_TABLE: Dict[Tuple[VehicleOperationalState, TriggerEvent], VehicleOperationalState] = {
        # From S0_NORMAL
        (VehicleOperationalState.S0_NORMAL, TriggerEvent.ALL_NOMINAL): VehicleOperationalState.S0_NORMAL,
        (VehicleOperationalState.S0_NORMAL, TriggerEvent.TRANSIENT_ANOMALY): VehicleOperationalState.S1_ADVISORY,
        (VehicleOperationalState.S0_NORMAL, TriggerEvent.PERSISTENT_MODERATE): VehicleOperationalState.S2_LIMP_MODE,
        (VehicleOperationalState.S0_NORMAL, TriggerEvent.SEVERE_THRESHOLD_BREACH): VehicleOperationalState.S3_CRITICAL_FAULT,
        (VehicleOperationalState.S0_NORMAL, TriggerEvent.SAFETY_LOCKOUT): VehicleOperationalState.S4_EMERGENCY_HALT,

        # From S1_ADVISORY
        (VehicleOperationalState.S1_ADVISORY, TriggerEvent.ALL_NOMINAL): VehicleOperationalState.S0_NORMAL,
        (VehicleOperationalState.S1_ADVISORY, TriggerEvent.TRANSIENT_ANOMALY): VehicleOperationalState.S1_ADVISORY,
        (VehicleOperationalState.S1_ADVISORY, TriggerEvent.PERSISTENT_MODERATE): VehicleOperationalState.S2_LIMP_MODE,
        (VehicleOperationalState.S1_ADVISORY, TriggerEvent.SEVERE_THRESHOLD_BREACH): VehicleOperationalState.S3_CRITICAL_FAULT,
        (VehicleOperationalState.S1_ADVISORY, TriggerEvent.SAFETY_LOCKOUT): VehicleOperationalState.S4_EMERGENCY_HALT,

        # From S2_LIMP_MODE
        (VehicleOperationalState.S2_LIMP_MODE, TriggerEvent.ALL_NOMINAL): VehicleOperationalState.S1_ADVISORY,
        (VehicleOperationalState.S2_LIMP_MODE, TriggerEvent.TRANSIENT_ANOMALY): VehicleOperationalState.S2_LIMP_MODE,
        (VehicleOperationalState.S2_LIMP_MODE, TriggerEvent.PERSISTENT_MODERATE): VehicleOperationalState.S2_LIMP_MODE,
        (VehicleOperationalState.S2_LIMP_MODE, TriggerEvent.SEVERE_THRESHOLD_BREACH): VehicleOperationalState.S3_CRITICAL_FAULT,
        (VehicleOperationalState.S2_LIMP_MODE, TriggerEvent.SAFETY_LOCKOUT): VehicleOperationalState.S4_EMERGENCY_HALT,
        (VehicleOperationalState.S2_LIMP_MODE, TriggerEvent.MANUAL_OVERRIDE_RESET): VehicleOperationalState.S0_NORMAL,

        # From S3_CRITICAL_FAULT
        (VehicleOperationalState.S3_CRITICAL_FAULT, TriggerEvent.ALL_NOMINAL): VehicleOperationalState.S2_LIMP_MODE,
        (VehicleOperationalState.S3_CRITICAL_FAULT, TriggerEvent.SEVERE_THRESHOLD_BREACH): VehicleOperationalState.S3_CRITICAL_FAULT,
        (VehicleOperationalState.S3_CRITICAL_FAULT, TriggerEvent.SAFETY_LOCKOUT): VehicleOperationalState.S4_EMERGENCY_HALT,
        (VehicleOperationalState.S3_CRITICAL_FAULT, TriggerEvent.MANUAL_OVERRIDE_RESET): VehicleOperationalState.S2_LIMP_MODE,

        # From S4_EMERGENCY_HALT
        (VehicleOperationalState.S4_EMERGENCY_HALT, TriggerEvent.ALL_NOMINAL): VehicleOperationalState.S4_EMERGENCY_HALT, # Lockout persists
        (VehicleOperationalState.S4_EMERGENCY_HALT, TriggerEvent.MANUAL_OVERRIDE_RESET): VehicleOperationalState.S0_NORMAL,
    }

    # Moore Machine Output Mapping λ(q)
    OUTPUT_MAP: Dict[VehicleOperationalState, MooreMachineOutput] = {
        VehicleOperationalState.S0_NORMAL: MooreMachineOutput(
            state=VehicleOperationalState.S0_NORMAL,
            max_throttle_percent=100,
            check_engine_light="OFF",
            buzzer_alert="OFF",
            ecu_fuel_cutoff=False,
            diagnostic_message="System Nominal: Full powertrain authority available."
        ),
        VehicleOperationalState.S1_ADVISORY: MooreMachineOutput(
            state=VehicleOperationalState.S1_ADVISORY,
            max_throttle_percent=100,
            check_engine_light="STEADY_AMBER",
            buzzer_alert="OFF",
            ecu_fuel_cutoff=False,
            diagnostic_message="Advisory: Transient anomaly logged in volatile ECU memory."
        ),
        VehicleOperationalState.S2_LIMP_MODE: MooreMachineOutput(
            state=VehicleOperationalState.S2_LIMP_MODE,
            max_throttle_percent=45,
            check_engine_light="FLASHING_AMBER",
            buzzer_alert="INTERMITTENT",
            ecu_fuel_cutoff=False,
            diagnostic_message="Degraded Limp-Home Mode: Throttle limited to 45% to prevent thermal runaway."
        ),
        VehicleOperationalState.S3_CRITICAL_FAULT: MooreMachineOutput(
            state=VehicleOperationalState.S3_CRITICAL_FAULT,
            max_throttle_percent=15,
            check_engine_light="FLASHING_RED",
            buzzer_alert="CONTINUOUS",
            ecu_fuel_cutoff=False,
            diagnostic_message="Critical Hazard: Severe component strain. Pull over safely immediately!"
        ),
        VehicleOperationalState.S4_EMERGENCY_HALT: MooreMachineOutput(
            state=VehicleOperationalState.S4_EMERGENCY_HALT,
            max_throttle_percent=0,
            check_engine_light="SOLID_RED",
            buzzer_alert="CONTINUOUS",
            ecu_fuel_cutoff=True,
            diagnostic_message="Emergency Lockout: Fuel injection cutoff active. Engine disabled for mechanical safety."
        ),
    }

    def __init__(self):
        self.current_state = VehicleOperationalState.S0_NORMAL
        self.transition_history: List[Dict[str, Any]] = []

    def transition(self, event: TriggerEvent) -> MooreMachineOutput:
        key = (self.current_state, event)
        next_state = self.TRANSITION_TABLE.get(key, self.current_state)
        
        # Log formal transition step
        self.transition_history.append({
            "from_state": self.current_state.value,
            "trigger_symbol": event.value,
            "to_state": next_state.value
        })
        self.current_state = next_state
        return self.OUTPUT_MAP[self.current_state]

    def get_current_output(self) -> MooreMachineOutput:
        return self.OUTPUT_MAP[self.current_state]

    def reset(self):
        self.current_state = VehicleOperationalState.S0_NORMAL
        self.transition_history.clear()
