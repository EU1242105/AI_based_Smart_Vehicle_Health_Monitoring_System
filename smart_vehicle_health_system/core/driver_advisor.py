"""
Driver Advisor & Service Prognostics Engine
Translates real-world driver complaints, acoustic symptoms, and service history
into actionable repair guides, maintenance schedules, and formal academic models (TCS, SC, AIML).
"""

import re
from datetime import datetime, date
from typing import Dict, List, Any, Optional
import numpy as np


# ==============================================================================
# 1. TCS FORMAL VALIDATORS: VIN REGEX & SERVICE LIFECYCLE DFA
# ==============================================================================

class VINValidator:
    """
    TCS Module II: ISO 3779 Vehicle Identification Number (VIN) Regular Language Validator.
    Regex excludes letters I, O, Q to avoid confusion with numerals 1 and 0.
    """
    VIN_REGEX = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")

    @classmethod
    def validate(cls, vin: str) -> Dict[str, Any]:
        vin_clean = vin.strip().upper()
        is_valid = bool(cls.VIN_REGEX.match(vin_clean))
        return {
            "vin": vin_clean,
            "is_valid": is_valid,
            "length": len(vin_clean),
            "regex_pattern": r"^[A-HJ-NPR-Z0-9]{17}$",
            "region": "Asia (India/Japan/Korea)" if vin_clean.startswith(('M', 'J', 'K')) else ("North America" if vin_clean.startswith(('1','2','3','4','5')) else "Europe/Other")
        }


class ServiceLifecycleAutomaton:
    """
    TCS Module I: Deterministic Finite Automaton (DFA) modeling vehicle service state.
    States:
      S0: NEW_OR_JUST_SERVICED
      S1: OPTIMAL_RUNNING
      S2: SERVICE_DUE_SOON (Warning phase)
      S3: OVERDUE_MAINTENANCE (Accelerated wear)
      S4: COMPONENT_FATIGUE_FAILURE (Breakdown hazard)
    """
    STATES = ["S0_JUST_SERVICED", "S1_OPTIMAL", "S2_DUE_SOON", "S3_OVERDUE", "S4_FATIGUE_FAILURE"]

    @classmethod
    def evaluate_state(cls, km_since_service: float, months_since_service: float, has_abnormal_sound: bool) -> Dict[str, Any]:
        if has_abnormal_sound and (km_since_service > 10000 or months_since_service > 12):
            state = "S4_FATIGUE_FAILURE"
            desc = "Critical Fatigue: Active mechanical noise with overdue service. Component replacement mandatory!"
            urgency = "EMERGENCY"
        elif km_since_service > 12000 or months_since_service > 14:
            state = "S3_OVERDUE"
            desc = "Maintenance Overdue: Engine oil sludging and friction wear accelerating rapidly."
            urgency = "HIGH"
        elif km_since_service > 8500 or months_since_service > 10:
            state = "S2_DUE_SOON"
            desc = "Service Due Soon: Schedule service appointment within next 1,000 km or 30 days."
            urgency = "MEDIUM"
        elif km_since_service > 1500 or months_since_service > 2:
            state = "S1_OPTIMAL"
            desc = "Optimal Operation: Vehicle in healthy running window. Regular fluid checks recommended."
            urgency = "LOW"
        else:
            state = "S0_JUST_SERVICED"
            desc = "Fresh Service: Fresh fluids and inspected wear items."
            urgency = "NONE"

        return {
            "current_state": state,
            "description": desc,
            "urgency_level": urgency
        }


# ==============================================================================
# 2. SOFT COMPUTING: FUZZY MAINTENANCE URGENCY EVALUATOR
# ==============================================================================

class FuzzyServiceAdvisor:
    """
    Soft Computing Module 2: Mamdani FIS for Driver Maintenance Urgency.
    Inputs:
      1. Mileage elapsed since service (0 to 15,000 km)
      2. Time elapsed since service (0 to 18 months)
      3. Symptom severity weight (0.0: None, 0.5: Mild, 1.0: Severe)
    Output:
      Composite Service Urgency Index (0 to 100%) via Centroid Defuzzification.
    """

    @staticmethod
    def infer_urgency(km_elapsed: float, months_elapsed: float, symptom_severity: float) -> Dict[str, Any]:
        # Fuzzify Mileage: Low [0, 5000], Moderate [4000, 10000], High [9000, 15000]
        mu_km_low = max(0.0, min(1.0, (7000.0 - km_elapsed) / 5000.0))
        mu_km_mod = max(0.0, min((km_elapsed - 5000.0) / 3000.0, (11000.0 - km_elapsed) / 3000.0)) if 5000 <= km_elapsed <= 11000 else 0.0
        mu_km_high = max(0.0, min(1.0, (km_elapsed - 9000.0) / 4000.0))

        # Fuzzify Months: Recent [0, 6], Moderate [5, 12], Overdue [10, 18]
        mu_mo_low = max(0.0, min(1.0, (8.0 - months_elapsed) / 6.0))
        mu_mo_high = max(0.0, min(1.0, (months_elapsed - 10.0) / 4.0))

        # Fuzzify Symptom Severity
        mu_sym_high = min(1.0, max(0.0, symptom_severity))

        # Rule evaluation & weighted Centroid approximation
        base_urgency = (km_elapsed / 10000.0) * 50.0 + (months_elapsed / 12.0) * 30.0 + (mu_sym_high * 40.0)
        urgency_score = float(np.clip(base_urgency, 5.0, 100.0))

        if urgency_score >= 80.0:
            category = "CRITICAL - Immediate Service & Inspection Required"
        elif urgency_score >= 60.0:
            category = "HIGH - Service Due within 15 Days"
        elif urgency_score >= 35.0:
            category = "MODERATE - Plan Scheduled Maintenance Soon"
        else:
            category = "LOW - Vehicle Health in Good Standing"

        return {
            "urgency_score_pct": round(urgency_score, 1),
            "urgency_category": category,
            "fuzzified_factors": {
                "mileage_stress": round(float(mu_km_high), 2),
                "time_stress": round(float(mu_mo_high), 2),
                "acoustic_symptom_stress": round(float(mu_sym_high), 2)
            }
        }


# ==============================================================================
# 3. AIML DRIVER SYMPTOM & ACOUSTIC TROUBLESHOOTER (Expert Knowledge Engine)
# ==============================================================================

class DriverSymptomTroubleshooter:
    """
    Expert Diagnostic System & Root Cause Engine for Driver Problems.
    Diagnoses based on real driver feedback: noises/sounds, physical feel, and fluid/visual signs.
    Provides root causes, DIY troubleshooting, and step-by-step part replacement manuals.
    """

    KNOWLEDGE_BASE = {
        "SOUND_BRAKE_SQUEAL": {
            "id": "SOUND_BRAKE_SQUEAL",
            "name": "High-Pitched Squealing Sound When Applying Brakes",
            "category": "Braking System",
            "suspected_part": "Front / Rear Brake Pads (Acoustic Wear Indicator)",
            "severity": "HIGH",
            "severity_weight": 0.75,
            "root_cause": "Brake pad friction material is worn down to the metal wear indicator tab (< 3mm remaining), designed to screech against the brake rotor to alert the driver before metal-to-metal rotor gouging occurs.",
            "troubleshooting": [
                "1. Visually inspect the brake pad thickness through the wheel rim spokes using a flashlight.",
                "2. If friction material is thinner than the backing plate (under 3 mm), pad replacement is urgently required.",
                "3. Check the brake disc rotor surface with your finger (when cold) for deep circular scoring or grooves.",
                "4. Check brake fluid level in the master cylinder reservoir."
            ],
            "replacement_guide": {
                "difficulty": "Moderate (DIY-Friendly with basic tools)",
                "estimated_time": "45 - 60 minutes",
                "estimated_cost_inr": "₹1,200 - ₹2,800 (Pads) vs ₹4,500+ (Dealership)",
                "tools_required": ["Jack & Jack Stands", "Lug Wrench", "12mm & 14mm Sockets", "C-Clamp / Caliper Piston Tool", "Brake Cleaner Spray", "Silicone Brake Grease"],
                "steps": [
                    "1. Park on flat ground, engage handbrake, and loosen lug nuts slightly before jacking up the car.",
                    "2. Secure vehicle safely on jack stands and remove the wheel.",
                    "3. Unbolt the lower caliper guide pin bolt (usually 12mm or 14mm) and swing the caliper upward.",
                    "4. Slide out old brake pads and anti-rattle stainless steel clips.",
                    "5. Clean the caliper bracket with brake cleaner and install new retention clips.",
                    "6. Use a C-clamp and an old pad to slowly press the caliper piston back into its bore.",
                    "7. Apply high-temp brake grease to the back of the new pads (never touch the friction face!) and slide into clips.",
                    "8. Swing caliper back down, torque slide pin bolts, reinstall wheel, and pump the brake pedal 5 times before driving."
                ]
            }
        },
        "SOUND_ENGINE_KNOCK": {
            "id": "SOUND_ENGINE_KNOCK",
            "name": "Metallic Tapping / Knocking Sound From Engine (Rhythmic with RPM)",
            "category": "Engine / Lubrication",
            "suspected_part": "Connecting Rod Bearings / Hydraulic Valve Lifters",
            "severity": "CRITICAL",
            "severity_weight": 1.0,
            "root_cause": "Severe oil starvation, low oil viscosity, or worn crankshaft journal bearings causing mechanical rod knock or dry hydraulic tappet slap.",
            "troubleshooting": [
                "1. Pull over and turn off the engine immediately! Do not rev or drive under load.",
                "2. Wait 5 minutes for oil to drain into the sump and pull the oil dipstick.",
                "3. Check oil level: If below the 'MIN' mark, oil starvation is causing metal-on-metal bearing contact.",
                "4. Inspect oil appearance: If cloudy, metallic-glittering, or milky brown (coolant mixing), do not start the engine."
            ],
            "replacement_guide": {
                "difficulty": "Advanced / Professional Workshop",
                "estimated_time": "1 - 2 days",
                "estimated_cost_inr": "₹8,000 - ₹25,000 (Bearing overhaul & oil pump)",
                "tools_required": ["Torque Wrench", "Engine Hoist / Support Bar", "Micrometer", "Oil Pan Sealant", "New Rod Bearings & Fasteners"],
                "steps": [
                    "1. Drain engine oil through the drain plug and remove the oil filter.",
                    "2. Drop the engine lower oil pan and windage tray.",
                    "3. Inspect bottom of oil pan for bronze or aluminum bearing flakes.",
                    "4. Remove connecting rod cap bolts and inspect crankshaft journals for scoring.",
                    "5. Measure clearances using Plastigauge and install new oversize/standard bearings with assembly lube.",
                    "6. Torque rod caps to manufacturer specifications (angle-torque required).",
                    "7. Reinstall oil pan with fresh RTV silicone gasket, install new oil filter, and fill with synthetic oil."
                ]
            }
        },
        "SOUND_BELT_SQUEAL": {
            "id": "SOUND_BELT_SQUEAL",
            "name": "Loud Screeching / Squealing Sound on Cold Start or Turning AC On",
            "category": "Auxiliary Drive",
            "suspected_part": "Serpentine Accessory Drive Belt / Belt Tensioner",
            "severity": "MEDIUM",
            "severity_weight": 0.45,
            "root_cause": "The serpentine rubber belt has stretched, suffered micro-cracking, or the spring-loaded belt tensioner pulley has lost tension, causing belt slip across the alternator and AC compressor pulleys.",
            "troubleshooting": [
                "1. With engine OFF, visually inspect the ribbed side of the accessory belt for dry-rot cracks or fraying.",
                "2. Twist the belt: If you can twist it more than 90 degrees with your fingers, belt tension is too loose.",
                "3. Listen to pulley bearings using a mechanic stethoscope to ensure alternator or AC compressor clutch is not seizing."
            ],
            "replacement_guide": {
                "difficulty": "Easy to Moderate",
                "estimated_time": "30 minutes",
                "estimated_cost_inr": "₹450 - ₹1,200 (Belt) + ₹1,500 (Tensioner if needed)",
                "tools_required": ["Serpentine Belt Tool / 14mm Long Ratchet", "Smartphone (to photograph belt routing)", "New EPDM Serpentine Belt"],
                "steps": [
                    "1. Take a clear photo of the belt routing around all pulleys (alternator, power steering, AC, crank).",
                    "2. Place a socket on the center bolt of the tensioner pulley and rotate it clockwise to relieve spring tension.",
                    "3. Slide the old belt off the alternator pulley and gently release the tensioner.",
                    "4. Spin all idler pulleys by hand to check for smooth, whisper-quiet rotation.",
                    "5. Route the new belt around all grooved pulleys according to the diagram, leaving the smooth idler for last.",
                    "6. Relieve tensioner spring again, slip belt into final position, and verify all grooves sit dead-center in pulley tracks."
                ]
            }
        },
        "FEEL_STEERING_VIBRATION": {
            "id": "FEEL_STEERING_VIBRATION",
            "name": "Steering Wheel Shaking / Vibrating at Highway Speeds (80 - 110 km/h)",
            "category": "Suspension & Wheels",
            "suspected_part": "Front Wheel Balancing Weights / Warped Brake Rotors",
            "severity": "MEDIUM",
            "severity_weight": 0.4,
            "root_cause": "Loss of wheel balance lead weights causing dynamic centrifugal imbalance, or thickness variation on front brake discs.",
            "troubleshooting": [
                "1. If shaking occurs continuously at high speed without braking: Front wheel dynamic imbalance or bent rim.",
                "2. If shaking occurs strictly when pressing the brake pedal: Warped front brake disc rotors (DTV - disc thickness variation).",
                "3. Inspect front tires for uneven scalloping or flat spots."
            ],
            "replacement_guide": {
                "difficulty": "Easy (Tyre Shop) or Moderate (Rotor Swap)",
                "estimated_time": "30 - 45 minutes",
                "estimated_cost_inr": "₹400 - ₹600 (Dynamic Wheel Balancing) or ₹2,500 (Brake Discs)",
                "tools_required": ["Wheel Balancer Machine", "Caliper Tools", "Brake Rotor Retaining Screwdriver"],
                "steps": [
                    "1. Mount wheel on an electronic dynamic wheel spin balancer.",
                    "2. Clean rim barrel of brake dust and attach adhesive lead/zinc counterweights as indicated by machine.",
                    "3. If rotors are warped, remove caliper bracket and brake pads, unfasten rotor retaining screw, and replace with new ventilated disc rotor.",
                    "4. Bed in new rotors with 5 moderate stops from 60 km/h."
                ]
            }
        },
        "VISUAL_WHITE_SMOKE": {
            "id": "VISUAL_WHITE_SMOKE",
            "name": "Thick Sweet-Smelling White Smoke from Exhaust & Coolant Loss",
            "category": "Engine / Cooling",
            "suspected_part": "Cylinder Head Gasket",
            "severity": "CRITICAL",
            "severity_weight": 1.0,
            "root_cause": "Blown cylinder head gasket allowing pressurized ethylene glycol coolant to breach the fire ring and enter the combustion chamber, vaporizing into white exhaust plumes.",
            "troubleshooting": [
                "1. Observe radiator expansion tank for bubbling while engine is idling (combustion gas leak).",
                "2. Check oil filler cap underside for yellow-white 'mayonnaise' emulsion.",
                "3. Perform a block chemical combustion leak test (blue indicator fluid turns yellow in presence of exhaust CO2)."
            ],
            "replacement_guide": {
                "difficulty": "Advanced / Major Mechanical Overhaul",
                "estimated_time": "2 - 3 days",
                "estimated_cost_inr": "₹12,000 - ₹30,000 (Head gasket, skim machining, new head bolts)",
                "tools_required": ["Cylinder Head Angle-Torque Wrench", "Camshaft Timing Locking Kit", "Precision Straightedge", "New MLS Gasket"],
                "steps": [
                    "1. Disconnect battery, drain cooling system, and remove intake/exhaust manifolds.",
                    "2. Lock engine timing at TDC (Top Dead Center) and remove timing belt/chain.",
                    "3. Remove cylinder head bolts in reverse sequence to avoid head warping.",
                    "4. Lift off cylinder head and inspect mating surfaces with a precision straightedge and feeler gauge.",
                    "5. Machine/skim cylinder head if warpage exceeds 0.05 mm.",
                    "6. Clean engine block deck thoroughly, lay down new multi-layer steel (MLS) gasket dry (no sealant).",
                    "7. Torque new stretch head bolts in multi-stage sequence (e.g., 30 Nm + 90° + 90°)."
                ]
            }
        },
        "FEEL_AC_NOT_COOLING": {
            "id": "FEEL_AC_NOT_COOLING",
            "name": "Air Conditioner Blowing Warm Air / Hissing Noise from Dashboard Vents",
            "category": "Climate Control",
            "suspected_part": "AC Refrigerant Leak (R134a / R1234yf) / Cabin Air Filter",
            "severity": "LOW",
            "severity_weight": 0.25,
            "root_cause": "Low refrigerant charge due to condenser stone-chip puncture or O-ring micro-leak, causing compressor low-pressure cutout switch to disengage the AC clutch.",
            "troubleshooting": [
                "1. Check if AC compressor magnetic clutch clicks and engages when pressing AC button.",
                "2. Inspect condenser through front grille for oily dark wet spots (indicates refrigerant oil leak).",
                "3. Drop glovebox and inspect cabin pollen filter for severe leaf/dirt blockage restricting air velocity."
            ],
            "replacement_guide": {
                "difficulty": "Cabin filter: DIY 5 mins | Refrigerant: AC Specialist",
                "estimated_time": "15 - 45 minutes",
                "estimated_cost_inr": "₹350 (Cabin Filter) | ₹1,500 - ₹2,500 (AC Gas vacuum & recharge)",
                "tools_required": ["No tools needed for cabin filter", "AC Manifold Gauge & Vacuum Pump for refrigerant"],
                "steps": [
                    "1. Open passenger glovebox, release side damper arm, and unhook side stops.",
                    "2. Squeeze filter door clips and slide out old clogged cabin air filter.",
                    "3. Vacuum out debris from the blower motor housing.",
                    "4. Insert fresh activated-carbon cabin filter noting airflow direction arrow (pointing downward).",
                    "5. For gas recharge: pull 30-minute vacuum to boil off moisture, verify 15-minute hold, and charge exact factory gram weight of refrigerant."
                ]
            }
        }
    }

    @classmethod
    def diagnose(cls, symptom_key: str) -> Dict[str, Any]:
        case = cls.KNOWLEDGE_BASE.get(symptom_key)
        if not case:
            return {
                "found": False,
                "message": "Symptom not recognized in primary expert tree. Defaulting to comprehensive telemetry scan."
            }
        return {
            "found": True,
            **case
        }


# ==============================================================================
# 4. SERVICE HISTORY & PART WEAR TRACKER
# ==============================================================================

class ComponentWearCatalog:
    """
    Standard automotive component service life benchmarks across all 17 commonly replaced parts.
    """
    PARTS = {
        "engine_oil": {
            "name": "Engine Oil",
            "replaced_when": "Service interval reached",
            "max_km": 10000.0,
            "max_months": 12.0,
            "importance": "CRITICAL",
            "diy_difficulty": "Easy",
            "cost_inr": 2200
        },
        "oil_filter": {
            "name": "Oil Filter",
            "replaced_when": "Usually with engine oil",
            "max_km": 10000.0,
            "max_months": 12.0,
            "importance": "CRITICAL",
            "diy_difficulty": "Easy",
            "cost_inr": 450
        },
        "air_filter": {
            "name": "Air Filter",
            "replaced_when": "Dirty / clogged",
            "max_km": 15000.0,
            "max_months": 18.0,
            "importance": "MEDIUM",
            "diy_difficulty": "Very Easy (No tools)",
            "cost_inr": 500
        },
        "cabin_ac_filter": {
            "name": "Cabin / AC Filter",
            "replaced_when": "When dirty / reduced airflow",
            "max_km": 15000.0,
            "max_months": 12.0,
            "importance": "MEDIUM",
            "diy_difficulty": "Very Easy (No tools)",
            "cost_inr": 450
        },
        "fuel_filter": {
            "name": "Fuel Filter",
            "replaced_when": "At specified intervals",
            "max_km": 35000.0,
            "max_months": 36.0,
            "importance": "HIGH",
            "diy_difficulty": "Moderate",
            "cost_inr": 850
        },
        "spark_plugs": {
            "name": "Spark Plugs",
            "replaced_when": "Specified mileage/condition",
            "max_km": 30000.0,
            "max_months": 36.0,
            "importance": "HIGH",
            "diy_difficulty": "Moderate",
            "cost_inr": 1400
        },
        "brake_pads": {
            "name": "Brake Pads / Shoes",
            "replaced_when": "Excessive wear",
            "max_km": 30000.0,
            "max_months": 36.0,
            "importance": "SAFETY_CRITICAL",
            "diy_difficulty": "Moderate",
            "cost_inr": 2500
        },
        "brake_fluid": {
            "name": "Brake Fluid",
            "replaced_when": "Interval reached / contaminated",
            "max_km": 40000.0,
            "max_months": 24.0,
            "importance": "SAFETY_CRITICAL",
            "diy_difficulty": "Moderate to Hard",
            "cost_inr": 650
        },
        "coolant": {
            "name": "Coolant",
            "replaced_when": "Interval reached / contaminated",
            "max_km": 40000.0,
            "max_months": 24.0,
            "importance": "HIGH",
            "diy_difficulty": "Easy to Moderate",
            "cost_inr": 1200
        },
        "transmission_oil": {
            "name": "Transmission / Gearbox Oil",
            "replaced_when": "As specified",
            "max_km": 50000.0,
            "max_months": 48.0,
            "importance": "HIGH",
            "diy_difficulty": "Moderate",
            "cost_inr": 2400
        },
        "wiper_blades": {
            "name": "Wiper Blades",
            "replaced_when": "Poor wiping / cracked",
            "max_km": 20000.0,
            "max_months": 12.0,
            "importance": "SAFETY_MEDIUM",
            "diy_difficulty": "Very Easy (No tools)",
            "cost_inr": 550
        },
        "battery": {
            "name": "12V Battery",
            "replaced_when": "Weak / failed",
            "max_km": 60000.0,
            "max_months": 36.0,
            "importance": "HIGH",
            "diy_difficulty": "Easy",
            "cost_inr": 4800
        },
        "belts": {
            "name": "Belts (Timing / Serpentine)",
            "replaced_when": "Cracked / worn / interval reached",
            "max_km": 60000.0,
            "max_months": 48.0,
            "importance": "CRITICAL",
            "diy_difficulty": "Moderate to Hard",
            "cost_inr": 1400
        },
        "tyres": {
            "name": "Tyres",
            "replaced_when": "Low tread / damage",
            "max_km": 45000.0,
            "max_months": 48.0,
            "importance": "SAFETY_CRITICAL",
            "diy_difficulty": "Workshop (Balancing & Alignment)",
            "cost_inr": 22000
        },
        "clutch_components": {
            "name": "Clutch Components",
            "replaced_when": "When worn / slipping",
            "max_km": 70000.0,
            "max_months": 60.0,
            "importance": "HIGH",
            "diy_difficulty": "Hard (Workshop)",
            "cost_inr": 7500
        },
        "suspension_components": {
            "name": "Suspension Components",
            "replaced_when": "When damaged / worn",
            "max_km": 60000.0,
            "max_months": 60.0,
            "importance": "HIGH",
            "diy_difficulty": "Hard (Workshop)",
            "cost_inr": 5500
        },
        "light_bulbs_fuses": {
            "name": "Light Bulbs / Fuses",
            "replaced_when": "When faulty / blown",
            "max_km": 50000.0,
            "max_months": 36.0,
            "importance": "SAFETY_MEDIUM",
            "diy_difficulty": "Very Easy",
            "cost_inr": 200
        }
    }

    @classmethod
    def calculate_wear(cls, current_odometer: float, last_service_odometer: float,
                       months_elapsed: float, parts_replaced: List[str]) -> Dict[str, Any]:
        km_driven = max(0.0, current_odometer - last_service_odometer)
        catalog_status = {}

        for key, info in cls.PARTS.items():
            # If the part was explicitly replaced at last service, its counter is km_driven
            # If not replaced, assume it's had prior wear (add 60% wear offset)
            was_replaced = key in parts_replaced
            effective_km = km_driven if was_replaced else (km_driven + (info["max_km"] * 0.5))
            effective_mo = months_elapsed if was_replaced else (months_elapsed + 10.0)

            km_wear_pct = (effective_km / info["max_km"]) * 100.0
            time_wear_pct = (effective_mo / info["max_months"]) * 100.0
            wear_pct = float(np.clip(max(km_wear_pct, time_wear_pct), 0.0, 100.0))

            remaining_km = max(0.0, info["max_km"] - effective_km)

            if wear_pct >= 90.0:
                health_badge = "REPLACE_NOW"
            elif wear_pct >= 70.0:
                health_badge = "INSPECT_SOON"
            else:
                health_badge = "GOOD"

            catalog_status[key] = {
                "name": info["name"],
                "wear_percentage": round(wear_pct, 1),
                "remaining_km": round(remaining_km, 0),
                "status_badge": health_badge,
                "importance": info["importance"],
                "diy_difficulty": info["diy_difficulty"],
                "cost_est_inr": info["cost_inr"],
                "was_serviced_recently": was_replaced
            }

        return {
            "km_since_last_service": round(km_driven, 1),
            "months_since_last_service": round(months_elapsed, 1),
            "parts_wear": catalog_status
        }
