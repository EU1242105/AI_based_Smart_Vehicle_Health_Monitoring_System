"""
Diagnostic Wizard & DIY vs Mechanic Decision Engine (Adaptive Vehicle Logic)
Implements vehicle-type conditioned diagnostic question trees and part-service workflows:
- Covers all 17 commonly replaced automotive components and their specific replacement triggers.
- Multi-category support: Passenger Cars & SUVs, Two-Wheelers, EVs, Commercial Trucks.

Aligned with:
- TCS: Conditional Formal Grammar & Subsystem Transition Automata
- Soft Computing: Vehicle-Categorized Fuzzy Decision Inference
- AIML: Contextual Question Generation & Multi-Class Failure Diagnosis
"""

from typing import Dict, List, Any, Optional
import numpy as np


class DiagnosticWizardEngine:
    # --------------------------------------------------------------------------
    # 1. COMMONLY REPLACED PARTS & SERVICE BENCHMARKS (17 CORE COMPONENTS)
    # --------------------------------------------------------------------------
    COMMONLY_REPLACED_PARTS = {
        "ENGINE_OIL": {
            "name": "Engine Oil",
            "replaced_when": "Service interval reached (every 10,000 km or 1 year)",
            "category": "Engine Lubrication System",
            "diy_friendly": True,
            "cost_inr": "₹1,800 - ₹3,200",
            "time_estimate": "30 minutes"
        },
        "OIL_FILTER": {
            "name": "Oil Filter",
            "replaced_when": "Usually with engine oil at every service interval",
            "category": "Engine Lubrication System",
            "diy_friendly": True,
            "cost_inr": "₹250 - ₹550",
            "time_estimate": "15 minutes"
        },
        "AIR_FILTER": {
            "name": "Air Filter",
            "replaced_when": "Dirty / clogged with dust, soot, or debris",
            "category": "Air Intake & Induction",
            "diy_friendly": True,
            "cost_inr": "₹350 - ₹700",
            "time_estimate": "5 - 10 minutes"
        },
        "CABIN_AC_FILTER": {
            "name": "Cabin / AC Filter",
            "replaced_when": "Dirty / reduced airflow / musty odor (annual)",
            "category": "Climate Control System",
            "diy_friendly": True,
            "cost_inr": "₹350 - ₹650",
            "time_estimate": "10 minutes"
        },
        "FUEL_FILTER": {
            "name": "Fuel Filter",
            "replaced_when": "At specified service intervals (every 30,000–40,000 km)",
            "category": "Fuel Delivery System",
            "diy_friendly": False,
            "cost_inr": "₹450 - ₹1,200",
            "time_estimate": "25 minutes"
        },
        "SPARK_PLUGS": {
            "name": "Spark Plugs",
            "replaced_when": "Petrol vehicles, at specified mileage / condition (20k-40k km)",
            "category": "Ignition System",
            "diy_friendly": True,
            "cost_inr": "₹600 - ₹1,800",
            "time_estimate": "30 minutes"
        },
        "BRAKE_PADS_SHOES": {
            "name": "Brake Pads / Shoes",
            "replaced_when": "Excessive wear (< 3mm friction material) / squealing / grinding",
            "category": "Braking System",
            "diy_friendly": False,
            "cost_inr": "₹1,500 - ₹3,500",
            "time_estimate": "45 minutes"
        },
        "BRAKE_FLUID": {
            "name": "Brake Fluid",
            "replaced_when": "Periodically / interval reached (every 2 years) / contaminated",
            "category": "Hydraulic Braking System",
            "diy_friendly": False,
            "cost_inr": "₹400 - ₹800",
            "time_estimate": "40 minutes"
        },
        "COOLANT": {
            "name": "Coolant",
            "replaced_when": "Periodically / interval reached (every 2-3 years) / contaminated",
            "category": "Engine Cooling System",
            "diy_friendly": True,
            "cost_inr": "₹600 - ₹1,400",
            "time_estimate": "30 minutes"
        },
        "TRANSMISSION_OIL": {
            "name": "Transmission / Gearbox Oil",
            "replaced_when": "As specified by OEM (40,000–80,000 km) / hard shifting",
            "category": "Transmission & Drivetrain",
            "diy_friendly": False,
            "cost_inr": "₹1,200 - ₹3,500",
            "time_estimate": "45 minutes"
        },
        "WIPER_BLADES": {
            "name": "Wiper Blades",
            "replaced_when": "Poor wiping / cracked rubber / streaking / damaged / worn",
            "category": "Visibility & Safety",
            "diy_friendly": True,
            "cost_inr": "₹300 - ₹800",
            "time_estimate": "5 minutes"
        },
        "BATTERY_ELECTRICAL": {
            "name": "Battery (12V)",
            "replaced_when": "Weak / failing / slow cranking / failed (3-4 years)",
            "category": "Electrical & Starting System",
            "diy_friendly": True,
            "cost_inr": "₹4,000 - ₹6,500",
            "time_estimate": "15 minutes"
        },
        "DRIVE_BELTS": {
            "name": "Belts (Timing / Serpentine)",
            "replaced_when": "Cracked / worn / interval reached / squealing",
            "category": "Engine Ancillary Drive",
            "diy_friendly": False,
            "cost_inr": "₹600 - ₹1,800 (Serpentine) / ₹4,500+ (Timing)",
            "time_estimate": "30 mins (Serpentine) / 3 hrs (Timing)"
        },
        "TYRES": {
            "name": "Tyres",
            "replaced_when": "Low tread (< 1.6mm / TWI) / damage / sidewall crack / worn",
            "category": "Wheels & Tyres",
            "diy_friendly": False,
            "cost_inr": "₹4,000 - ₹7,500 per tyre",
            "time_estimate": "30 minutes (Workshop)"
        },
        "CLUTCH_COMPONENTS": {
            "name": "Clutch Components",
            "replaced_when": "When worn / slipping / hard pedal / high bite point",
            "category": "Transmission & Clutch System",
            "diy_friendly": False,
            "cost_inr": "₹4,500 - ₹11,000",
            "time_estimate": "4 - 5 hours (Workshop)"
        },
        "SUSPENSION_COMPONENTS": {
            "name": "Suspension Components",
            "replaced_when": "When damaged / worn / clunking over bumps / excessive bounce",
            "category": "Suspension & Steering System",
            "diy_friendly": False,
            "cost_inr": "₹2,500 - ₹7,000",
            "time_estimate": "2 - 3 hours (Workshop)"
        },
        "LIGHTS_FUSES": {
            "name": "Light Bulbs / Fuses",
            "replaced_when": "When faulty / blown filament / non-functional",
            "category": "Electrical & Lighting System",
            "diy_friendly": True,
            "cost_inr": "₹50 - ₹250",
            "time_estimate": "5 - 10 minutes"
        }
    }

    # --------------------------------------------------------------------------
    # 2. TARGETED DIAGNOSTIC CHECK TEMPLATES WITH FRAMED QUESTIONS
    # --------------------------------------------------------------------------
    CHECK_TEMPLATES = {
        # 1. ENGINE OIL
        "ENGINE_OIL": {
            "issue_name": "Engine Oil Service Due / Low Level / Degradation",
            "category": "Engine Lubrication System",
            "replaced_when": "Service interval reached",
            "vehicle_types": ["Car / Sedan", "Compact SUV / SUV", "Two-Wheeler", "Commercial Mini-Truck", "Hatchback"],
            "base_risk": 4.5,
            "base_complexity": 2.0,
            "questions": [
                {
                    "id": "oil_service_interval",
                    "text": "Has the vehicle reached its recommended engine oil service interval?",
                    "options": [
                        "Within interval (< 8,000 km or < 10 months since last change)",
                        "Service interval reached (~10,000 km or 12 months)",
                        "Overdue for oil change (> 12,000 km or 1.5+ years)"
                    ]
                },
                {
                    "id": "dipstick_level_color",
                    "text": "Pull out and wipe engine dipstick: What is the oil level and color?",
                    "options": [
                        "Amber / Golden brown and between MIN and MAX marks",
                        "Dark brown / black but slippery and at correct level",
                        "Critically low (below MIN mark) or pitch black with gritty residue"
                    ]
                },
                {
                    "id": "oil_pressure_smell",
                    "text": "Is there an oil warning light on the dashboard, metallic ticking, or a burning oil smell?",
                    "options": [
                        "None - Engine runs quietly with no warning lights",
                        "Faint burning oil smell after high-speed highway driving",
                        "Red oil canister light glowing or audible metallic tapping from engine"
                    ]
                }
            ]
        },

        # 2. OIL FILTER
        "OIL_FILTER": {
            "issue_name": "Engine Oil Filter Element (Replacement Due with Oil)",
            "category": "Engine Lubrication System",
            "replaced_when": "Usually with engine oil",
            "vehicle_types": ["Car / Sedan", "Compact SUV / SUV", "Two-Wheeler", "Commercial Mini-Truck", "Hatchback"],
            "base_risk": 4.0,
            "base_complexity": 2.5,
            "questions": [
                {
                    "id": "filter_change_history",
                    "text": "Was the oil filter replaced simultaneously during the last oil change?",
                    "options": [
                        "Yes - Brand new OEM filter element installed with fresh oil",
                        "No - Only oil was changed; old filter was left in place",
                        "Uncertain / Service history not documented"
                    ]
                },
                {
                    "id": "filter_leakage_seepage",
                    "text": "Inspect around the oil filter canister base or housing under the engine:",
                    "options": [
                        "Completely dry and clean around filter sealing gasket",
                        "Minor oily road grime / slight weeping around rubber O-ring",
                        "Active oil dripping or filter canister dented / loose"
                    ]
                },
                {
                    "id": "cold_start_oil_light",
                    "text": "How long does the dashboard oil pressure light stay lit on a cold morning start?",
                    "options": [
                        "Extinguishes immediately within 1 second of firing up",
                        "Stays on for 2-3 seconds before turning off (Bypass valve slow)",
                        "Flickers intermittently at idle RPM"
                    ]
                }
            ]
        },

        # 3. AIR FILTER
        "AIR_FILTER": {
            "issue_name": "Engine Air Intake Filter Choked / Restricted Airflow",
            "category": "Air Intake & Fuel Induction",
            "replaced_when": "Dirty / clogged",
            "vehicle_types": ["Car / Sedan", "Compact SUV / SUV", "Two-Wheeler", "Commercial Mini-Truck", "Hatchback"],
            "base_risk": 2.0,
            "base_complexity": 1.0,
            "questions": [
                {
                    "id": "air_filter_visual",
                    "text": "Unclip air cleaner housing box and inspect the paper filter element against light:",
                    "options": [
                        "Clean white or yellow paper pleats with light shining through",
                        "Dusty gray surface with minor dust in pleat valleys",
                        "Heavily blackened with soot, trapped dried leaves, or clogged solid"
                    ]
                },
                {
                    "id": "engine_breathing_power",
                    "text": "Does the engine struggle to breathe, hesitate on throttle revs, or blow black smoke?",
                    "options": [
                        "Smooth instantaneous throttle response across rev range",
                        "Mild sluggishness when accelerating uphill",
                        "Noticeable power loss, engine bogging, or heavy black exhaust smoke"
                    ]
                },
                {
                    "id": "fuel_mileage_impact",
                    "text": "Have you noticed an unexplained drop in fuel economy over recent weeks?",
                    "options": [
                        "Fuel mileage is normal and consistent",
                        "Mileage dropped by 1 to 2 km/L in dusty conditions",
                        "Significant mileage drop (> 3-4 km/L) with rich fuel smell"
                    ]
                }
            ]
        },

        # 4. CABIN / AC FILTER
        "CABIN_AC_FILTER": {
            "issue_name": "Cabin Pollen & AC Filter (Dirty / Odour / Reduced Airflow)",
            "category": "Climate Control & Cabin Air",
            "replaced_when": "When dirty / reduced airflow",
            "vehicle_types": ["Car / Sedan", "Compact SUV / SUV", "Commercial Mini-Truck", "Hatchback"],
            "base_risk": 1.5,
            "base_complexity": 1.0,
            "questions": [
                {
                    "id": "ac_airflow_throw",
                    "text": "Set AC blower fan speed to maximum: How strong is the air velocity from vents?",
                    "options": [
                        "Powerful blast of chilled air across all vents",
                        "Airflow feels weaker than usual despite high fan speed",
                        "Very feeble trickle of air; blower motor whines under strain"
                    ]
                },
                {
                    "id": "cabin_smell_odour",
                    "text": "What does the air smell like when you first switch ON the air conditioning?",
                    "options": [
                        "Neutral, fresh, and odorless air",
                        "Faint stale air for the first 10 seconds",
                        "Unpleasant musty, sour vinegar, or damp mildew odor entering cabin"
                    ]
                },
                {
                    "id": "cabin_filter_inspection",
                    "text": "If you open the glove box and slide out the cabin filter tray:",
                    "options": [
                        "Filter element looks clean and white/light gray",
                        "Moderate dust layer accumulated on top pleated surface",
                        "Thick black soot layer, dead insect debris, mold, or torn paper frame"
                    ]
                }
            ]
        },

        # 5. FUEL FILTER
        "FUEL_FILTER": {
            "issue_name": "Fuel Filter Element (Specified Interval / Fuel Starvation)",
            "category": "Fuel Delivery System",
            "replaced_when": "At specified intervals",
            "vehicle_types": ["Car / Sedan", "Compact SUV / SUV", "Two-Wheeler", "Commercial Mini-Truck", "Hatchback"],
            "base_risk": 5.0,
            "base_complexity": 3.5,
            "questions": [
                {
                    "id": "fuel_filter_mileage",
                    "text": "How many kilometers has the vehicle driven since the fuel filter was last replaced?",
                    "options": [
                        "Under 20,000 km (Recent filter replacement)",
                        "20,000 to 40,000 km (Approaching manufacturer specified interval)",
                        "Over 40,000 km or never replaced since purchase"
                    ]
                },
                {
                    "id": "acceleration_hesitation",
                    "text": "Does the engine jerk, stutter, or lose momentum under hard acceleration or hill climbing?",
                    "options": [
                        "Pulls smoothly with consistent fuel delivery",
                        "Occasional hesitation or flat spot above 3,500 RPM",
                        "Severe jerking, engine sputtering, or cutting out under load"
                    ]
                },
                {
                    "id": "crank_to_start_time",
                    "text": "How long does it take for the engine to fire up when turning ignition?",
                    "options": [
                        "Fires up instantly on first crank (1-2 seconds)",
                        "Requires 4-5 seconds of prolonged cranking before fuel catches",
                        "Extended cranking or requires multiple key cycles to prime fuel pump"
                    ]
                }
            ]
        },

        # 6. SPARK PLUGS
        "SPARK_PLUGS": {
            "issue_name": "Ignition Spark Plugs (Specified Mileage / Electrode Wear / Misfire)",
            "category": "Ignition System",
            "replaced_when": "Specified mileage/condition (petrol vehicles)",
            "vehicle_types": ["Car / Sedan", "Compact SUV / SUV", "Two-Wheeler", "Hatchback"],
            "base_risk": 3.5,
            "base_complexity": 2.5,
            "questions": [
                {
                    "id": "spark_plug_mileage",
                    "text": "What is the recorded mileage on the current set of spark plugs?",
                    "options": [
                        "Under 20,000 km (Standard plugs) or under 60,000 km (Iridium plugs)",
                        "20,000 to 40,000 km (Standard copper/nickel plugs due for replacement)",
                        "Over 40,000 km (Standard) or over 100,000 km (Iridium overdue)"
                    ]
                },
                {
                    "id": "idle_stability",
                    "text": "How does the engine idle when stopped at a red traffic light?",
                    "options": [
                        "Rock steady idle RPM with negligible engine vibration",
                        "Slight intermittent tachometer needle flutter or vibration",
                        "Noticeable engine shaking, rough misfiring idle, or Check Engine light flashing"
                    ]
                },
                {
                    "id": "electrode_visual",
                    "text": "If spark plugs are removed for visual inspection, what do the electrode tips look like?",
                    "options": [
                        "Light grayish-tan deposit with crisp square electrode edges",
                        "Slight rounding of center electrode with normal gap",
                        "Heavy black carbon crust, wet fuel fouling, or melted/eroded center tip"
                    ]
                }
            ]
        },

        # 7. BRAKE PADS / SHOES
        "BRAKE_PADS_SHOES": {
            "issue_name": "Brake Pads / Shoes (Excessive Wear / Acoustic Indicator Squeal)",
            "category": "Braking System",
            "replaced_when": "Excessive wear",
            "vehicle_types": ["Car / Sedan", "Compact SUV / SUV", "Two-Wheeler", "Commercial Mini-Truck", "Hatchback"],
            "base_risk": 7.0,
            "base_complexity": 3.5,
            "questions": [
                {
                    "id": "pad_lining_thickness",
                    "text": "Look through wheel spokes at the outer brake pad: Estimate friction lining thickness:",
                    "options": [
                        "Generous friction material remaining (> 6 mm)",
                        "Moderate wear (3 mm to 5 mm - replacement due soon)",
                        "Critically thin (< 3 mm) or backing plate almost touching disc rotor"
                    ]
                },
                {
                    "id": "brake_sound_type",
                    "text": "What acoustic sound is produced when stepping on the brake pedal?",
                    "options": [
                        "Completely silent stopping",
                        "High-pitched screech or squeal (acoustic metal wear indicator tab)",
                        "Harsh metallic grinding or scraping sound (metal backing plate on iron rotor)"
                    ]
                },
                {
                    "id": "rotor_disc_condition",
                    "text": "Inspect the shiny surface of the circular brake disc rotor:",
                    "options": [
                        "Mirror-smooth face with zero grooving",
                        "Minor superficial circular hairline grooves",
                        "Deep scored channels, heavy lip on outer edge, or blue heat discoloration"
                    ]
                }
            ]
        },

        # 8. BRAKE FLUID
        "BRAKE_FLUID": {
            "issue_name": "Brake Fluid (Interval Reached / Contaminated / Spongy Pedal)",
            "category": "Hydraulic Braking System",
            "replaced_when": "Interval reached / contaminated",
            "vehicle_types": ["Car / Sedan", "Compact SUV / SUV", "Two-Wheeler", "Commercial Mini-Truck", "Hatchback"],
            "base_risk": 6.5,
            "base_complexity": 3.0,
            "questions": [
                {
                    "id": "fluid_reservoir_color",
                    "text": "Inspect translucent brake master cylinder reservoir under the bonnet: What is fluid color?",
                    "options": [
                        "Clear, transparent light amber/honey fluid between MIN and MAX marks",
                        "Dark amber tea color with level near MIN mark",
                        "Pitch dark brown / cloudy murky black fluid or moisture tester > 3%"
                    ]
                },
                {
                    "id": "pedal_firmness_travel",
                    "text": "How does the brake pedal feel when firmly pressed with engine running?",
                    "options": [
                        "Firm resistance with short pedal travel and high bite point",
                        "Slightly soft pedal requiring deeper push to achieve full braking force",
                        "Spongy pedal that sinks slowly toward floorboard or requires pumping"
                    ]
                },
                {
                    "id": "brake_flush_history",
                    "text": "When was the vehicle's hydraulic brake fluid last completely flushed and bled?",
                    "options": [
                        "Within the last 12 to 24 months (Fresh DOT 3 / DOT 4)",
                        "Between 2 to 3 years ago (Replacement interval reached)",
                        "Over 3 years ago or never flushed since vehicle was new"
                    ]
                }
            ]
        },

        # 9. COOLANT
        "COOLANT": {
            "issue_name": "Engine Radiator Coolant (Interval Reached / Contaminated)",
            "category": "Engine Cooling System",
            "replaced_when": "Interval reached / contaminated",
            "vehicle_types": ["Car / Sedan", "Compact SUV / SUV", "Two-Wheeler", "Commercial Mini-Truck", "Hatchback"],
            "base_risk": 6.0,
            "base_complexity": 2.5,
            "questions": [
                {
                    "id": "coolant_color_clarity",
                    "text": "With engine cold, check expansion reservoir: What is fluid level and visual condition?",
                    "options": [
                        "Bright clear translucent Green, Pink, or Blue coolant at MAX mark",
                        "Level is low near MIN line but fluid is clean",
                        "Rusty brown murky fluid, oily sludge floating, or reservoir empty"
                    ]
                },
                {
                    "id": "temperature_gauge_behavior",
                    "text": "How does the instrument cluster engine temperature gauge behave while driving?",
                    "options": [
                        "Pointer locks solidly at normal midpoint (85°C – 92°C)",
                        "Creeps higher during heavy traffic or slow hill climbs",
                        "Spikes into red danger zone, steam from under bonnet, or high temp lamp ON"
                    ]
                },
                {
                    "id": "coolant_service_age",
                    "text": "When was the radiator cooling system last drained, flushed, and refilled?",
                    "options": [
                        "Within last 2 years / 40,000 km",
                        "2 to 4 years ago (Interval reached for standard ethylene glycol)",
                        "Over 4 years ago or plain tap water was topped up repeatedly"
                    ]
                }
            ]
        },

        # 10. TRANSMISSION / GEARBOX OIL
        "TRANSMISSION_OIL": {
            "issue_name": "Transmission / Gearbox Oil (As Specified / Hard Shifting)",
            "category": "Transmission & Drivetrain",
            "replaced_when": "As specified",
            "vehicle_types": ["Car / Sedan", "Compact SUV / SUV", "Two-Wheeler", "Commercial Mini-Truck", "Hatchback"],
            "base_risk": 5.5,
            "base_complexity": 3.5,
            "questions": [
                {
                    "id": "gearbox_service_mileage",
                    "text": "What is the service mileage on the current transmission / gear oil?",
                    "options": [
                        "Within manufacturer interval (< 40,000 km)",
                        "Approaching specified interval (40,000 to 60,000 km)",
                        "Exceeded specified interval (> 60,000 km) or never replaced"
                    ]
                },
                {
                    "id": "gear_shifting_feel",
                    "text": "How does the gear lever feel when shifting between gears?",
                    "options": [
                        "Smooth, effortless gear engagement with crisp gate slotting",
                        "Stiff or notchy shift lever, especially during cold starts into 1st/2nd gear",
                        "Hard metallic crunching, grinding into gear, or transmission slipping"
                    ]
                },
                {
                    "id": "fluid_appearance_smell",
                    "text": "If transmission fluid is inspected via dipstick or check plug:",
                    "options": [
                        "Clean red / honey amber fluid with mild petroleum scent",
                        "Slightly darkened fluid with minor loss of transparency",
                        "Burnt acrid smell, dark opaque black color, or fine metallic shimmer"
                    ]
                }
            ]
        },

        # 11. WIPER BLADES
        "WIPER_BLADES": {
            "issue_name": "Windshield Wiper Blades (Poor Wiping / Cracked / Damaged)",
            "category": "Visibility & Safety Equipment",
            "replaced_when": "Poor wiping / cracked",
            "vehicle_types": ["Car / Sedan", "Compact SUV / SUV", "Commercial Mini-Truck", "Hatchback"],
            "base_risk": 2.0,
            "base_complexity": 1.0,
            "questions": [
                {
                    "id": "wiper_glass_clearing",
                    "text": "Operate windshield wipers with washer spray: How effectively is water cleared?",
                    "options": [
                        "Clean, silent, crystal-clear single sweep with zero streaks",
                        "Leaves thin water lines or light hazy film in driver line of sight",
                        "Broad un-wiped water bands, severe water smearing, or violent chattering"
                    ]
                },
                {
                    "id": "rubber_edge_inspection",
                    "text": "Lift wiper arms off windshield and slide fingertips along rubber squeegee blade:",
                    "options": [
                        "Pliable, razor-sharp straight rubber lip with no imperfections",
                        "Hardened or stiff rubber with microscopic nicks along edge",
                        "Rubber edge is torn, split, cracked, or flapping loose from metal frame"
                    ]
                },
                {
                    "id": "wiper_blade_age",
                    "text": "How long have the current wiper blades been installed on the vehicle?",
                    "options": [
                        "Under 6 months (Recently replaced)",
                        "6 to 12 months (Normal tropical sun & monsoon life reached)",
                        "Over 1 year (Rubber degraded by UV radiation and road heat)"
                    ]
                }
            ]
        },

        # 12. BATTERY (12V)
        "BATTERY_ELECTRICAL": {
            "issue_name": "12V Starter Battery (Weak / Failed / Slow Cranking)",
            "category": "Electrical & Starting System",
            "replaced_when": "Weak / failed",
            "vehicle_types": ["Car / Sedan", "Compact SUV / SUV", "Two-Wheeler", "Commercial Mini-Truck", "Hatchback"],
            "base_risk": 3.0,
            "base_complexity": 2.0,
            "questions": [
                {
                    "id": "starter_crank_speed",
                    "text": "How does the engine crank on a cold morning start?",
                    "options": [
                        "Crisp, rapid crank and starts within 1-2 seconds",
                        "Sluggish 'chug... chug... vroom' with mild instrument cluster dimming",
                        "Rapid clicking sound, dashboard lights go black, or engine will not crank"
                    ]
                },
                {
                    "id": "terminal_corrosion_crust",
                    "text": "Inspect battery terminal posts under the hood:",
                    "options": [
                        "Clean, tight lead clamps with zero corrosion",
                        "Light white or blue powdery sulfate crust on terminal posts",
                        "Heavy corrosion buildup, loose clamps, or swollen/bulging battery case"
                    ]
                },
                {
                    "id": "battery_age_voltage",
                    "text": "What is the age of the 12V battery and multimeter resting voltage (engine OFF)?",
                    "options": [
                        "Under 2 years old / Voltage above 12.6V (Healthy charge)",
                        "2 to 4 years old / Voltage between 12.2V – 12.4V (Weakened state)",
                        "Over 4 years old / Voltage below 12.0V or fails load test"
                    ]
                }
            ]
        },

        # 13. BELTS (SERPENTINE / TIMING)
        "DRIVE_BELTS": {
            "issue_name": "Drive Belts (Cracked / Worn / Interval Reached / Squealing)",
            "category": "Engine Ancillary Drive System",
            "replaced_when": "Cracked / worn / interval reached",
            "vehicle_types": ["Car / Sedan", "Compact SUV / SUV", "Commercial Mini-Truck", "Hatchback"],
            "base_risk": 6.5,
            "base_complexity": 4.5,
            "questions": [
                {
                    "id": "belt_acoustic_squeal",
                    "text": "Does the engine bay emit a piercing high-pitched screeching or chirping sound?",
                    "options": [
                        "Completely silent belt operation",
                        "Brief 5-second chirp on wet or cold morning startup that disappears",
                        "Continuous loud screeching during revving, steering lock, or AC engagement"
                    ]
                },
                {
                    "id": "ribbed_underside_cracks",
                    "text": "Examine the ribbed underside of the serpentine belt using a bright flashlight:",
                    "options": [
                        "Smooth, pliable black rubber ribs with no cracks or fraying",
                        "Minor superficial hairline cracks spaced over 1 inch apart",
                        "Multiple deep transverse cracks per rib, chunks of rubber missing, or frayed cords"
                    ]
                },
                {
                    "id": "timing_belt_interval",
                    "text": "For vehicles equipped with an internal rubber Timing Belt: What is the current odometer reading?",
                    "options": [
                        "Engine uses steel timing chain / Or timing belt replaced recently",
                        "Approaching recommended replacement interval (70,000–90,000 km)",
                        "Exceeded replacement interval (> 100,000 km or 5+ years - High valve collision risk)"
                    ]
                }
            ]
        },

        # 14. TYRES
        "TYRES": {
            "issue_name": "Road Tyres (Low Tread / Damage / Uneven Wear / Vibration)",
            "category": "Wheels, Tyres & Suspension",
            "replaced_when": "Low tread / damage",
            "vehicle_types": ["Car / Sedan", "Compact SUV / SUV", "Two-Wheeler", "Commercial Mini-Truck", "Hatchback"],
            "base_risk": 5.5,
            "base_complexity": 2.5,
            "questions": [
                {
                    "id": "tread_wear_indicator",
                    "text": "Check tyre tread relative to the built-in Tread Wear Indicator (TWI) bridge bars in grooves:",
                    "options": [
                        "Generous tread depth well above wear bars (> 4.0 mm remaining)",
                        "Tread approaching wear bars (2.0 mm to 3.0 mm remaining)",
                        "Tread flush with TWI bars (< 1.6 mm legal limit) or bald smooth patches"
                    ]
                },
                {
                    "id": "sidewall_damage_cuts",
                    "text": "Inspect outer and inner tyre sidewalls for physical trauma:",
                    "options": [
                        "Clean, uniform sidewall with zero cracks or bulges",
                        "Fine micro-cracking from sun weathering or mild curb scuffing",
                        "Visible sidewall bubble/egg, deep cuts exposing internal nylon cords"
                    ]
                },
                {
                    "id": "speed_vibration_alignment",
                    "text": "Is there steering wheel wobble at 80–100 km/h or uneven wear on one shoulder?",
                    "options": [
                        "Smooth steering and even tyre wear across all 4 tyres",
                        "Uneven wear on inner/outer edge (Wheel alignment required)",
                        "Pronounced steering oscillation at highway speeds or vehicle pulling hard"
                    ]
                }
            ]
        },

        # 15. CLUTCH COMPONENTS
        "CLUTCH_COMPONENTS": {
            "issue_name": "Manual Clutch Components (Worn / Slipping / High Bite Point)",
            "category": "Transmission & Clutch System",
            "replaced_when": "When worn / slipping",
            "vehicle_types": ["Car / Sedan", "Compact SUV / SUV", "Two-Wheeler", "Commercial Mini-Truck", "Hatchback"],
            "base_risk": 5.5,
            "base_complexity": 5.0,
            "questions": [
                {
                    "id": "clutch_slipping_acceleration",
                    "text": "In 3rd gear at 40 km/h, press the accelerator pedal firmly to the floor: What happens?",
                    "options": [
                        "Vehicle road speed climbs smoothly and proportionally with engine RPM",
                        "Slight momentary lag before road speed matches engine acceleration",
                        "Engine RPM flares and screams upwards but vehicle fails to accelerate (Clutch Slip)"
                    ]
                },
                {
                    "id": "clutch_pedal_bite_point",
                    "text": "Where does the clutch begin to grab ('bite point') as you lift your left foot off pedal?",
                    "options": [
                        "Comfortable bite in lower or middle third of pedal travel",
                        "Bite point has migrated high up, near the very top of pedal release",
                        "Heavy stiff pedal effort, or vehicle violently judders/shudders on take-off"
                    ]
                },
                {
                    "id": "burning_clutch_smell",
                    "text": "Do you notice a pungent, acrid burning friction smell after hill starts or stop-go traffic?",
                    "options": [
                        "No burning smell under any operating condition",
                        "Occasional faint warm odor after reversing up steep inclines",
                        "Strong acrid burning clutch lining odor wafting directly into cabin"
                    ]
                }
            ]
        },

        # 16. SUSPENSION COMPONENTS
        "SUSPENSION_COMPONENTS": {
            "issue_name": "Suspension Components (Damaged / Worn Struts, Bushings & Links)",
            "category": "Suspension & Steering System",
            "replaced_when": "When damaged / worn",
            "vehicle_types": ["Car / Sedan", "Compact SUV / SUV", "Commercial Mini-Truck", "Hatchback"],
            "base_risk": 5.0,
            "base_complexity": 4.5,
            "questions": [
                {
                    "id": "suspension_clunk_noises",
                    "text": "What sounds occur when driving over speed humps, potholes, or uneven village roads?",
                    "options": [
                        "Firm, muted thud with instant vehicle body stabilization",
                        "Occasional hollow rattling or rubber squeaking over rough patches",
                        "Harsh metallic clunking, banging, or knocking from front or rear wheels"
                    ]
                },
                {
                    "id": "fender_rebound_bounce",
                    "text": "Firmly push down on each corner vehicle fender above the tyre and release quickly:",
                    "options": [
                        "Rises up once and immediately settles into resting height",
                        "Bounces twice before coming to rest (Weak shock damping)",
                        "Continues to bounce or oscillate multiple times (Blown shock absorber)"
                    ]
                },
                {
                    "id": "strut_fluid_leakage",
                    "text": "Inspect the vertical shock absorber cylindrical shafts behind the road wheels:",
                    "options": [
                        "Completely dry shock body with intact rubber dust boots",
                        "Thin film of road grime or light weeping near top seal",
                        "Fresh wet oil actively leaking down shock absorber body or torn rubber boot"
                    ]
                }
            ]
        },

        # 17. LIGHT BULBS / FUSES
        "LIGHTS_FUSES": {
            "issue_name": "Light Bulbs & Fuses (Faulty / Blown Filament / Fuse Trip)",
            "category": "Electrical & Lighting System",
            "replaced_when": "When faulty / blown",
            "vehicle_types": ["Car / Sedan", "Compact SUV / SUV", "Two-Wheeler", "Commercial Mini-Truck", "Hatchback"],
            "base_risk": 2.5,
            "base_complexity": 1.5,
            "questions": [
                {
                    "id": "lighting_failure_scope",
                    "text": "Turn ON exterior lighting switches (Headlights, High Beam, Tail, Brake, Turn signals):",
                    "options": [
                        "All exterior lamps and cabin illumination lights illuminate brightly",
                        "A single individual headlight, brake light, or turn indicator bulb is unlit",
                        "Entire lighting circuit (both headlights, all tail lamps, or entire dashboard) is dead"
                    ]
                },
                {
                    "id": "fuse_box_inspection",
                    "text": "Open the interior or engine-bay fuse box: What is the physical state of the fuse?",
                    "options": [
                        "Metal wire link inside transparent fuse housing is continuous and intact",
                        "Internal metal link is broken / melted through or blackened",
                        "Fuse blew immediately again after inserting a new replacement fuse (Short circuit)"
                    ]
                },
                {
                    "id": "associated_accessories",
                    "text": "Are any other electrical accessories (Horn, Windshield Washer, 12V Socket, Power Windows) non-responsive?",
                    "options": [
                        "All vehicle accessories operate normally",
                        "One specific accessory stopped working at the exact same time as the light",
                        "Multiple unrelated electrical circuits fail intermittently"
                    ]
                }
            ]
        },

        # Specialized / Legacy Templates
        "TWO_WHEELER_CHAIN": {
            "issue_name": "Motorcycle Drive Chain Slack / Dry Links / Sprocket Hooking",
            "category": "Two-Wheeler Final Drive",
            "replaced_when": "Chain elongation / hooked teeth / dry links",
            "vehicle_types": ["Two-Wheeler (Motorcycle / Scooter)", "Two-Wheeler"],
            "base_risk": 5.0,
            "base_complexity": 2.5,
            "questions": [
                {
                    "id": "chain_slack",
                    "text": "Measure vertical chain slack midway between front and rear sprockets:",
                    "options": ["Normal slack (20-30 mm)", "Excessive slack (> 45 mm, slaps swingarm)", "Extremely tight (< 15 mm)"]
                },
                {
                    "id": "chain_lubrication",
                    "text": "Inspect chain links: Is there grease/lube or surface oxidation?",
                    "options": ["Well-lubricated black/clear sheen", "Dry with red surface rust / stiff links"]
                },
                {
                    "id": "sprocket_teeth",
                    "text": "Look at rear wheel sprocket teeth profile:",
                    "options": ["Symmetric rounded teeth", "Hooked / shark-fin sharp worn teeth"]
                }
            ]
        },
        "TWO_WHEELER_FORK": {
            "issue_name": "Front Telescopic Fork Oil Seal Rupture / Cone-set Wobble",
            "category": "Two-Wheeler Suspension & Steering",
            "replaced_when": "Oil seal rupture / notched steering bearing",
            "vehicle_types": ["Two-Wheeler (Motorcycle / Scooter)", "Two-Wheeler"],
            "base_risk": 6.5,
            "base_complexity": 5.5,
            "questions": [
                {
                    "id": "fork_oil_rings",
                    "text": "Inspect shiny chrome fork stanchion tubes:",
                    "options": ["Dry and clean", "Wet hydraulic oil rings / dripping on brake disc"]
                },
                {
                    "id": "handlebar_wobble",
                    "text": "During gentle braking or decelerating from 40 km/h with loose grip:",
                    "options": ["Steers straight and stable", "Handlebar oscillates / wobbles vigorously"]
                }
            ]
        },
        "TWO_WHEELER_START_BATTERY": {
            "issue_name": "Motorcycle VRLA Battery Discharge / Weak Spark Plug",
            "category": "Two-Wheeler Electrical",
            "replaced_when": "Weak / failed",
            "vehicle_types": ["Two-Wheeler (Motorcycle / Scooter)", "Two-Wheeler"],
            "base_risk": 3.0,
            "base_complexity": 2.0,
            "questions": [
                {
                    "id": "self_start_sound",
                    "text": "When pressing the self-start thumb button:",
                    "options": ["Cranks engine briskly", "Single click sound or sluggish chug", "Total dead silence"]
                },
                {
                    "id": "horn_and_indicators",
                    "text": "With engine OFF, turn ignition switch ON and press horn:",
                    "options": ["Loud sharp horn sound", "Feeble squeak / horn fades out"]
                }
            ]
        },
        "EV_12V_AUX_BATTERY": {
            "issue_name": "EV 12V Auxiliary Battery Low Voltage (Main Contactors Inhibit)",
            "category": "EV Low-Voltage Architecture",
            "replaced_when": "Weak / failed",
            "vehicle_types": ["Electric Vehicle (EV - Car/SUV/Scooter)", "Electric Vehicle (EV)"],
            "base_risk": 3.0,
            "base_complexity": 2.0,
            "questions": [
                {
                    "id": "instrument_ready_light",
                    "text": "Press start with brake depressed: Does green 'READY' illuminate?",
                    "options": ["Yes - 'READY' illuminates normally", "No - Screen stays black or flashes 12V error"]
                },
                {
                    "id": "headlights_and_horn",
                    "text": "Turn headlights ON before entering driving mode:",
                    "options": ["Bright headlights and normal horn", "Extremely dim headlights / unresponsive"]
                }
            ]
        },
        "EV_CHARGING_FAULT": {
            "issue_name": "EV AC/DC Fast Charging Protocol Error / Earth Ground Fault",
            "category": "EV Charging Interface",
            "replaced_when": "Socket / charging port / ground fault",
            "vehicle_types": ["Electric Vehicle (EV - Car/SUV/Scooter)", "Electric Vehicle (EV)"],
            "base_risk": 6.0,
            "base_complexity": 4.0,
            "questions": [
                {
                    "id": "charging_gun_lock",
                    "text": "When inserting charging gun into socket:",
                    "options": ["Audible click and gun locks securely", "Gun does not lock mechanically"]
                },
                {
                    "id": "charging_indicator_color",
                    "text": "Color of the charging LED halo around the charge port:",
                    "options": ["Green / Cyan pulsing (Charging)", "Red / Amber solid (Fault tripped)"]
                }
            ]
        },
        "EV_RANGE_DEGRADATION": {
            "issue_name": "High-Voltage Traction Battery Pack Cell Imbalance / Cooling Fault",
            "category": "EV High-Voltage Traction Battery",
            "replaced_when": "Cell imbalance / coolant degradation",
            "vehicle_types": ["Electric Vehicle (EV - Car/SUV/Scooter)", "Electric Vehicle (EV)"],
            "base_risk": 8.0,
            "base_complexity": 7.5,
            "questions": [
                {
                    "id": "sudden_soc_drop",
                    "text": "State of Charge (SoC %) meter behavior under acceleration:",
                    "options": ["Linear gradual percentage drop", "Sudden 15-20% drop under hard throttle"]
                },
                {
                    "id": "battery_coolant_reservoir",
                    "text": "Check blue/clear low-conductivity dielectric coolant reservoir under bonnet:",
                    "options": ["Level between MIN and MAX marks", "Level below MIN / air bubbles in sight glass"]
                }
            ]
        },
        "COMMERCIAL_LEAF_SPRING": {
            "issue_name": "Rear Leaf Spring Shackle Bushing Wear / Center Bolt Fracture",
            "category": "Commercial Suspension & Payload",
            "replaced_when": "Cracked leaf / sheared center bolt",
            "vehicle_types": ["Commercial Mini-Truck / Pickup", "Commercial Mini-Truck"],
            "base_risk": 7.0,
            "base_complexity": 5.0,
            "questions": [
                {
                    "id": "vehicle_lean",
                    "text": "When viewed from rear on level road, is mini-truck body sagging to one side?",
                    "options": ["Level stance", "Noticeable lean to left or right corner (> 2 inches)"]
                },
                {
                    "id": "leaf_cracks",
                    "text": "Inspect the multi-leaf spring pack with a flashlight:",
                    "options": ["Leaves aligned squarely", "Cracked/sheared leaf or displaced center bolt"]
                }
            ]
        },
        "COMMERCIAL_DIESEL_FUEL": {
            "issue_name": "Commercial Diesel Primary Fuel Filter Choked / Water in Fuel",
            "category": "Commercial Diesel Fuel Injection",
            "replaced_when": "Interval reached / water sensor alert",
            "vehicle_types": ["Commercial Mini-Truck / Pickup", "Commercial Mini-Truck"],
            "base_risk": 5.0,
            "base_complexity": 3.0,
            "questions": [
                {
                    "id": "water_in_fuel_light",
                    "text": "Is the amber 'Water in Fuel Filter' symbol illuminated on dashboard?",
                    "options": ["No - Light is off", "Yes - Warning symbol lit continuously"]
                },
                {
                    "id": "black_smoke_under_load",
                    "text": "Exhaust appearance when accelerating fully with cargo payload:",
                    "options": ["Clean / faint clear exhaust", "Heavy billowing black soot / severe loss of pull"]
                }
            ]
        }
    }

    # --------------------------------------------------------------------------
    # 3. PRELIMINARY ISSUE IDENTIFICATION (AI SYMPTOM & PART ROUTING)
    # --------------------------------------------------------------------------
    @classmethod
    def identify_preliminary_issue(cls, user_data: Dict[str, Any]) -> Dict[str, Any]:
        v_type = str(user_data.get("vehicle_type", "Car / Sedan"))
        fuel = str(user_data.get("fuel_type", "Petrol"))
        sound = str(user_data.get("unusual_sound", "None")).lower()
        smell = str(user_data.get("unusual_smell", "None")).lower()
        smoke = str(user_data.get("smoke_color", "None")).lower()
        start_prob = str(user_data.get("start_problem", "No")).lower()
        overheating = str(user_data.get("overheating", "No")).lower()
        vibration = str(user_data.get("vibration", "None")).lower()
        warning_light = str(user_data.get("warning_light", "None")).lower()
        desc = str(user_data.get("problem_description", "")).lower()

        # Check if user explicitly selected a target part from dropdown
        target_part = str(user_data.get("target_part", user_data.get("suspected_part", "NONE"))).upper().strip()

        # If a valid part was explicitly selected from dropdown, prioritize it directly!
        if target_part in cls.CHECK_TEMPLATES:
            template = cls.CHECK_TEMPLATES[target_part]
            return {
                "suspected_issue_key": target_part,
                "issue_name": template["issue_name"],
                "subsystem": template["category"],
                "base_risk": template["base_risk"],
                "confidence_score": 96.5,
                "suggested_checks": template["questions"]
            }

        scores = {k: 0.0 for k in cls.CHECK_TEMPLATES.keys()}

        # Keyword and symptom heuristics for all 17 parts
        # 1. Wiper blades
        if "wiper" in desc or "streak" in desc or "windshield" in desc or "smear" in desc or "rain" in sound:
            scores["WIPER_BLADES"] += 12.0

        # 2. Tyres
        if "tyre" in desc or "tire" in desc or "tread" in desc or "puncture" in desc or "wobble at 80" in vibration or "highway" in vibration:
            scores["TYRES"] += 11.0

        # 3. Clutch
        if "clutch" in desc or "slip" in desc or "high bite" in desc or "clutch" in smell or "gear rev" in desc:
            scores["CLUTCH_COMPONENTS"] += 12.0

        # 4. Drive Belts
        if "belt" in desc or "screech" in sound or "squeal" in sound or "serpentine" in desc or "chirp" in sound:
            scores["DRIVE_BELTS"] += 11.0

        # 5. Brake pads & Brake fluid
        if "brake" in desc or "squeal" in sound or "grind" in sound or "rotor" in desc:
            scores["BRAKE_PADS_SHOES"] += 10.0
        if "spongy" in desc or "pedal sink" in desc or "brake fluid" in desc:
            scores["BRAKE_FLUID"] += 11.0

        # 6. Cabin / AC Filter
        if "cabin" in desc or "ac filter" in desc or "pollen" in desc or "musty" in smell or "weak ac" in desc or "hiss" in sound:
            scores["CABIN_AC_FILTER"] += 11.0

        # 7. Engine Oil & Oil Filter
        if "engine oil" in desc or "oil level" in desc or "dipstick" in desc or "oil canister" in warning_light or "burning oil" in smell or "blue" in smoke:
            scores["ENGINE_OIL"] += 10.0
            scores["OIL_FILTER"] += 8.0

        # 8. Air Filter
        if "air filter" in desc or "air intake" in desc or "choked filter" in desc or "black smoke" in smoke:
            scores["AIR_FILTER"] += 10.0

        # 9. Fuel Filter
        if "fuel filter" in desc or "jerking" in desc or "stutter" in desc or "starvation" in desc:
            scores["FUEL_FILTER"] += 10.0

        # 10. Spark Plugs
        if "spark" in desc or "plug" in desc or "misfire" in desc or "rough idle" in vibration or "engine shakes" in vibration:
            scores["SPARK_PLUGS"] += 11.0

        # 11. Coolant
        if "coolant" in desc or "overheat" in overheating or "temp" in warning_light or "sweet" in smell or "radiator" in desc or "white" in smoke:
            scores["COOLANT"] += 11.0

        # 12. Transmission Oil
        if "transmission" in desc or "gearbox" in desc or "gear oil" in desc or "hard shift" in desc or "notchy" in desc:
            scores["TRANSMISSION_OIL"] += 10.0

        # 13. Suspension Components
        if "suspension" in desc or "shock" in desc or "strut" in desc or "clunk" in sound or "bushing" in desc or "bounce" in vibration:
            scores["SUSPENSION_COMPONENTS"] += 11.0

        # 14. Light Bulbs / Fuses
        if "bulb" in desc or "fuse" in desc or "headlight" in desc or "horn stopped" in desc or "light" in desc:
            scores["LIGHTS_FUSES"] += 11.0

        # 15. 12V Battery
        if "battery" in desc or "click" in sound or "won't start" in start_prob or "slow" in start_prob or "12v" in warning_light:
            scores["BATTERY_ELECTRICAL"] += 10.0

        # Two-Wheeler routing
        if "motorcycle" in v_type.lower() or "two" in v_type.lower() or "scooter" in v_type.lower():
            if "chain" in sound or "chain" in desc:
                scores["TWO_WHEELER_CHAIN"] += 12.0
            if "fork" in desc or "cone" in desc:
                scores["TWO_WHEELER_FORK"] += 12.0

        # EV routing
        elif "ev" in v_type.lower() or "electric" in v_type.lower() or fuel.upper() == "EV":
            if "charge" in desc or "charging" in warning_light:
                scores["EV_CHARGING_FAULT"] += 14.0
            elif "range" in desc or "turtle" in warning_light:
                scores["EV_RANGE_DEGRADATION"] += 14.0
            elif "12v" in desc or "aux" in desc:
                scores["EV_12V_AUX_BATTERY"] += 12.0

        # Commercial routing
        elif "commercial" in v_type.lower() or "truck" in v_type.lower():
            if "leaf" in desc or "spring" in desc or "bounce" in vibration:
                scores["COMMERCIAL_LEAF_SPRING"] += 14.0
            elif "water in fuel" in warning_light or "diesel filter" in desc:
                scores["COMMERCIAL_DIESEL_FUEL"] += 14.0

        # Filter valid candidates by vehicle type
        valid_candidates = {
            k: v for k, v in scores.items()
            if any(vt.lower() in v_type.lower() for vt in cls.CHECK_TEMPLATES[k].get("vehicle_types", []))
        }
        if not valid_candidates:
            valid_candidates = scores

        best_issue_key = max(valid_candidates, key=valid_candidates.get)
        if valid_candidates[best_issue_key] == 0.0:
            if "two" in v_type.lower():
                best_issue_key = "TWO_WHEELER_START_BATTERY"
            elif "ev" in v_type.lower() or fuel.upper() == "EV":
                best_issue_key = "EV_12V_AUX_BATTERY"
            elif "commercial" in v_type.lower():
                best_issue_key = "COMMERCIAL_DIESEL_FUEL"
            else:
                best_issue_key = "ENGINE_OIL"

        template = cls.CHECK_TEMPLATES[best_issue_key]
        confidence = min(96.0, 60.0 + valid_candidates[best_issue_key] * 3.5)

        return {
            "suspected_issue_key": best_issue_key,
            "issue_name": template["issue_name"],
            "subsystem": template["category"],
            "base_risk": template["base_risk"],
            "confidence_score": round(confidence, 1),
            "suggested_checks": template["questions"]
        }

    # --------------------------------------------------------------------------
    # 4. FINAL 3-TIER CLASSIFICATION & ACTION PLAN GENERATOR
    # --------------------------------------------------------------------------
    @classmethod
    def evaluate_final_decision(cls, issue_key: str, answers: Dict[str, str], vehicle_info: Dict[str, Any]) -> Dict[str, Any]:
        template = cls.CHECK_TEMPLATES.get(issue_key, cls.CHECK_TEMPLATES["ENGINE_OIL"])
        risk = float(template["base_risk"])
        complexity = float(template["base_complexity"])
        findings_summary = []

        # Issue-specific answer evaluations
        ans_str = " ".join(answers.values()).lower()

        # Engine Oil
        if issue_key == "ENGINE_OIL":
            if "overdue" in ans_str:
                findings_summary.append("Engine oil change is overdue; thermal shearing has depleted detergent & anti-wear additives.")
                risk += 1.5
            if "critically low" in ans_str or "below min" in ans_str:
                findings_summary.append("Oil level is below safe operating threshold; risk of bearing starvation.")
                risk += 2.5
                complexity += 1.0
            if "oil canister" in ans_str or "metallic" in ans_str:
                findings_summary.append("Low oil hydraulic pressure detected; avoid operating engine under load.")
                risk += 3.0

        # Oil Filter
        elif issue_key == "OIL_FILTER":
            if "no" in ans_str or "old filter" in ans_str:
                findings_summary.append("Oil filter was not replaced with oil; old pleated medium saturated with particulate matter.")
                complexity -= 0.5
            if "active" in ans_str or "dripping" in ans_str:
                findings_summary.append("Active oil leakage at filter rubber sealing ring or canister casing.")
                risk += 2.0

        # Air Filter
        elif issue_key == "AIR_FILTER":
            if "heavily blackened" in ans_str or "clogged" in ans_str:
                findings_summary.append("Air cleaner filter element is clogged with road dust, choking engine breathing.")
                complexity -= 0.5
            if "noticeable power loss" in ans_str or "black smoke" in ans_str:
                findings_summary.append("Rich air-fuel mixture caused by restricted intake manifold airflow.")
                risk += 1.0

        # Cabin / AC Filter
        elif issue_key == "CABIN_AC_FILTER":
            findings_summary.append("Cabin HVAC pollen filter is saturated with particulate matter.")
            if "musty" in ans_str or "mildew" in ans_str:
                findings_summary.append("Trapped moisture has caused mold/mildew spore growth in filter pleats.")
            if "feeble" in ans_str or "whines" in ans_str:
                findings_summary.append("Severe restriction straining AC blower motor.")
                complexity -= 0.5

        # Fuel Filter
        elif issue_key == "FUEL_FILTER":
            if "over 40,000" in ans_str:
                findings_summary.append("Fuel filter has exceeded manufacturer replacement interval.")
            if "severe jerking" in ans_str or "sputtering" in ans_str:
                findings_summary.append("High fuel flow restriction causing injector fuel starvation under throttle.")
                risk += 2.0
                complexity += 1.0

        # Spark Plugs
        elif issue_key == "SPARK_PLUGS":
            if "over 40,000" in ans_str or "due" in ans_str:
                findings_summary.append("Spark plugs have completed service life; center electrode gap has widened.")
            if "noticeable engine shaking" in ans_str or "misfiring" in ans_str:
                findings_summary.append("Cylinder combustion misfire detected; unburnt fuel entering catalytic converter.")
                risk += 2.5
                complexity += 1.0

        # Brake Pads / Shoes
        elif issue_key == "BRAKE_PADS_SHOES":
            if "critically thin" in ans_str or "< 3 mm" in ans_str:
                findings_summary.append("Brake pad friction lining is severely depleted (< 3 mm).")
                risk += 2.5
            if "grinding" in ans_str or "scraping" in ans_str:
                findings_summary.append("CRITICAL: Metal backing plate grinding into iron disc rotor.")
                risk += 3.5
                complexity += 1.5

        # Brake Fluid
        elif issue_key == "BRAKE_FLUID":
            if "dark" in ans_str or "> 3%" in ans_str:
                findings_summary.append("Brake fluid is hygroscopically contaminated with moisture (> 3%).")
                risk += 2.0
            if "spongy" in ans_str or "sinks" in ans_str:
                findings_summary.append("Hydraulic vapor lock or trapped air in brake lines; spongy pedal response.")
                risk += 3.5
                complexity += 1.5

        # Coolant
        elif issue_key == "COOLANT":
            if "rusty" in ans_str or "empty" in ans_str:
                findings_summary.append("Cooling expansion reservoir empty or coolant contaminated with rust/scale.")
                risk += 3.0
            if "spikes into red" in ans_str or "steam" in ans_str:
                findings_summary.append("CRITICAL: Engine overheating hazard; risk of cylinder head warping.")
                risk += 4.0
                complexity += 2.0

        # Transmission Oil
        elif issue_key == "TRANSMISSION_OIL":
            if "exceeded" in ans_str:
                findings_summary.append("Transmission gear oil has broken down due to thermal and shear stress.")
            if "grinding" in ans_str or "slipping" in ans_str:
                findings_summary.append("Synchronizer ring friction failure or hydraulic transmission slip.")
                risk += 2.5
                complexity += 2.0

        # Wiper Blades
        elif issue_key == "WIPER_BLADES":
            findings_summary.append("Windshield wiper rubber squeegee is oxidized, hardened, or split.")
            if "broad" in ans_str or "severe" in ans_str:
                findings_summary.append("Poor wiping visibility presents safety hazard in wet weather.")
                risk += 1.0

        # 12V Battery
        elif issue_key == "BATTERY_ELECTRICAL":
            if "rapid clicking" in ans_str or "lights go black" in ans_str:
                findings_summary.append("12V battery lacks sufficient CCA (Cold Cranking Amps) to turn starter motor.")
                risk += 1.5
            if "heavy corrosion" in ans_str:
                findings_summary.append("Lead sulfate oxidation causing high electrical contact resistance.")

        # Drive Belts
        elif issue_key == "DRIVE_BELTS":
            if "screeching" in ans_str:
                findings_summary.append("Serpentine drive belt slipping across alternator / AC compressor pulleys.")
                risk += 1.5
            if "multiple deep" in ans_str or "chunks" in ans_str:
                findings_summary.append("Severe transverse rib cracking; imminent belt snapping hazard.")
                risk += 3.0
                complexity += 1.5
            if "exceeded replacement interval" in ans_str:
                findings_summary.append("CRITICAL: Timing belt replacement overdue; catastrophic valve interference hazard.")
                risk += 5.0
                complexity += 3.0

        # Tyres
        elif issue_key == "TYRES":
            if "flush with twi" in ans_str or "bald" in ans_str:
                findings_summary.append("Tyre tread depth has reached legal wear limit (< 1.6 mm); hydroplaning risk.")
                risk += 2.5
            if "bubble" in ans_str or "deep cuts" in ans_str:
                findings_summary.append("CRITICAL: Internal tyre sidewall cord delamination / blowout risk.")
                risk += 4.5
                complexity += 1.5

        # Clutch Components
        elif issue_key == "CLUTCH_COMPONENTS":
            if "flares and screams" in ans_str or "clutch slip" in ans_str:
                findings_summary.append("Clutch friction disc is glazed and slipping under engine torque.")
                risk += 2.5
                complexity += 3.0
            if "acrid burning" in ans_str:
                findings_summary.append("Extreme thermal friction wear on clutch pressure plate & flywheel.")
                risk += 2.0

        # Suspension Components
        elif issue_key == "SUSPENSION_COMPONENTS":
            if "metallic clunking" in ans_str:
                findings_summary.append("Lower control arm ball joint or stabilizer link bushing has excessive play.")
                risk += 2.5
                complexity += 2.0
            if "actively leaking" in ans_str or "oscillate" in ans_str:
                findings_summary.append("Hydraulic shock absorber seal ruptured; vehicle damping compromised.")
                risk += 2.0
                complexity += 2.0

        # Light Bulbs / Fuses
        elif issue_key == "LIGHTS_FUSES":
            if "melted" in ans_str or "broken" in ans_str:
                findings_summary.append("Electrical circuit fuse blown due to over-current surge.")
            if "blew immediately" in ans_str:
                findings_summary.append("Dead short circuit to vehicle ground detected in wiring harness.")
                risk += 2.0
                complexity += 2.0

        # Generic summary fallback
        if not findings_summary:
            findings_summary.append(f"Physical inspection indicates {template['issue_name']} needs attention.")

        # Composite Fuzzy Decision Score (0 to 100)
        fuzzy_score = (risk * 5.5) + (complexity * 4.5)
        fuzzy_score = float(np.clip(fuzzy_score, 10.0, 100.0))

        # 3-Tier Classification Decision
        if fuzzy_score >= 68.0:
            result_tier = "MECHANIC_REQUIRED"
            tier_badge = "🔴 Mechanic Required"
            tier_meaning = "The issue involves safety-critical mechanics, timing belts, clutch replacement, or hydraulic systems requiring specialized workshop tools."
            color = "#ff1744"
        elif fuzzy_score >= 42.0:
            result_tier = "INSPECTION_REQUIRED"
            tier_badge = "🟡 Inspection Required"
            tier_meaning = "Visual inspection and basic servicing can be started by the owner, but professional workshop validation is advised before high-speed driving."
            color = "#ffb300"
        else:
            result_tier = "DIY"
            tier_badge = "🟢 DIY Repairable"
            tier_meaning = "Safe, straightforward maintenance or replacement that can be completed directly by the owner with simple hand tools."
            color = "#00e676"

        action_plan = cls._generate_action_plan(issue_key, result_tier, vehicle_info)

        return {
            "result_tier": result_tier,
            "tier_badge": tier_badge,
            "tier_meaning": tier_meaning,
            "badge_color": color,
            "fuzzy_decision_score": round(fuzzy_score, 1),
            "risk_rating": round(min(10.0, risk), 1),
            "complexity_rating": round(min(10.0, complexity), 1),
            "findings_summary": findings_summary,
            "action_plan": action_plan
        }

    # --------------------------------------------------------------------------
    # 5. DETAILED ACTION PLANS & REPAIR GUIDES
    # --------------------------------------------------------------------------
    @classmethod
    def _generate_action_plan(cls, issue_key: str, tier: str, vehicle_info: Dict[str, Any]) -> Dict[str, Any]:
        brand = vehicle_info.get("brand", "Vehicle")
        model = vehicle_info.get("model", "")

        plans = {
            "ENGINE_OIL": {
                "primary_action": f"Drain old engine oil, replace crush washer, and refill with fresh synthetic oil on {brand} {model}.",
                "steps": [
                    "1. Warm engine for 5 minutes, then turn OFF and park on flat level ground with handbrake firmly set.",
                    "2. Place oil catch basin under engine sump oil drain bolt.",
                    "3. Use a 14mm or 17mm ring spanner to loosen drain bolt counter-clockwise; allow dirty oil to drain completely.",
                    "4. Clean drain bolt, fit a new copper crush washer, and tighten securely to 35-40 Nm torque.",
                    "5. Unscrew engine oil filler cap on top of valve cover and pour in manufacturer-specified oil grade (e.g. 5W-30 API SP/SN).",
                    "6. Check level with dipstick; ensure oil sits exactly between MIN and MAX marks. Run engine 2 minutes and re-check."
                ],
                "tools_needed": ["14mm / 17mm Ring Spanner", "Oil Drain Pan (5L capacity)", "Funnel", "Clean Microfiber Rag", "Fresh Engine Oil"],
                "cost_estimate": "₹1,800 - ₹3,200 (Engine oil can) vs ₹500 (Workshop labor)",
                "estimated_time": "30 minutes"
            },
            "OIL_FILTER": {
                "primary_action": f"Replace oil filter element and lubricate rubber O-ring gasket on {brand} {model}.",
                "steps": [
                    "1. Locate oil filter (under engine bay or top-mounted cartridge).",
                    "2. Place catch pan beneath filter to catch residual oil spillage.",
                    "3. Use an oil filter strap wrench to unscrew old filter counter-clockwise.",
                    "4. Verify old rubber gasket came off with the filter; clean metal mating surface with rag.",
                    "5. Smear clean fresh engine oil onto the rubber seal ring of the new filter.",
                    "6. Spin new filter on by hand until gasket contacts base, then hand-tighten an additional 3/4 turn (Do not over-torque with wrench)."
                ],
                "tools_needed": ["Oil Filter Cup / Strap Wrench", "Catch Pan", "Clean Rag", "New OEM Oil Filter"],
                "cost_estimate": "₹250 - ₹550 (Filter element)",
                "estimated_time": "15 minutes"
            },
            "AIR_FILTER": {
                "primary_action": f"Replace air intake paper filter element in {brand} {model} air cleaner box.",
                "steps": [
                    "1. Open front bonnet and locate black plastic air filter housing box.",
                    "2. Unsnap the 3-4 metal or plastic retaining spring clips holding the air box lid.",
                    "3. Lift upper air box lid and slide out the dirty, dust-choked air filter element.",
                    "4. Use a clean vacuum or dry cloth to wipe out loose leaves and dust from bottom of air box.",
                    "5. Seat the new rectangular air filter element firmly into the rubber perimeter channel.",
                    "6. Re-engage air box cover lid and snap all spring clips securely shut."
                ],
                "tools_needed": ["None (Tool-less spring clips) or Phillips Screwdriver", "Clean Dry Cloth", "New Air Filter Element"],
                "cost_estimate": "₹350 - ₹700 (New filter element)",
                "estimated_time": "5 - 10 minutes (Easy DIY)"
            },
            "CABIN_AC_FILTER": {
                "primary_action": f"Replace cabin pollen & AC filter element behind glovebox of {brand} {model}.",
                "steps": [
                    "1. Empty glovebox contents and open glovebox door fully.",
                    "2. Squeeze side damper stopper tabs inwards or unscrew side damper pin to let glovebox swing down.",
                    "3. Locate horizontal rectangular plastic filter access door on HVAC blower casing.",
                    "4. Pinch locking clip on right side and remove access door.",
                    "5. Slide out old clogged cabin filter; note the printed AIRFLOW DIRECTION arrow (usually pointing down).",
                    "6. Slide in new activated-carbon cabin filter matching arrow direction and snap access door shut."
                ],
                "tools_needed": ["None (Hand operation)", "Flashlight", "New Activated Carbon Cabin Filter"],
                "cost_estimate": "₹350 - ₹650 (Filter element)",
                "estimated_time": "10 minutes (Easy DIY)"
            },
            "FUEL_FILTER": {
                "primary_action": f"Depressurize fuel line and replace fuel filter element on {brand} {model}.",
                "steps": [
                    "1. Remove fuel pump fuse from fuse box and crank engine 5 seconds to depressurize fuel lines.",
                    "2. Locate fuel filter (underbody near fuel tank or in engine bay).",
                    "3. Wrap fuel line fittings in a shop rag to absorb residual fuel seepage.",
                    "4. Depress quick-connect plastic release tabs and slide fuel lines off filter nipples.",
                    "5. Loosen mounting bracket clamp bolt and slide out old filter.",
                    "6. Install new filter verifying arrow points toward engine; click fuel lines firmly until audible click.",
                    "7. Reinsert fuel pump fuse, turn ignition ON for 5 seconds 3 times to prime system, then start engine."
                ],
                "tools_needed": ["Quick-Disconnect Fuel Line Tool / Pliers", "10mm Socket Wrench", "Eye Protection", "Catch Basin", "New Fuel Filter"],
                "cost_estimate": "₹450 - ₹1,200 (Part) + ₹400 (Workshop labor)",
                "estimated_time": "25 - 35 minutes"
            },
            "SPARK_PLUGS": {
                "primary_action": f"Inspect, check gap, and replace ignition spark plugs on {brand} {model}.",
                "steps": [
                    "1. Ensure engine is stone cold. Disconnect negative 12V battery terminal.",
                    "2. Unplug electrical connectors from individual ignition coils on top of cylinder head.",
                    "3. Remove 10mm bolt securing each ignition coil and pull coils straight up.",
                    "4. Insert 16mm magnetic spark plug socket with extension bar and unscrew old spark plugs counter-clockwise.",
                    "5. Verify gap on new plugs with feeler gauge (typically 0.8 mm - 1.0 mm).",
                    "6. Thread new spark plugs in by hand first to prevent cross-threading.",
                    "7. Torque to 20-25 Nm using torque wrench. Reinstall coils, bolts, and connectors."
                ],
                "tools_needed": ["16mm Magnetic Spark Plug Socket", "Ratchet & Extension Bar", "10mm Socket", "Feeler Gauge", "Torque Wrench"],
                "cost_estimate": "₹600 - ₹1,800 (Set of spark plugs) vs ₹400 (Labor)",
                "estimated_time": "30 minutes"
            },
            "BRAKE_PADS_SHOES": {
                "primary_action": f"Install new front brake pad friction set and inspect disc rotors for {brand} {model}.",
                "steps": [
                    "1. Park on hard level ground, chock rear wheels, loosen lug nuts slightly, jack vehicle up, and set on axle stands.",
                    "2. Remove road wheel to expose brake caliper and disc rotor.",
                    "3. Unscrew bottom 14mm caliper slide pin bolt and pivot caliper body upward.",
                    "4. Slide worn friction pads out from stainless anti-rattle abutment clips.",
                    "5. Use C-clamp or caliper piston tool with an old pad to push caliper piston flush back into bore.",
                    "6. Apply thin film of high-temperature synthetic brake grease to pad metal backing ears (NOT friction face).",
                    "7. Insert new brake pads, swing caliper down, torque slide pin to 30 Nm, reinstall wheel, and pump brake pedal 5 times before driving."
                ],
                "tools_needed": ["Trolley Jack & Axle Stands", "14mm / 17mm Spanners", "C-Clamp / Piston Retractor", "High-Temp Brake Grease", "Brake Cleaner"],
                "cost_estimate": "₹1,500 - ₹3,500 (Set of front pads) + ₹500 (Labor)",
                "estimated_time": "45 minutes"
            },
            "BRAKE_FLUID": {
                "primary_action": f"Perform complete brake fluid hydraulic flush and bleeding with DOT 4 on {brand} {model}.",
                "steps": [
                    "1. Clean master cylinder reservoir cap thoroughly to prevent dirt ingress.",
                    "2. Siphon out old dark moisture-laden brake fluid from reservoir and fill with fresh DOT 4 fluid.",
                    "3. Attach clear vinyl tubing to caliper bleeder screw (starting at furthest wheel from master cylinder).",
                    "4. Have assistant pump brake pedal 3 times and hold firm pressure.",
                    "5. Open 8mm bleeder screw 1/4 turn to allow air bubbles and murky fluid to purge into catch bottle, then close screw.",
                    "6. Release pedal. Repeat process until crystal-clear fluid flows with zero air bubbles, keeping reservoir full throughout."
                ],
                "tools_needed": ["8mm / 10mm Bleeder Spanner", "Clear Vinyl Tubing (1/4 inch)", "Catch Bottle", "Turkey Baster / Syringe", "Fresh DOT 4 Fluid"],
                "cost_estimate": "₹400 - ₹800 (Fluid) + ₹500 (Workshop bleeding)",
                "estimated_time": "40 minutes"
            },
            "COOLANT": {
                "primary_action": f"Drain, flush, and refill radiator with 50/50 ethylene glycol coolant on {brand} {model}.",
                "steps": [
                    "1. CAUTION: Engine must be completely COLD. Opening hot radiator causes severe scalding steam burns.",
                    "2. Position drain basin under bottom radiator petcock drain valve.",
                    "3. Unscrew petcock thumb valve and remove radiator pressure cap to drain old spent coolant.",
                    "4. Close drain valve and refill radiator with distilled water and radiator flush chemical.",
                    "5. Run engine with cabin heater on HIGH for 10 minutes, let cool, and drain.",
                    "6. Refill with fresh 50/50 ethylene glycol premix coolant; squeeze upper hose to burp air bubbles, and fill expansion tank to MAX line."
                ],
                "tools_needed": ["Pliers / Screwdriver", "Large Drain Basin (10L)", "Funnel", "Fresh 50/50 Pre-diluted Coolant"],
                "cost_estimate": "₹600 - ₹1,400 (Coolant fluid)",
                "estimated_time": "30 - 45 minutes"
            },
            "TRANSMISSION_OIL": {
                "primary_action": f"Drain and refill transmission / gearbox oil to specified level on {brand} {model}.",
                "steps": [
                    "1. Drive vehicle 5 minutes to warm gear oil, then raise vehicle on 4 level jack stands.",
                    "2. CRITICAL: Always remove transmission FILL plug first to guarantee you can refill it!",
                    "3. Position drain pan and remove bottom transmission DRAIN plug.",
                    "4. Clean magnetic drain plug of metallic shavings and install new copper/aluminum crush washer.",
                    "5. Torque drain plug to 35 Nm.",
                    "6. Use fluid pump or flex hose with funnel to fill fresh manual gear oil (e.g. 75W-90 GL-4) or ATF through fill port until oil begins to weep out of hole. Reinstall fill plug."
                ],
                "tools_needed": ["17mm / 24mm Sockets / Allen Hex Key", "Hand Fluid Transfer Pump", "Level Jack Stands", "Drain Basin"],
                "cost_estimate": "₹1,200 - ₹3,500 (Fluid) + ₹600 (Workshop labor)",
                "estimated_time": "45 minutes"
            },
            "WIPER_BLADES": {
                "primary_action": f"Replace driver and passenger windshield wiper blades on {brand} {model}.",
                "steps": [
                    "1. Lift windshield wiper arm away from glass until it locks in upright service position.",
                    "2. Place a folded microfiber towel on glass beneath wiper arm as insurance against arm snapping down.",
                    "3. Locate the small plastic release tab on the J-hook connector joint.",
                    "4. Depress release tab and slide the old wiper blade assembly downward toward windshield base to disconnect.",
                    "5. Slide new frameless or bracketed blade into J-hook slot until you hear a sharp, solid 'CLICK'.",
                    "6. Remove protective yellow/green plastic squeegee guard and gently lower arm back onto glass."
                ],
                "tools_needed": ["Zero Tools Needed (Hand operation)", "Folded Microfiber Towel", "Pair of New Wiper Blades"],
                "cost_estimate": "₹300 - ₹800 (Pair of wiper blades)",
                "estimated_time": "5 minutes (Pure DIY)"
            },
            "BATTERY_ELECTRICAL": {
                "primary_action": f"Clean terminals and replace 12V automotive starter battery on {brand} {model}.",
                "steps": [
                    "1. Ensure ignition is OFF and keys are out of vehicle.",
                    "2. Loosen negative (-) 10mm terminal clamp nut first; wiggle and remove cable to chassis ground.",
                    "3. Loosen positive (+) 10mm terminal clamp nut and remove cable.",
                    "4. Unbolt battery hold-down clamp bracket (typically 10mm or 12mm nuts).",
                    "5. Lift old battery out by handles (wear gloves; heavy unit ~15 kg).",
                    "6. Clean cable terminal clamps using wire brush and baking soda solution.",
                    "7. Set new battery into tray, bolt hold-down clamp, connect positive (+) first, then negative (-), and coat posts with petroleum jelly."
                ],
                "tools_needed": ["10mm / 12mm Spanner / Deep Socket", "Wire Terminal Brush", "Baking Soda Solution", "Petroleum Jelly / Grease"],
                "cost_estimate": "₹4,000 - ₹6,500 (New battery with exchange rebate)",
                "estimated_time": "15 - 20 minutes"
            },
            "DRIVE_BELTS": {
                "primary_action": f"Inspect tensioner pulley and replace serpentine accessory drive belt on {brand} {model}.",
                "steps": [
                    "1. Draw or photograph engine belt routing diagram across alternator, water pump, and AC compressor.",
                    "2. Place long breaker bar or 14mm/15mm serpentine belt tool onto automatic tensioner pulley center bolt.",
                    "3. Rotate tensioner in direction of spring relief to slacken belt.",
                    "4. Slip old belt off smooth water pump pulley first, then remove from grooved pulleys.",
                    "5. Spin each idler and alternator pulley by hand; check for gritty bearings, wobbles, or noise.",
                    "6. Route new belt around all grooved pulleys according to diagram, relieve tensioner, slide belt over smooth pulley, and release tensioner."
                ],
                "tools_needed": ["Serpentine Belt Tool / Breaker Bar", "14mm / 15mm Sockets", "Flashlight", "New OEM Serpentine Belt"],
                "cost_estimate": "₹600 - ₹1,800 (Serpentine) vs ₹4,500+ (Timing belt kit at workshop)",
                "estimated_time": "30 mins (Serpentine) / 3 hours (Timing belt at workshop)"
            },
            "TYRES": {
                "primary_action": f"Rotate tyres, balance wheels, and adjust computerized 3D wheel alignment for {brand} {model}.",
                "steps": [
                    "1. Check tyre tread depth across center and shoulders using depth gauge; if < 2mm, replace tyres in pairs.",
                    "2. Have tyre center mount wheels on dynamic spin balancer and apply clip-on / adhesive balance weights.",
                    "3. Mount vehicle on computerized 3D alignment rack; measure camber, caster, and toe angles.",
                    "4. Technician will adjust front tie-rod turnbuckles to bring toe angles to 0.0° factory spec to stop inner tyre scrubbing.",
                    "5. Inflate all tyres to OEM cold pressure (typically 32-35 PSI) and reset TPMS system."
                ],
                "tools_needed": ["Computerized 3D Alignment Rig & Dynamic Spin Balancer (Tyre Workshop)"],
                "cost_estimate": "₹350 - ₹500 (Alignment & Balancing) vs ₹4,000 - ₹7,500 per new tyre",
                "estimated_time": "30 minutes (Workshop)"
            },
            "CLUTCH_COMPONENTS": {
                "primary_action": f"Replace complete clutch kit (friction disc, pressure plate & release bearing) for {brand} {model}.",
                "steps": [
                    "1. Vehicle requires hoist; disconnect battery, starter motor, gear shift linkages, and driveshafts.",
                    "2. Support engine with crossbar fixture; unbolt bellhousing and lower transmission gearbox using transmission jack.",
                    "3. Unbolt clutch pressure plate in star pattern; remove worn friction disc.",
                    "4. Inspect engine flywheel face for thermal heat cracking or deep grooves; skim flywheel if needed.",
                    "5. Center new clutch disc on flywheel using alignment arbor tool; torque pressure plate bolts to 25 Nm.",
                    "6. Replace clutch release throwout bearing on gearbox input shaft quill; reinstall transmission and bleed hydraulic clutch line."
                ],
                "tools_needed": ["Transmission Jack", "Engine Support Bar", "Clutch Alignment Spline Tool", "Torque Wrench (Workshop)"],
                "cost_estimate": "₹4,500 - ₹11,000 (Complete 3-piece OEM clutch kit + transmission labor)",
                "estimated_time": "4 - 5 hours at workshop"
            },
            "SUSPENSION_COMPONENTS": {
                "primary_action": f"Replace worn suspension strut cartridges, top mounts, and control arm bushes for {brand} {model}.",
                "steps": [
                    "1. Raise vehicle on lift; disconnect sway bar end link and ABS wheel speed sensor wiring.",
                    "2. Unbolt lower strut knuckle clevis bolts and top strut tower nuts; remove MacPherson strut assembly.",
                    "3. DANGER: Compress coil spring using hydraulic strut spring compressor before unbolting top center nut.",
                    "4. Install new hydraulic/gas strut shock absorber, new polyurethane bump stop, and top mount bearing.",
                    "5. Press out cracked lower control arm rubber bushes and press in new OEM polyurethane bushings.",
                    "6. Reassemble strut, torque knuckle bolts to 110 Nm, and perform mandatory front wheel alignment."
                ],
                "tools_needed": ["Hydraulic Strut Spring Compressor", "Ball Joint Separator", "Heavy Impact Wrench", "Hydraulic Bushing Press"],
                "cost_estimate": "₹2,500 - ₹7,000 (Struts / bushing set + workshop labor)",
                "estimated_time": "2 - 3 hours at workshop"
            },
            "LIGHTS_FUSES": {
                "primary_action": f"Replace blown filament bulb or mini blade fuse on {brand} {model}.",
                "steps": [
                    "1. Switch vehicle ignition OFF and turn lighting switches to OFF.",
                    "2. For Bulb: Access rear of headlamp or tail lamp housing from engine bay or boot trim panel.",
                    "3. Unclip electrical wire plug; rotate bulb socket 1/4 turn counter-clockwise or release metal spring clip.",
                    "4. Remove old bulb. Insert new halogen/LED bulb (Do NOT touch quartz glass with bare fingers; finger oils cause premature bulb rupture!).",
                    "5. For Fuse: Open interior fuse panel under dashboard or under bonnet.",
                    "6. Use white plastic fuse puller tool to extract suspected blown fuse. Inspect metal filament wire.",
                    "7. Replace only with an identical color and ampere-rated fuse (e.g., 10A Red, 15A Blue, 20A Yellow)."
                ],
                "tools_needed": ["Plastic Fuse Puller", "Clean Tissue / Nitrile Gloves (for bulbs)", "New Bulb / Mini Blade Fuse"],
                "cost_estimate": "₹50 - ₹250 (Bulb / pack of blade fuses)",
                "estimated_time": "5 - 10 minutes (Easy DIY)"
            }
        }

        if issue_key in plans:
            return plans[issue_key]

        # Fallback to general workshop plan
        return {
            "primary_action": f"Perform comprehensive diagnostic inspection of {issue_key.replace('_', ' ').title()} on {brand} {model}.",
            "steps": [
                "1. Connect OBD-II diagnostic scanner to 16-pin port under driver dashboard.",
                "2. Read active and pending diagnostic trouble codes (DTCs) across Engine, ABS, and BCM modules.",
                "3. Perform physical inspection of associated wiring harnesses, mechanical linkages, and fluid levels.",
                "4. Clear fault codes after repairs and verify with a 10 km road test."
            ],
            "tools_needed": ["OBD-II Diagnostic Scanner", "Standard Workshop Hand Tools"],
            "cost_estimate": "₹400 - ₹800 (Diagnostic Scan & Basic Inspection)",
            "estimated_time": "30 minutes"
        }
