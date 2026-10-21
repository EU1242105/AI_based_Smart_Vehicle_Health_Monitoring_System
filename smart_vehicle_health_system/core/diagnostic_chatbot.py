"""
AI Automotive Diagnostic Chatbot Engine
Processes natural language driver complaints, acoustic sound descriptions,
and operational symptoms to diagnose root causes and generate step-by-step repair guides.

Grounded in:
- TCS: Lexical Tokenization, Regex Pattern Matchers, Dialogue State Automaton (DFA)
- Soft Computing: Fuzzy Confidence & Symptom Match Degree (μ)
- AIML: Intent Classification & Expert Diagnostic Tree
"""

import re
from typing import Dict, List, Any, Optional, Tuple


class DiagnosticChatbotEngine:
    # --------------------------------------------------------------------------
    # 1. COMPREHENSIVE AUTOMOTIVE DIAGNOSTIC KNOWLEDGE REPOSITORY
    # --------------------------------------------------------------------------
    DIAGNOSTIC_CASES = [
        {
            "id": "BRAKE_PAD_SQUEAL",
            "title": "Brake Pad Wear Indicator Squeal",
            "subsystem": "Braking System",
            "sound_keywords": ["squeal", "squeak", "screech", "screeching", "whine", "chirp", "high-pitched", "awaaz", "sound", "noise"],
            "component_keywords": ["brake", "brakes", "braking", "pedal", "pad", "pads", "stopping", "slow down"],
            "context_keywords": ["press", "apply", "push", "touch", "light", "morning"],
            "faulty_part": "Front / Rear Brake Pads (Acoustic Wear Tab)",
            "severity": "HIGH (Inspect within 150 km)",
            "safe_to_drive": "Yes, for short low-speed trips to a repair shop, but avoid high speeds or hard emergency braking.",
            "root_cause": "The brake friction lining has worn down to approximately 2.5 - 3 mm. A built-in metal acoustic wear clip is deliberately scraping against the steel rotor to produce a loud high-pitched screech to alert you before bare backing-plate metal ruins the rotor.",
            "troubleshooting": [
                "Turn front steering wheel fully to one side and look through rim spokes with a torch.",
                "Check pad thickness: If the friction pad is thinner than the metal backing plate (< 3mm), replacement is urgent.",
                "Gently feel the brake disc rotor (when stone-cold): If you feel deep circular grooves, rotors may also need resurfacing."
            ],
            "replacement_guide": {
                "difficulty": "Moderate DIY",
                "estimated_time": "45 - 60 minutes",
                "cost_est_inr": "₹1,200 - ₹2,800 (Pads only) vs ₹4,500+ (Dealership)",
                "tools_required": ["Floor Jack & Jack Stands", "Lug Nut Wrench", "12mm & 14mm Sockets / Ratchet", "C-Clamp or Caliper Retractor", "Brake Cleaner", "Anti-Seize Brake Grease"],
                "steps": [
                    "Loosen wheel lug nuts half a turn, jack up car securely on jack stands, and remove wheel.",
                    "Unbolt the lower caliper guide pin bolt (usually 14mm) and pivot the caliper upward on its top pin.",
                    "Slide out the worn brake pads and stainless steel abutment clips.",
                    "Clean caliper bracket grooves with brake cleaner and snap new clips into place.",
                    "Use a C-clamp over an old pad to slowly compress the round caliper piston fully into its bore.",
                    "Apply high-temp brake grease sparingly to the steel backing plate ears (never on friction face!).",
                    "Install new pads, swing caliper down, torque guide pin bolt (approx. 32-35 Nm), and remount wheel.",
                    "CRITICAL: Pump the brake pedal 5-6 times until firm BEFORE turning on the engine or driving."
                ]
            }
        },
        {
            "id": "BRAKE_ROTOR_GRIND",
            "title": "Severe Metal-on-Metal Brake Grinding",
            "subsystem": "Braking System",
            "sound_keywords": ["grind", "grinding", "metal", "scraping", "harsh", "crunch", "karkash"],
            "component_keywords": ["brake", "brakes", "braking", "wheel", "pedal", "rotor", "disc"],
            "context_keywords": ["press", "hard", "stop", "apply", "deceleration"],
            "faulty_part": "Brake Pads Completely Depleted + Scored Brake Rotor",
            "severity": "CRITICAL EMERGENCY",
            "safe_to_drive": "NO. Braking distance is doubled and calipers risk seizing or losing hydraulic brake pressure.",
            "root_cause": "The friction lining has completely vanished (0 mm remaining). The bare steel backing plate is directly grinding against the cast iron brake rotor, creating iron gouges and extreme friction heat.",
            "troubleshooting": [
                "Inspect the brake disc: You will likely see bright rough circular gouges or heavy metallic shavings on the wheel rim.",
                "Check brake fluid reservoir: Fluid level will be low as caliper pistons are over-extended."
            ],
            "replacement_guide": {
                "difficulty": "Moderate DIY",
                "estimated_time": "1.5 hours",
                "cost_est_inr": "₹3,500 - ₹6,000 (New Pads + Pair of Disc Rotors)",
                "tools_required": ["Jack & Stands", "Lug Wrench", "14mm & 17mm Sockets", "Impact Screwdriver (for rotor screw)", "Brake Piston Tool", "New Rotors & Pads"],
                "steps": [
                    "Remove wheel and completely unbolt caliper and caliper mounting bracket (usually 17mm high-torque bolts).",
                    "Hang caliper from suspension spring using wire so hydraulic rubber brake hose does not stretch.",
                    "Remove rotor retaining Phillips countersunk screw (use impact screwdriver if rusted).",
                    "Slide off the damaged scored rotor; clean wheel hub face with wire brush.",
                    "Degrease brand new rotor with brake cleaner to strip factory packing oil.",
                    "Mount new rotor, reinstall caliper bracket, insert new pads with shims, compress piston, bolt caliper, and pump brake pedal."
                ]
            }
        },
        {
            "id": "ENGINE_KNOCKING_LUBRICATION",
            "title": "Engine Rod Knock / Low Oil Pressure Tap",
            "subsystem": "Engine Mechanical / Lubrication",
            "sound_keywords": ["knock", "knocking", "tap", "tapping", "tick", "ticking", "hammer", "rattle", "thud", "tik", "sound"],
            "component_keywords": ["engine", "motor", "hood", "bonnet", "oil", "rpm", "accelerator"],
            "context_keywords": ["speed", "rev", "idle", "cold", "hot", "fast", "speed up"],
            "faulty_part": "Crankshaft Connecting Rod Bearings / Hydraulic Valve Tappets",
            "severity": "CRITICAL MECHANICAL HAZARD",
            "safe_to_drive": "NO. Pull over and turn off the engine immediately. Continuing to drive will throw a rod through the engine block.",
            "root_cause": "Severe oil starvation or degraded oil viscosity has eliminated the hydrodynamic oil film. The piston connecting rod bearing is slapping directly against the crankshaft journal at every combustion stroke.",
            "troubleshooting": [
                "Pull over safely and turn off engine immediately.",
                "Wait 5 minutes for hot oil to settle into the sump, then pull the yellow engine oil dipstick.",
                "Wipe dipstick with clean cloth, re-insert fully, and pull again: Is the oil below the MIN dot?",
                "Look at oil texture: If it has shiny metallic glitter or smells burnt/charred, bearings are failing."
            ],
            "replacement_guide": {
                "difficulty": "Advanced / Professional Workshop",
                "estimated_time": "1 - 2 Days",
                "cost_est_inr": "₹12,000 - ₹35,000 (Engine lower-end overhaul / crank polish)",
                "tools_required": ["Engine Hoist", "Precision Micrometer / Plastigauge", "Torque Angle Gauge", "Oil Pan Sealant", "New Rod Bearings & Fasteners"],
                "steps": [
                    "If oil is simply low, top up immediately with manufacturer-spec oil (e.g. 5W-30 / 0W-20) to see if noise subsides.",
                    "If knocking persists under revs, engine lower pan must be dropped.",
                    "Inspect oil pickup strainer for plastic timing guide debris or bearing bronze particles.",
                    "Unbolt connecting rod end-caps, measure journal ovality, and install new oversized/standard bearing shells.",
                    "Install new high-volume oil pump and flush all internal oil galleries."
                ]
            }
        },
        {
            "id": "BELT_SQUEAL_ACCESSORY",
            "title": "Serpentine Accessory Drive Belt Slip",
            "subsystem": "Auxiliary Electrical & Cooling Drive",
            "sound_keywords": ["squeak", "squeal", "screech", "screaming", "chirp", "whistle", "shriek"],
            "component_keywords": ["belt", "fan belt", "serpentine", "alternator", "ac", "compressor", "pulley"],
            "context_keywords": ["start", "morning", "cold", "turn on", "ac on", "rain", "water", "steering lock"],
            "faulty_part": "Serpentine Drive Belt / Automatic Belt Tensioner",
            "severity": "MEDIUM",
            "safe_to_drive": "Yes for short distances, but if the belt snaps, alternator charging, water pump cooling, and AC immediately fail.",
            "root_cause": "The EPDM rubber accessory belt has stretched, oxidized, or lost tension from a weakened tensioner spring. Turning on AC or turning power steering adds mechanical resistance, causing the belt to slip over pulleys at high RPM.",
            "troubleshooting": [
                "With engine OFF, shine flashlight on belt ribs: look for cracks spaced closer than 1/2 inch apart or missing chunks.",
                "Twist the belt between thumb and index finger halfway between pulleys: if it twists over 90°, belt is excessively slack.",
                "Listen closely: Does the screech happen right when the AC clutch clicks ON?"
            ],
            "replacement_guide": {
                "difficulty": "Easy to Moderate DIY",
                "estimated_time": "30 minutes",
                "cost_est_inr": "₹450 - ₹1,200 (Belt) + ₹1,500 (Tensioner if needed)",
                "tools_required": ["14mm Long Ratchet / Serpentine Belt Tool", "Camera / Smartphone (to snap routing)"],
                "steps": [
                    "Photograph the exact belt routing serpentine path around alternator, AC, and crank pulleys.",
                    "Fit socket on center bolt of tensioner pulley and rotate clockwise against spring pressure to release tension.",
                    "Slip old belt off the top alternator pulley and let tensioner relax.",
                    "Spin idler pulleys by hand: ensure they rotate silky-smooth with zero gritty noise or axial play.",
                    "Route new belt along pulleys matching diagram; hold tensioner open and slip over smooth water pump pulley last.",
                    "Ensure all multi-rib grooves sit dead-center in each grooved pulley track before starting."
                ]
            }
        },
        {
            "id": "STEERING_HIGH_SPEED_VIBRATION",
            "title": "Front Wheel Dynamic Imbalance / Rotor DTV",
            "subsystem": "Steering, Wheels & Suspension",
            "sound_keywords": ["vibrate", "vibration", "shake", "shaking", "wobble", "shudder", "tremble", "hilna"],
            "component_keywords": ["steering", "wheel", "handle", "dash", "highway", "speed", "front"],
            "context_keywords": ["80", "90", "100", "110", "high speed", "highway", "braking", "cruising"],
            "faulty_part": "Front Wheel Balancing Weights Lost or Warped Brake Rotors",
            "severity": "MEDIUM",
            "safe_to_drive": "Yes, but prolonged vibration damages wheel bearings, tie rods, and causes irregular tire cupping.",
            "root_cause": "If shaking occurs continuously at 80-110 km/h: An adhesive wheel balance weight fell off the rim, causing centrifugal force imbalance. If shaking occurs strictly when depressing the brake pedal: The front brake discs have Disc Thickness Variation (warpage).",
            "troubleshooting": [
                "Drive at 90 km/h on a smooth flat road: If steering wheel oscillates side-to-side without touching the brakes = Wheel Balancing.",
                "Apply brakes at 90 km/h: If the steering wheel pulses violently in your hands only during braking = Warped Front Rotors.",
                "Inspect front tires for bulges, uneven tread wear, or bent rim lip."
            ],
            "replacement_guide": {
                "difficulty": "Easy (Tyre Shop) or Moderate (Rotor Swap)",
                "estimated_time": "20 - 40 minutes",
                "cost_est_inr": "₹350 - ₹600 (Dynamic Wheel Balancing & Alignment)",
                "tools_required": ["Dynamic Spin Balancer Machine", "Lead / Zinc Clip-on Weights"],
                "steps": [
                    "Visit an authorized tire center for automated 3D dynamic wheel balancing.",
                    "Technicians mount each wheel on high-speed spin balancer to locate exact heavy and light spots.",
                    "Counter-weights (10g - 40g) are affixed to inside and outside rim barrels to offset harmonic resonance.",
                    "If rotors are warped: Replace both front brake disc rotors together."
                ]
            }
        },
        {
            "id": "EXHAUST_WHITE_SMOKE_OVERHEAT",
            "title": "Blown Cylinder Head Gasket",
            "subsystem": "Engine Thermal Management",
            "sound_keywords": ["smoke", "steam", "white smoke", "sweet smell", "bubble", "hiss", "boil"],
            "component_keywords": ["exhaust", "silencer", "coolant", "radiator", "temperature", "temp", "overheating"],
            "context_keywords": ["hot", "gauge", "needle", "reservoir", "expansion tank"],
            "faulty_part": "Multi-Layer Steel Cylinder Head Gasket",
            "severity": "CRITICAL EMERGENCY",
            "safe_to_drive": "NO. Continued driving causes catastrophic hydraulic engine lock (hydro-lock) or cracked cylinder head.",
            "root_cause": "The seal between the engine block and cylinder head has breached. High-pressure cooling passages are leaking pressurized coolant into the combustion cylinders, where it vaporizes into billowing white exhaust clouds.",
            "troubleshooting": [
                "Observe exhaust smoke: Does it linger like thick white fog and smell sugary/sweet? (Ethylene glycol coolant vapor).",
                "Look inside coolant expansion reservoir while idling (careful - cold engine only!): Constant bubbling streams indicate exhaust gas pushing into cooling jacket.",
                "Inspect oil dipstick and underside of oil filler cap: Yellow-milky frothy emulsion confirms coolant mixing with oil."
            ],
            "replacement_guide": {
                "difficulty": "Advanced Major Mechanical Job",
                "estimated_time": "2 - 3 Days",
                "cost_est_inr": "₹15,000 - ₹32,000 (Gasket kit, head surface skim machining, stretch bolts, oil & coolant flush)",
                "tools_required": ["Cylinder Head Angle Torque Wrench", "Camshaft Locking Tool", "Straightedge & Feeler Gauge", "New Multi-Layer Head Gasket"],
                "steps": [
                    "Drain all coolant and engine oil; disconnect 12V battery and fuel pressure lines.",
                    "Remove intake manifold, exhaust manifold, and timing belt/chain timing kit.",
                    "Remove cylinder head bolts in reverse sequence to prevent warp; lift off cylinder head.",
                    "Check head surface flatness with precision steel straightedge: if warpage > 0.05 mm, send for CNC machine deck resurfacing.",
                    "Clean engine block deck thoroughly of carbon; fit new gasket dry with zero silicone.",
                    "Install brand new head bolts and torque in 3-stage angle torque sequence (e.g. 35 Nm + 90° + 90°)."
                ]
            }
        },
        {
            "id": "CV_AXLE_CLICKING_TURN",
            "title": "Worn CV (Constant Velocity) Joint Axle",
            "subsystem": "Drivetrain & Axles",
            "sound_keywords": ["click", "clicking", "clack", "pop", "popping", "snap", "snapping", "sound"],
            "component_keywords": ["turn", "turning", "steering", "wheel", "corner", "u-turn", "front"],
            "context_keywords": ["sharp turn", "left", "right", "lock", "accelerating in turn"],
            "faulty_part": "Outer Constant Velocity (CV) Joint / Axle Half-Shaft",
            "severity": "HIGH (Replace within 300 km)",
            "safe_to_drive": "Drive gently for immediate repair. If the joint disintegrates, wheel drive is lost completely and car cannot move.",
            "root_cause": "The accordion rubber CV boot tore or cracked, allowing protective molybdenum disulfide grease to fling out while road grit and water entered, rapidly grinding down the precision ball bearings and spider cage.",
            "troubleshooting": [
                "Look behind the front wheels at the rubber CV drive axle boots: Is there dark sticky black grease splattered all over the wheel well?",
                "Drive in a tight circle in an empty parking lot with steering turned fully to lock: Rhythmic clicking that speeds up with gas confirms outer CV joint failure."
            ],
            "replacement_guide": {
                "difficulty": "Moderate to Advanced",
                "estimated_time": "1.5 hours",
                "cost_est_inr": "₹2,500 - ₹5,500 (Complete CV Axle Shaft assembly)",
                "tools_required": ["32mm Axle Nut Socket", "Heavy Breaker Bar", "Ball Joint Separator", "Transmission Drain Pan"],
                "steps": [
                    "Remove axle nut cotter pin and loosen massive 32mm center axle nut with breaker bar before lifting car.",
                    "Jack vehicle up, remove wheel, and unbolt lower ball joint or strut mount bolts.",
                    "Pull wheel hub knuckle outward to slide CV axle splines out of front wheel bearing hub.",
                    "Pry inner CV joint out of transmission gearbox case with small pry bar (catch any leaking transmission fluid).",
                    "Snap new complete half-shaft axle into transmission until lock ring clicks firmly.",
                    "Insert outer splines into wheel hub, reinstall ball joint, torque axle nut to spec (usually 180-220 Nm), and re-stake nut collar."
                ]
            }
        },
        {
            "id": "AC_WARM_AIR_HISSING",
            "title": "AC Refrigerant Leak & Clogged Cabin Air Filter",
            "subsystem": "HVAC / Climate Control",
            "sound_keywords": ["hiss", "hissing", "gasp", "whistle", "no air", "weak air", "garam"],
            "component_keywords": ["ac", "air conditioner", "air conditioning", "cooling", "warm air", "cabin", "blower", "vent"],
            "context_keywords": ["not cooling", "warm", "hot", "dash", "dashboard", "glovebox"],
            "faulty_part": "AC System Refrigerant (R134a) Low & Cabin Pollen Filter Blocked",
            "severity": "LOW (Comfort issue)",
            "safe_to_drive": "Yes, vehicle is completely safe mechanically.",
            "root_cause": "The expansion valve is starved of liquid refrigerant due to a micro-leak in the condenser or O-rings, producing a continuous hissing sound from the dashboard vents, while a choked cabin filter restricts blower airflow.",
            "troubleshooting": [
                "With engine idling and AC button ON, open hood: Does the AC compressor center clutch spin, or is it stationary? (Low pressure switch disables clutch when gas is empty).",
                "Look at the front radiator/condenser: Check for dark damp oily patches where refrigerant oil leaked out.",
                "Open glovebox and drop cabin filter: If blackened with dust and leaves, airflow drops by up to 70%."
            ],
            "replacement_guide": {
                "difficulty": "Cabin Filter: 5 min DIY (No tools) | Gas Refill: AC Shop",
                "estimated_time": "15 to 45 mins",
                "cost_est_inr": "₹350 (Filter) | ₹1,500 - ₹2,200 (AC vacuum, leak test & gas recharge)",
                "tools_required": ["None for Cabin Filter", "AC Manifold Gauge, Vacuum Pump, R134a Can"],
                "steps": [
                    "Open passenger glovebox, release the damper cord on right side, and squeeze side tabs inward to drop glovebox.",
                    "Unclip the rectangular filter cover and slide out the old clogged cabin filter.",
                    "Vacuum out dry leaves or dust from the blower cavity.",
                    "Slide in new activated-carbon cabin filter with the AIRFLOW arrow pointing DOWN.",
                    "Take vehicle to AC workshop for automated nitrogen leak testing, vacuum pull (to remove moisture), and recharge of exact factory gram weight."
                ]
            }
        },
        {
            "id": "WIPER_BLADES_STREAKING",
            "title": "Windshield Wiper Blades Worn / Cracked Rubber Squeegee",
            "subsystem": "Visibility & Safety Equipment",
            "sound_keywords": ["chatter", "chattering", "squeak", "rubbing", "wipe"],
            "component_keywords": ["wiper", "wipers", "wiper blade", "wiper blades", "windshield", "glass", "rain", "washer"],
            "context_keywords": ["streaking", "smearing", "lines", "water film", "skipping", "cracked rubber", "bad wipe"],
            "faulty_part": "Windshield Wiper Blade Rubber Squeegee",
            "severity": "LOW (Visibility safety issue)",
            "safe_to_drive": "Yes in dry weather, but hazardous in heavy rain or monsoon conditions.",
            "root_cause": "The wiper squeegee natural rubber has oxidized from UV sun exposure and engine heat, developing micro-fissures and edge tears that skip water across the windshield.",
            "troubleshooting": [
                "Lift wiper arm and slide thumb along rubber edge: Feel for splits, tears, or hardened brittle rubber.",
                "Turn on windshield washer spray: Watch if wiper leaves distinct unswept bands or broad smearing.",
                "Check J-hook connector joint: Ensure the plastic clip is firmly seated without loose play."
            ],
            "replacement_guide": {
                "difficulty": "5 Minutes Easy DIY (Zero tools required)",
                "diy_possible": True,
                "parts": "Pair of OEM Frameless / Hybrid Wiper Blades (₹300 - ₹800)",
                "tools": ["None (Hand operation)"],
                "steps": [
                    "Lift wiper arm upright into lock position; place a towel on glass as safety cushion.",
                    "Depress the small plastic release tab on the J-hook connector.",
                    "Slide old blade down towards glass to unhook.",
                    "Slide new blade into J-hook until an audible 'click' locks it.",
                    "Remove protective plastic squeegee guard and lower arm gently onto windshield."
                ]
            }
        },
        {
            "id": "CLUTCH_SLIPPING_WEAR",
            "title": "Manual Clutch Friction Disc Slipping & Thermal Glazing",
            "subsystem": "Transmission & Clutch System",
            "sound_keywords": ["racing", "engine revs", "screaming", "whine"],
            "component_keywords": ["clutch", "clutch plate", "pressure plate", "gear", "gearbox"],
            "context_keywords": ["slipping", "slip", "high bite", "rpm increases without speed", "pedal hard", "burning smell", "judder"],
            "faulty_part": "Clutch Friction Disc & Pressure Plate",
            "severity": "HIGH (Imminent breakdown risk)",
            "safe_to_drive": "Drive only short distances under light throttle to workshop; avoid steep inclines.",
            "root_cause": "The friction lining on the clutch plate has worn down below its minimum thickness, losing friction bite against the flywheel and slipping under engine torque.",
            "troubleshooting": [
                "Clutch Slip Test: In 3rd gear at 40 km/h, floor accelerator. If engine RPM climbs rapidly without road speed increasing, clutch is slipping.",
                "Observe bite point: If clutch only grabs at the absolute top of pedal release, friction disc is thin.",
                "Sniff after hill start: Strong pungent burning friction smell confirms clutch overheating."
            ],
            "replacement_guide": {
                "difficulty": "Hard (Requires workshop vehicle lift)",
                "diy_possible": False,
                "parts": "Complete 3-Piece OEM Clutch Kit (Disc, Cover, Release Bearing) (₹4,500 - ₹11,000)",
                "tools": ["Transmission Jack", "Clutch Alignment Arbor", "Torque Wrench"],
                "steps": [
                    "Hoist vehicle and remove driveshafts, starter motor, and shift cables.",
                    "Unbolt bellhousing and lower transmission transaxle using jack.",
                    "Remove old pressure plate and worn friction disc; inspect flywheel.",
                    "Align new clutch disc with arbor tool; torque pressure plate in star pattern to 25 Nm.",
                    "Replace release throwout bearing and reassemble gearbox."
                ]
            }
        },
        {
            "id": "FUEL_FILTER_CHOKED",
            "title": "Fuel Filter Choked / Fuel Delivery Starvation",
            "subsystem": "Fuel Delivery System",
            "sound_keywords": ["stutter", "sputtering", "coughing"],
            "component_keywords": ["fuel", "fuel filter", "petrol", "diesel", "accelerator"],
            "context_keywords": ["jerking", "hesitation", "stutter under load", "loss of power", "uphill", "long crank", "fuel interval"],
            "faulty_part": "In-line / Cartridge Fuel Filter Element",
            "severity": "MEDIUM (Performance & drivability issue)",
            "safe_to_drive": "Yes, but drive gently to prevent fuel rail starvation and stalling.",
            "root_cause": "Fine sediment and rust particles from fuel tanks have clogged the internal paper pleats of the fuel filter, restricting flow rate at high engine loads.",
            "troubleshooting": [
                "Check service history: Has the fuel filter exceeded 35,000 - 40,000 km without replacement?",
                "Observe acceleration: Does the car pull normally at low speed but jerk or hesitate when revving past 3,500 RPM?",
                "Check starting time: Does engine take 4-5 seconds of continuous cranking to start when cold?"
            ],
            "replacement_guide": {
                "difficulty": "Moderate (25 minutes)",
                "diy_possible": True,
                "parts": "OEM Fuel Filter Cartridge (₹450 - ₹1,200)",
                "tools": ["Quick-disconnect line pliers", "10mm socket", "Clean catch rag"],
                "steps": [
                    "Remove fuel pump fuse and crank 5 seconds to depressurize line.",
                    "Wrap disconnect joints in rag; depress release tabs and slide lines off.",
                    "Unbolt filter bracket and replace with new filter matching flow direction arrow.",
                    "Reinstall fuse and cycle ignition key 3 times to prime lines before cranking."
                ]
            }
        },
        {
            "id": "SPARK_PLUGS_MISFIRE",
            "title": "Spark Plug Electrode Wear & Cylinder Ignition Misfire",
            "subsystem": "Ignition System",
            "sound_keywords": ["misfire", "chug", "popping", "ticking"],
            "component_keywords": ["spark plug", "spark plugs", "plugs", "ignition", "coil", "cylinder"],
            "context_keywords": ["rough idle", "engine shakes", "misfire", "check engine light flashing", "mileage drop", "hesitation"],
            "faulty_part": "Ignition Spark Plugs (Electrode Gap Worn / Carbon Fouled)",
            "severity": "MEDIUM to HIGH (Can damage catalytic converter)",
            "safe_to_drive": "Drive directly to service center; avoid high speeds if Check Engine light flashes.",
            "root_cause": "The nickel or iridium center electrode has eroded over tens of thousands of kilometers, widening the spark gap and resulting in intermittent misfires (OBD code P0300).",
            "troubleshooting": [
                "Inspect idle stability: Does tachometer needle fluctuate and engine vibrate at traffic lights?",
                "Scan OBD-II port: Check for DTC P0301, P0302, P0303, P0304 (Cylinder misfire codes).",
                "Unscrew spark plugs: Look for heavy black carbon deposits, rounded electrodes, or wet fuel tips."
            ],
            "replacement_guide": {
                "difficulty": "Moderate (30 minutes DIY)",
                "diy_possible": True,
                "parts": "Set of 3 or 4 OEM Spark Plugs (₹600 - ₹1,800)",
                "tools": ["16mm Magnetic Spark Plug Socket", "Ratchet & Extension", "10mm Socket", "Feeler Gauge"],
                "steps": [
                    "Disconnect battery negative terminal; unplug coil connectors.",
                    "Unbolt 10mm ignition coil bolts and pull coils straight out.",
                    "Unscrew old spark plugs using magnetic socket.",
                    "Check gap on new plugs (0.8 - 1.0 mm); thread in by hand to prevent cross-threading.",
                    "Torque to 20-25 Nm and reconnect coil packs."
                ]
            }
        }
    ]

    # --------------------------------------------------------------------------
    # 2. NLP TOKENIZER & INTENT MATCHER (TCS + Soft Computing + AIML)
    # --------------------------------------------------------------------------
    @classmethod
    def process_query(cls, user_message: str) -> Dict[str, Any]:
        msg_clean = user_message.strip().lower()
        
        # Tokenize message into words and bigrams
        tokens = re.findall(r"[a-z0-9]+", msg_clean)
        token_set = set(tokens)

        # Quick greetings / casual queries
        if any(w in token_set for w in ["hi", "hello", "hey", "namaste", "help", "karan"]) and len(tokens) <= 3:
            return {
                "type": "GREETING",
                "bot_reply": (
                    "👋 **Hello! I am your AI Vehicle Diagnostic Assistant.**\n\n"
                    "Tell me what problem, unusual sound, or symptom your car is having! For example:\n"
                    "• *'My car is making a squealing sound when I press the brakes'*\n"
                    "• *'There is a metallic knocking sound coming from under the hood'*\n"
                    "• *'The steering wheel shakes whenever I drive above 80 km/h'*\n"
                    "• *'White smoke is coming from the exhaust and engine is overheating'*\n"
                    "• *'AC is blowing warm air with a hissing sound'* \n\n"
                    "What are you experiencing right now?"
                ),
                "quick_replies": [
                    "Squeaking brakes", "Engine knocking sound", "Steering vibration at 90 km/h",
                    "White smoke from exhaust", "Belt screech on start", "AC blowing warm air"
                ]
            }

        # Service interval general question
        if ("service" in token_set or "maintenance" in token_set) and ("when" in token_set or "how" in token_set or "interval" in token_set or "schedule" in token_set):
            return {
                "type": "SERVICE_ADVICE",
                "bot_reply": (
                    "🗓️ **Standard Vehicle Service Intervals:**\n\n"
                    "• **Engine Oil & Filter:** Every **10,000 km or 12 months** (Whichever comes first; 7,500 km for heavy city driving).\n"
                    "• **Air Intake & Cabin Filters:** Inspect every 10,000 km; replace every **15,000 – 20,000 km**.\n"
                    "• **Brake Pads:** Average lifespan **30,000 – 40,000 km** (inspect thickness every 10,000 km).\n"
                    "• **Radiator Coolant:** Flush and replace every **40,000 km or 24 months**.\n"
                    "• **12V Starter Battery:** Typically lasts **3 to 4 years**.\n\n"
                    "💡 *You can track your exact vehicle service countdown and part wear in the **Vehicle Service History** section!*"
                ),
                "quick_replies": ["Track my service date", "Check brake pad wear", "Squeaking sound when braking"]
            }

        # Multi-factor Match Evaluation (Fuzzy Soft Computing Scoring)
        best_match = None
        highest_score = 0.0
        detected_sound = None

        for case in cls.DIAGNOSTIC_CASES:
            # Sound score
            sound_hits = sum(1 for w in case["sound_keywords"] if w in msg_clean)
            # Component score
            comp_hits = sum(1 for w in case["component_keywords"] if w in msg_clean)
            # Context score
            ctx_hits = sum(1 for w in case["context_keywords"] if w in msg_clean)

            # Soft Computing weighted match degree
            score = (sound_hits * 3.0) + (comp_hits * 4.0) + (ctx_hits * 1.5)

            if score > highest_score:
                highest_score = score
                best_match = case
                # detect sound keyword
                for s in case["sound_keywords"]:
                    if s in msg_clean:
                        detected_sound = s
                        break

        # If match score is high enough (>= 3.0)
        if best_match and highest_score >= 3.0:
            rep = best_match["replacement_guide"]
            trouble_text = "\n".join([f"  {step}" for step in best_match["troubleshooting"]])
            steps_text = "\n".join([f"  {i+1}. {s}" for i, s in enumerate(rep["steps"])])
            tools_text = ", ".join(rep["tools_required"])

            bot_reply = (
                f"🩺 **AI Diagnostic Findings:**\n"
                f"• **Condition Detected:** {best_match['title']}\n"
                f"• **Subsystem:** {best_match['subsystem']}\n"
                f"• **Suspected Faulty Part:** ⚠️ **{best_match['faulty_part']}**\n"
                f"• **Severity:** {best_match['severity']}\n"
                f"• **Safe to Continue Driving?** {best_match['safe_to_drive']}\n\n"
                f"🔍 **Root Cause Explanation:**\n"
                f"{best_match['root_cause']}\n\n"
                f"📋 **Immediate Troubleshooting & Inspection Steps:**\n"
                f"{trouble_text}\n\n"
                f"🛠️ **Step-by-Step DIY Part Replacement Manual:**\n"
                f"• **Difficulty:** {rep['difficulty']} | **Time:** {rep['estimated_time']}\n"
                f"• **Estimated Cost:** {rep['cost_est_inr']}\n"
                f"• **Tools Required:** {tools_text}\n\n"
                f"**Replacement Procedure:**\n"
                f"{steps_text}\n\n"
                f"💡 *Tip: After replacing this part, update your **Service History Log** on the dashboard to reset the component wear counter!*"
            )

            return {
                "type": "DIAGNOSIS",
                "case_id": best_match["id"],
                "title": best_match["title"],
                "faulty_part": best_match["faulty_part"],
                "confidence_pct": min(98.5, round(60.0 + highest_score * 3.5, 1)),
                "bot_reply": bot_reply,
                "quick_replies": [
                    "How to replace this part?", "What tools are needed?", "Is it safe to drive home?", "Check service countdown"
                ]
            }

        # Fallback response for unclassified questions
        return {
            "type": "FALLBACK",
            "bot_reply": (
                "🤔 **I hear what you're saying, but I need a little more detail to pinpoint the exact component.**\n\n"
                "Please tell me:\n"
                "1. **What sound is it?** *(e.g., squeal, knock, grind, screech, click, hiss, vibration)*\n"
                "2. **When does it happen?** *(e.g., when pressing brakes, during a sharp turn, over 80 km/h, on morning start)*\n"
                "3. **Where does it feel or sound like it's coming from?** *(e.g., front wheels, under the hood, exhaust, steering wheel)*\n\n"
                "You can also click any of the quick symptom buttons below:"
            ),
            "quick_replies": [
                "Squeaking brakes", "Engine knocking noise", "Steering vibration at 90 km/h",
                "Belt screech on start", "White smoke from exhaust", "AC blowing warm air"
            ]
        }
