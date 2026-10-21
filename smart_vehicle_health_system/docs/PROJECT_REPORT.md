# AI-Based Smart Vehicle Health Monitoring System
## Capstone Engineering Project Report
**Institution:** Department of Computer Engineering, St. John College of Engineering and Management (Autonomous)  
**Academic Year:** Semester V (Odd Term)  
**Integrated Core Subjects:**
1. **Theoretical Computer Science (TCS)** — Course Code: `24CSPCC501C`
2. **Soft Computing (SC)** — Course Code: `24CSPCC502C`
3. **Artificial Intelligence & Machine Learning (AIML)**

---

## 1. Executive Summary
Modern vehicles are sophisticated distributed cyber-physical systems generating hundreds of telematics parameters per second across the Controller Area Network (CAN-bus). Standard On-Board Diagnostic (OBD-II) units rely on rigid, static threshold binary triggers that fail to detect gradual subsystem degradation, handle noisy sensor dynamics, or provide predictive time-to-failure windows.

This capstone project implements an end-to-end smart vehicle health diagnostic and prognostics platform. The architecture grounds every layer in academic syllabus principles:
- **TCS Layer:** Employs formal language theory (Regular Expressions and Context-Free Grammars) to tokenize and validate SAE J2012 Diagnostic Trouble Codes (DTC) and telemetry packets, coupled with a 5-state Deterministic Finite Automaton (DFA) / Moore Machine that governs electronic throttle and safety shutdown modes.
- **Soft Computing Layer:** Implements a Mamdani Fuzzy Inference System (FIS) utilizing triangular and trapezoidal membership functions with Centroid defuzzification to calculate a continuous Vehicle Health Index (VHI, 0–100%) in the presence of sensor noise and uncertainty, alongside genetic algorithm optimization and neural learning.
- **AIML Layer:** Delivers probabilistic fault mode classification, Remaining Useful Life (RUL in kilometers) physical wear estimation, and multivariate statistical anomaly detection.
- **Application Interface:** A modern, zero-external-dependency web application providing real-time telemetry gauges, interactive fault simulation, manual sensor overrides, DTC scanners, and CSV/JSON reporting.

---

## 2. Subject Syllabus Mapping & Mathematical Formulations

### 2.1 Theoretical Computer Science (TCS)
#### A. Formal Language & SAE J2012 DTC Regular Expression (Modules I & II)
All standard automotive DTC codes follow SAE J2012, which forms a Type-3 regular language in the Chomsky hierarchy:
$$\Sigma = \{P, C, B, U\} \cup \{0, 1, 2, 3\} \cup \{0..9, A..F\}$$
$$L(DTC) = ^([PCBU])([0-3])([0-9A-F])([0-9A-F]{2})\$$
- **Character 1:** Subsystem Domain ($P$ = Powertrain, $C$ = Chassis, $B$ = Body, $U$ = Network CAN-bus).
- **Character 2:** Standard Identifier ($0$ = SAE/ISO standard, $1$ = Manufacturer proprietary).
- **Character 3:** Subsystem code ($1$ = Fuel/Air metering, $2$ = Injector circuit, $3$ = Ignition/Misfire, $7$ = Transmission, etc.).
- **Characters 4 & 5:** Fault specific identifier.

#### B. Deterministic Finite Automaton (DFA) / Moore Machine (Module I)
Operational safety governance is formulated as a 6-tuple Moore Machine:
$$M = (Q, \Sigma, \Delta, \delta, \lambda, q_0)$$
- **States $Q$:** $\{S_0: \text{NORMAL}, S_1: \text{ADVISORY}, S_2: \text{LIMP\_MODE}, S_3: \text{CRITICAL\_FAULT}, S_4: \text{EMERGENCY\_HALT}\}$
- **Alphabet $\Sigma$:** $\{\text{ALL\_NOMINAL}, \text{TRANSIENT\_ANOMALY}, \text{PERSISTENT\_MODERATE}, \text{SEVERE\_BREACH}, \text{SAFETY\_LOCKOUT}, \text{MANUAL\_RESET}\}$
- **Output Function $\lambda(q)$:**
  - $\lambda(S_0) = (\text{Throttle: } 100\%, \text{MIL: OFF}, \text{Buzzer: OFF}, \text{FuelCut: False})$
  - $\lambda(S_1) = (\text{Throttle: } 100\%, \text{MIL: STEADY\_AMBER}, \text{Buzzer: OFF}, \text{FuelCut: False})$
  - $\lambda(S_2) = (\text{Throttle: } 45\%, \text{MIL: FLASH\_AMBER}, \text{Buzzer: INTERMITTENT}, \text{FuelCut: False})$
  - $\lambda(S_3) = (\text{Throttle: } 15\%, \text{MIL: FLASH\_RED}, \text{Buzzer: CONTINUOUS}, \text{FuelCut: False})$
  - $\lambda(S_4) = (\text{Throttle: } 0\%, \text{MIL: SOLID\_RED}, \text{Buzzer: CONTINUOUS}, \text{FuelCut: True})$

---

### 2.2 Soft Computing (SC)
#### A. Fuzzy Membership Functions (Module 2)
The Mamdani FIS defines continuous linguistic variables:
- **Triangular Function:**
  $$\mu(x; a, b, c) = \max\left(0, \min\left(\frac{x - a}{b - a}, \frac{c - x}{c - b}\right)\right)$$
- **Trapezoidal Function:**
  $$\mu(x; a, b, c, d) = \max\left(0, \min\left(\frac{x - a}{b - a}, 1, \frac{d - x}{d - c}\right)\right)$$

#### B. Defuzzification via Centroid Method (COG)
The continuous Vehicle Health Index (VHI) across the universe $Z = [0, 100]\%$ is evaluated using Center of Gravity:
$$VHI = \frac{\int_{0}^{100} z \cdot \mu_{\text{agg}}(z) \, dz}{\int_{0}^{100} \mu_{\text{agg}}(z) \, dz}$$

---

### 2.3 Artificial Intelligence & Machine Learning (AIML)
#### A. Multi-Class Subsystem Failure Classifier
Probabilistic classification across 5 operational categories:
1. `NORMAL_OPERATION`
2. `COOLING_SYSTEM_OVERHEAT`
3. `OIL_STARVATION_LOW_PRESSURE`
4. `ELECTRICAL_ALTERNATOR_DEGRADATION`
5. `DRIVETRAIN_TRANSMISSION_VIBRATION`

Logits are scaled via Softmax with temperature $T=1.5$:
$$P(C_k \mid X) = \frac{\exp(-d_k(X) / T)}{\sum_{j} \exp(-d_j(X) / T)}$$

#### B. Remaining Useful Life (RUL) Physical Wear Regression
Estimated residual driving distance before threshold failure:
$$RUL = \frac{RUL_0 \times \left(\frac{VHI}{100}\right)}{1.0 + 2.5 \cdot \text{Stress}_{\text{thermal}} + 3.5 \cdot \text{Stress}_{\text{lubrication}} + 1.8 \cdot \text{Stress}_{\text{vibration}}}$$

---

## 3. Project Structure
```
smart_vehicle_health_system/
├── core/
│   ├── tcs_module.py             # Automata, DFA, Moore Machine, SAE J2012 Regex, CFG parser
│   ├── soft_computing_module.py  # Mamdani FIS, Centroid Defuzzifier, Pure-NumPy MLP, GA
│   ├── aiml_module.py            # Failure Classifier, RUL Regressor, Multivariate Anomaly Detector
│   ├── pipeline.py               # Integrated Multi-disciplinary Diagnostic Pipeline
│   └── telemetry_simulator.py    # Dynamic multi-scenario telematics generator
├── web/
│   ├── app_server.py             # Python standard library HTTP/REST backend
│   └── index.html                # Interactive modern cockpit web dashboard
├── data/
│   └── vehicle_telemetry_dataset.csv # Pre-generated 750-row synthetic telemetry dataset
├── docs/
│   ├── PROJECT_REPORT.md         # Comprehensive academic technical report
│   └── Smart_Vehicle_Health_Monitoring_Capstone_Report.pdf # Formatted PDF report
├── main.py                       # CLI entrypoint (--demo, --server, --test, --export-dataset)
└── README.md                     # Quickstart documentation
```

---

## 4. Experimental Evaluation & Validation

| Scenario | Primary Fault Vector | TCS State | Fuzzy VHI | AIML Diagnosis | Est. RUL | Actuator Mode |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal Cruise** | RPM 2240, Temp 90.3°C, Oil 48.2 PSI | `S0_NORMAL` | 91.6% (Pristine) | `NORMAL_OPERATION` (64%) | 13,732 km | 100% Throttle, Lamps OFF |
| **Coolant Boil** | Temp 123.5°C, DTC P0217 | `S3_CRITICAL` | 12.2% (Hazard) | `COOLING_OVERHEAT` (44%) | 417 km | 15% Throttle, Flash Red |
| **Oil Starvation** | Oil 10.1 PSI, DTC P0524 | `S3_CRITICAL` | 23.0% (Hazard) | `OIL_STARVATION` (54%) | 415 km | 15% Throttle, Flash Red |
| **Alternator Decay**| Voltage 10.29V, DTC P0562 | `S3_CRITICAL` | 43.3% (Degraded)| `ELECTRICAL_DECAY` (68%) | 6,498 km | 15% Throttle, Amber Lamp |
| **Drivetrain Shudder**| Vib 8.6 m/s², RPM 3924 | `S3_CRITICAL` | 43.3% (Degraded)| `TRANSMISSION_SLIP` (66%) | 1,051 km | 15% Throttle, Flash Amber |

---

## 5. How to Run the Application
1. **Automated Terminal Demo:**
   ```bash
   python3 main.py --demo
   ```
2. **Launch Interactive Web Dashboard:**
   ```bash
   python3 main.py --server
   ```
   Open your browser at `http://localhost:8080` to interact with live gauges, sliders, and fault injectors.
3. **Execute Comprehensive Unit Tests:**
   ```bash
   python3 main.py --test
   ```
4. **Export Synthetic Telemetry Dataset:**
   ```bash
   python3 main.py --export-dataset
   ```
