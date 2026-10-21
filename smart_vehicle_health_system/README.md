# AI-Based Smart Vehicle Health Monitoring System
**Semester V Capstone Project | St. John College of Engineering & Management**  
**Core Subject Integration:** Theoretical Computer Science (TCS) + Soft Computing (SC) + AIML

---

### Quick Start Guide

1. **Run Terminal Diagnostic Demo:**
   ```bash
   python3 main.py --demo
   ```

2. **Launch Interactive Web Dashboard:**
   ```bash
   python3 main.py --server
   ```
   Open `http://localhost:8080` in your browser.

3. **Run Automated Test Suite:**
   ```bash
   python3 main.py --test
   ```

4. **Export Synthetic Telematics CSV:**
   ```bash
   python3 main.py --export-dataset
   ```

---

### Key Architectural Highlights
- **TCS (24CSPCC501C):** 5-state Deterministic Finite Automaton (DFA) / Moore Machine safety governor + SAE J2012 OBD-II Diagnostic Trouble Code (DTC) Regular Expression & Context-Free Grammar (CFG) telematics parser.
- **Soft Computing (24CSPCC502C):** 12-rule Mamdani Fuzzy Inference System with Centroid Defuzzification (calculating 0–100% Vehicle Health Index) + Pure-NumPy Multi-Layer Perceptron (MLP) with backpropagation + Genetic Algorithm feature selector.
- **AIML:** Probabilistic Subsystem Failure Classifier + Remaining Useful Life (RUL in km) predictive maintenance regression + Multivariate Anomaly Detector.
- **Zero External Dependencies:** Runs on standard Python 3 + NumPy/SciPy/Pandas.
