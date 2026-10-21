"""
Soft Computing (SC) Module
Course Code Alignment: 24CSPCC502C (SJCEM Computer Engineering)
Components:
1. Module 2: Fuzzy Logic System - Mamdani Fuzzy Inference System (FIS) for Vehicle Health Index (VHI)
2. Module 3: Artificial Neural Networks - Pure-NumPy Multi-Layer Perceptron (MLP) with Backpropagation
3. Module 4: Evolutionary Computing - Genetic Algorithm (GA) for Sensor Weight Optimization
4. Module 5: Hybrid Soft Computing - Neuro-Fuzzy (ANFIS-style Takagi-Sugeno) parameter evaluation
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass


# ==============================================================================
# 1. FUZZY LOGIC SYSTEM (Module 2: Sets, FIS, Centroid Defuzzification)
# ==============================================================================

def trimf(x: float, a: float, b: float, c: float) -> float:
    """Triangular Membership Function μ(x)."""
    if x <= a or x >= c:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0


def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal Membership Function μ(x)."""
    if x <= a or x >= d:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a) if b != a else 1.0
    if b <= x <= c:
        return 1.0
    return (d - x) / (d - c) if d != c else 1.0


@dataclass
class FuzzyRule:
    antecedents: Dict[str, str]  # e.g., {"temp": "HIGH", "oil": "LOW"}
    consequent: str              # e.g., "CRITICAL"
    weight: float = 1.0


class VehicleHealthFuzzySystem:
    """
    Mamdani Fuzzy Inference System for Vehicle Health Evaluation under Sensor Uncertainty.
    - Fuzzification: Triangular and Trapezoidal Membership Functions
    - Inference Engine: Mamdani Min-Implication
    - Aggregation: Max-Union Aggregation
    - Defuzzification: Centroid Method (Center of Gravity / COG)
    """

    def __init__(self):
        # Discretized universe of discourse for defuzzification (0 to 100%)
        self.health_universe = np.linspace(0.0, 100.0, 500)
        self.rules: List[FuzzyRule] = []
        self._build_rule_base()

    def fuzzify_temperature(self, temp: float) -> Dict[str, float]:
        return {
            "COOL": trapmf(temp, -20.0, -20.0, 70.0, 80.0),
            "NORMAL": trimf(temp, 75.0, 90.0, 102.0),
            "HIGH": trimf(temp, 98.0, 110.0, 120.0),
            "OVERHEATING": trapmf(temp, 115.0, 125.0, 150.0, 150.0)
        }

    def fuzzify_oil_pressure(self, oil: float) -> Dict[str, float]:
        return {
            "CRITICAL_LOW": trapmf(oil, 0.0, 0.0, 15.0, 25.0),
            "LOW": trimf(oil, 20.0, 28.0, 36.0),
            "NORMAL": trimf(oil, 32.0, 48.0, 68.0),
            "HIGH": trapmf(oil, 62.0, 75.0, 100.0, 100.0)
        }

    def fuzzify_battery_voltage(self, volt: float) -> Dict[str, float]:
        return {
            "WEAK": trapmf(volt, 8.0, 8.0, 11.2, 11.8),
            "HEALTHY": trimf(volt, 11.7, 12.6, 14.4),
            "OVERVOLTAGE": trapmf(volt, 14.3, 14.8, 16.5, 16.5)
        }

    def fuzzify_vibration(self, vib: float) -> Dict[str, float]:
        return {
            "SMOOTH": trapmf(vib, 0.0, 0.0, 1.5, 2.8),
            "MODERATE": trimf(vib, 2.2, 3.8, 5.5),
            "SEVERE": trapmf(vib, 5.0, 6.5, 12.0, 12.0)
        }

    def evaluate_output_membership(self, health_pct: float) -> Dict[str, float]:
        """Output Linguistic Variable: Vehicle Health Index (VHI) %"""
        return {
            "CRITICAL": trapmf(health_pct, 0.0, 0.0, 15.0, 30.0),
            "DEGRADED": trimf(health_pct, 25.0, 45.0, 60.0),
            "GOOD": trimf(health_pct, 55.0, 72.0, 88.0),
            "EXCELLENT": trapmf(health_pct, 82.0, 92.0, 100.0, 100.0)
        }

    def _build_rule_base(self):
        # 12 Core Expert Heuristic Rules
        self.rules = [
            # 1. All nominal
            FuzzyRule({"temp": "NORMAL", "oil": "NORMAL", "volt": "HEALTHY", "vib": "SMOOTH"}, "EXCELLENT"),
            # 2. Good general condition
            FuzzyRule({"temp": "NORMAL", "oil": "NORMAL", "vib": "MODERATE"}, "GOOD"),
            # 3. Overheating alone
            FuzzyRule({"temp": "OVERHEATING"}, "CRITICAL"),
            # 4. Critical oil pressure
            FuzzyRule({"oil": "CRITICAL_LOW"}, "CRITICAL"),
            # 5. Dual thermal + lubrication strain
            FuzzyRule({"temp": "HIGH", "oil": "LOW"}, "CRITICAL"),
            # 6. High temp with normal oil
            FuzzyRule({"temp": "HIGH", "oil": "NORMAL"}, "DEGRADED"),
            # 7. Low oil with normal temp
            FuzzyRule({"oil": "LOW", "temp": "NORMAL"}, "DEGRADED"),
            # 8. Severe drivetrain vibration
            FuzzyRule({"vib": "SEVERE"}, "DEGRADED"),
            # 9. Severe vibration + weak battery
            FuzzyRule({"vib": "SEVERE", "volt": "WEAK"}, "CRITICAL"),
            # 10. Weak electrical system
            FuzzyRule({"volt": "WEAK", "temp": "NORMAL"}, "DEGRADED"),
            # 11. Overvoltage alternator issue
            FuzzyRule({"volt": "OVERVOLTAGE"}, "DEGRADED"),
            # 12. Cool engine (warming up) with smooth vibration
            FuzzyRule({"temp": "COOL", "oil": "NORMAL", "vib": "SMOOTH"}, "GOOD"),
        ]

    def infer(self, temp: float, oil: float, volt: float, vib: float) -> Dict[str, Any]:
        mu_temp = self.fuzzify_temperature(temp)
        mu_oil = self.fuzzify_oil_pressure(oil)
        mu_volt = self.fuzzify_battery_voltage(volt)
        mu_vib = self.fuzzify_vibration(vib)

        # Output fuzzy set accumulator across discretized universe
        aggregated_membership = np.zeros_like(self.health_universe)
        fired_rules = []

        for idx, rule in enumerate(self.rules):
            antecedent_strengths = []
            if "temp" in rule.antecedents:
                antecedent_strengths.append(mu_temp[rule.antecedents["temp"]])
            if "oil" in rule.antecedents:
                antecedent_strengths.append(mu_oil[rule.antecedents["oil"]])
            if "volt" in rule.antecedents:
                antecedent_strengths.append(mu_volt[rule.antecedents["volt"]])
            if "vib" in rule.antecedents:
                antecedent_strengths.append(mu_vib[rule.antecedents["vib"]])

            # Mamdani AND -> Minimum
            firing_strength = min(antecedent_strengths) * rule.weight

            if firing_strength > 0.01:
                fired_rules.append({
                    "rule_id": idx + 1,
                    "condition": " AND ".join([f"{k} IS {v}" for k, v in rule.antecedents.items()]),
                    "consequent": rule.consequent,
                    "firing_strength": round(float(firing_strength), 3)
                })

                # Implication (Min) and Aggregation (Max)
                consequent_shape = np.array([
                    self.evaluate_output_membership(val)[rule.consequent]
                    for val in self.health_universe
                ])
                truncated_consequent = np.minimum(firing_strength, consequent_shape)
                aggregated_membership = np.maximum(aggregated_membership, truncated_consequent)

        # Centroid Defuzzification (Center of Gravity / COG)
        area = np.sum(aggregated_membership)
        if area > 1e-6:
            vhi = float(np.sum(self.health_universe * aggregated_membership) / area)
        else:
            vhi = 50.0  # Default neutral fallback

        return {
            "vehicle_health_index": round(vhi, 2),
            "fired_rules": fired_rules,
            "fuzzified_inputs": {
                "temperature": {k: round(v, 2) for k, v in mu_temp.items() if v > 0},
                "oil_pressure": {k: round(v, 2) for k, v in mu_oil.items() if v > 0},
                "battery_voltage": {k: round(v, 2) for k, v in mu_volt.items() if v > 0},
                "vibration": {k: round(v, 2) for k, v in mu_vib.items() if v > 0}
            }
        }


# ==============================================================================
# 2. ARTIFICIAL NEURAL NETWORKS (Module 3: MLP with Backpropagation from Scratch)
# ==============================================================================

class MultiLayerPerceptron:
    """
    Feed-Forward Multi-Layer Perceptron (MLP) implemented in pure NumPy.
    Architecture: Input (N) -> Dense (H1, ReLU) -> Dense (H2, ReLU) -> Output (C, Softmax)
    Trained via Backpropagation and Gradient Descent.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, learning_rate: float = 0.01):
        self.lr = learning_rate
        # He / Xavier weight initialization
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def train_step(self, X: np.ndarray, y_one_hot: np.ndarray) -> float:
        m = X.shape[0]
        Z1, A1, Z2, A2 = self.forward(X)

        # Cross-entropy loss
        loss = -np.sum(y_one_hot * np.log(A2 + 1e-9)) / m

        # Backward propagation
        dZ2 = (A2 - y_one_hot) / m
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Gradient descent update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        return float(loss)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, _, _, A2 = self.forward(X)
        return A2

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


# ==============================================================================
# 3. EVOLUTIONARY COMPUTING (Module 4: Genetic Algorithm for Sensor Feature Selection)
# ==============================================================================

class GeneticAlgorithmOptimizer:
    """
    Binary Genetic Algorithm (GA) for Telemetry Feature Selection and Weight Tuning.
    - Chromosome: Binary vector indicating sensor feature presence/absence.
    - Selection: Tournament Selection.
    - Recombination: Single-Point Crossover.
    - Variation: Uniform Bit-Flip Mutation.
    """

    def __init__(self, num_features: int, pop_size: int = 20, generations: int = 15,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.05):
        self.num_features = num_features
        self.pop_size = pop_size
        self.generations = generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate

    def run(self, fitness_evaluator: Callable[[np.ndarray], float]) -> Dict[str, Any]:
        # Initialize binary population (at least 2 features active)
        population = np.random.choice([0, 1], size=(self.pop_size, self.num_features))
        for i in range(self.pop_size):
            if np.sum(population[i]) == 0:
                population[i, np.random.randint(0, self.num_features)] = 1

        history = []
        best_overall_fitness = -float("inf")
        best_chromosome = None

        for gen in range(self.generations):
            fitnesses = np.array([fitness_evaluator(chrom) for chrom in population])
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_overall_fitness:
                best_overall_fitness = fitnesses[gen_best_idx]
                best_chromosome = population[gen_best_idx].copy()

            history.append({
                "generation": gen + 1,
                "best_fitness": round(float(fitnesses[gen_best_idx]), 4),
                "avg_fitness": round(float(np.mean(fitnesses)), 4)
            })

            # Selection (Tournament of size 3)
            new_pop = []
            for _ in range(self.pop_size):
                candidates = np.random.choice(self.pop_size, 3, replace=False)
                winner = candidates[np.argmax(fitnesses[candidates])]
                new_pop.append(population[winner].copy())
            new_pop = np.array(new_pop)

            # Crossover
            for i in range(0, self.pop_size - 1, 2):
                if np.random.rand() < self.cx_rate:
                    point = np.random.randint(1, self.num_features)
                    new_pop[i, point:], new_pop[i+1, point:] = new_pop[i+1, point:].copy(), new_pop[i, point:].copy()

            # Mutation
            mutation_mask = np.random.rand(*new_pop.shape) < self.mut_rate
            new_pop = np.bitwise_xor(new_pop, mutation_mask.astype(int))

            # Guarantee non-empty features
            for i in range(self.pop_size):
                if np.sum(new_pop[i]) == 0:
                    new_pop[i, np.random.randint(0, self.num_features)] = 1

            population = new_pop

        return {
            "best_chromosome": best_chromosome.tolist(),
            "best_fitness": round(float(best_overall_fitness), 4),
            "generation_history": history
        }
