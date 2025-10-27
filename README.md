# Chaotic Levy-Flight-Driven Siberian Tiger Optimization for Enhanced Data Clustering

This repository contains the implementation of the algorithm proposed in the paper:

> **Chaotic Levy-Flight-Driven Siberian Tiger Optimization for Enhanced Data Clustering**  
> *[Your Full Name]*  
> Accepted in **Cybernetics & Systems (Taylor & Francis, 2025)**

---

## 📘 Abstract
This study introduces a novel **Chaotic Levy-Flight-Driven Siberian Tiger Optimization (CLFSTO)** algorithm for improved data clustering performance.  
By integrating **chaotic initialization** and **Levy flight perturbation** strategies into the original **Siberian Tiger Optimization (STO)**, the proposed CLFSTO achieves superior exploration–exploitation balance, faster convergence, and enhanced clustering accuracy across various benchmark datasets.

The algorithm’s efficiency is validated using multiple performance metrics and comparative experiments against well-known optimization algorithms such as PSO, GA, GWO, WOA, and K-Means.

---

## ⚙️ Implementation Details

### Programming Environment
- **Language:** Python 3.10+
- **Libraries:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`
- **Execution Platform:** Google Colab / Jupyter Notebook

### Key Components
| Component | Description |
|------------|-------------|
| `clfsto.py` | Main implementation of the CLFSTO algorithm |
| `sto.py` | Standard Siberian Tiger Optimization (baseline) |
| `utils.py` | Helper functions for data preprocessing and evaluation |
| `datasets/` | Benchmark datasets (Liver, Cancer, Wine, Iris, Glass) |
| `results/` | Contains CSV files of iteration-wise convergence results |
| `plots/` | Includes all final figures used in the publication (Box plots, Convergence curves) |

---

## 🧩 Algorithm Overview

### CLFSTO Enhancements:
1. **Chaotic Initialization:** Improves diversity of initial population.
2. **Levy Flight Perturbation:** Prevents premature convergence.
3. **Dynamic Parameter Adaptation:** Ensures better global–local search transition.

### Flow:
```
Initialize population → Apply chaotic mapping → Evaluate fitness
→ Update positions via Levy Flight strategy → Select best solutions
→ Iterate until convergence → Output optimal cluster centers
```

---

## 📊 Experimental Setup

| Parameter | Value |
|------------|--------|
| Population Size | 50 |
| Maximum Iterations | 500 |
| Datasets | Liver, Cancer, Wine, Iris, Glass |
| Comparison Algorithms | STO, CSTO, SSTO, PSO, GA, GWO, WOA, K-Means |
| Evaluation Metrics | Accuracy, Fitness, Convergence, Execution Time |

---

## 📈 Results and Visualization

### Figures
- **Figure 2:** Example grayscale clustering results (600 DPI, TIF)
- **Figure 3:** Box plots of the last 30 iterations (300 DPI, JPEG)
- **Figure 4:** Convergence curves of all algorithms (300 DPI, JPEG)

All figures are located in `/plots/` and correspond to the paper’s final accepted version.

---

## 🔍 Citation
If you use this code in your research, please cite the following paper:

```
@article{yourlastname2025clfsto,
  title={Chaotic Levy-Flight-Driven Siberian Tiger Optimization for Enhanced Data Clustering},
  author={Your Name},
  journal={Cybernetics & Systems},
  year={2025},
  publisher={Taylor & Francis}
}
```

---

## 📬 Contact
For questions or collaborations, please contact:  
📧 your.email@domain.com  
🌐 [ResearchGate / ORCID / LinkedIn link if preferred]

---

## 📜 License
This project is distributed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more details.

---
