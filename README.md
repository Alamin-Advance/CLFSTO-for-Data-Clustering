# Chaotic Levy-Flight-Driven Siberian Tiger Optimization for Enhanced Data Clustering

This repository contains the implementation of the algorithm proposed in the paper:

> **Chaotic Levy-Flight-Driven Siberian Tiger Optimization for Enhanced Data Clustering**  
> *[Md Al Amin Hossain and Dr. Tahir Sağ]*  
> Accepted in **Cybernetics & Systems (Taylor & Francis, 2025)**


## Abstract
This study introduces a novel **Chaotic Levy-Flight-Driven Siberian Tiger Optimization (CLFSTO)** algorithm for improved data clustering performance. By integrating **chaotic initialization** and **Levy flight perturbation** strategies into the original **Siberian Tiger Optimization (STO)**, the proposed CLFSTO achieves superior exploration–exploitation balance, faster convergence, and enhanced clustering accuracy across various benchmark datasets.
The algorithm’s efficiency is validated using multiple performance metrics and comparative experiments against well-known optimization algorithms such as PSO, GA, GWO, WOA, and K-Means.


## Implementation Details

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
| `datasets/` | Benchmark datasets (Liver, Cancer, Wine, Iris, Glass, etc) |
| `results/` | Contains CSV files of iteration-wise convergence results |
| `plots/` | Includes all final figures used in the publication (Box plots, Convergence curves) |


## Algorithm Overview

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

## Experimental Setup

| Parameter | Value |
|------------|--------|
| Population Size | 30 |
| Maximum Iterations | 1000*number_of_variables |
| Datasets | Liver, Cancer, Wine, Iris, Glass, Balance, Diabetes, Seeds, Ecoli, Heart |
| Comparison Algorithms | STO, CSTO, SSTO, PSO, GA, GWO, WOA, K-Means |


## Citation
If you use this code in your research, please cite the following paper:

```
@article{yourlastname2025clfsto,
  title={Chaotic Levy-Flight-Driven Siberian Tiger Optimization for Enhanced Data Clustering},
  author={Hossain, Md Al Amin  and Sağ, Tahir},
  journal={Cybernetics & Systems},
  year={2025},
  Vol={}
  Issues={}
  publisher={Taylor & Francis}
  Doi={}
}
```

## Contact
For questions or collaborations, please contact:  
📧 alaminh1411@gmail.com  
🌐 [ORCID:https://orcid.org/my-orcid?orcid=0000-0003-3382-5300]


## License
This project is distributed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more details.
