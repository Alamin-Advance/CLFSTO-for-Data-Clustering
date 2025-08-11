# Chaotic-Levy-Flight-Siberian-Tiger-Optimization-CLFSTO-for-Data-Clustering
📌 Overview
This repository presents CLFSTO—an enhanced Siberian Tiger Optimization (STO) algorithm integrated with chaotic logistic maps and Levy Flight (LF) for efficient data clustering. The proposed method improves exploration and exploitation, overcoming premature convergence and local optima issues in traditional clustering algorithms like K-means, PSO, and GWO.

🔹 Key Features:
✅ Chaotic Logistic Maps – Enhances population diversity and dynamic parameter control.
✅ Levy Flight (LF) – Improves global search capabilities, avoiding stagnation.
✅ Superior Clustering Performance – Outperforms K-means, GA, GWO, WOA, PSO, and STO on 10 benchmark datasets.
✅ Fitness Metrics – Evaluated using Silhouette Score and Within-Cluster Variance (WCS).

📊 Benchmark Datasets
Dataset	Samples	Features	Clusters
Iris	150	4	3
Wine	178	13	3
Glass	214	9	6
Balance	625	4	3
Diabetes	768	8	2
Heart	303	13	2
Cancer	569	30	2
Ecoli	336	7	8
Seeds	210	7	3
Liver	345	6	2
(All datasets sourced from UCI Machine Learning Repository and scikit-learn.)

⚙️ Algorithm Workflow
Initialization – Chaotic sequences generate diverse initial cluster centers.

Prey Hunting Phase (Exploration) – Levy Flight updates cluster centers:

c
j
(
t
+
1
)
=
c
j
(
t
)
+
Δ
x
⋅
(
T
P
−
l
⋅
X
b
e
s
t
)
c 
j
(t+1)
​
 =c 
j
(t)
​
 +Δx⋅(TP−l⋅X 
best
​
 )
Fighting Phase (Exploitation) – Chaotic refinement:

c
j
(
t
+
1
)
=
c
j
(
t
)
+
r
t
⋅
U
B
−
c
j
(
t
)
t
+
1
c 
j
(t+1)
​
 =c 
j
(t)
​
 +r 
t
​
 ⋅ 
t+1
UB−c 
j
(t)
​
 
​
 
Fitness Evaluation – Minimizes Within-Cluster Variance (WCS) and maximizes Silhouette Score.

📌 Flowchart of CLFSTO:
https://via.placeholder.com/600x400?text=CLFSTO+Algorithm+Flowchart (Replace with actual diagram)

📈 Performance Metrics
Metric	Formula	Goal
Within-Cluster Variance (WCS)	
∑
j
=
1
K
∑
x
i
∈
C
j
∥
x
i
−
C
j
∥
2
∑ 
j=1
K
​
 ∑ 
x 
i
​
 ∈C 
j
​
 
​
 ∥x 
i
​
 −C 
j
​
 ∥ 
2
 	Minimize (Compact clusters)
Silhouette Score	
s
(
x
i
)
=
b
(
x
i
)
−
a
(
x
i
)
max
⁡
(
a
(
x
i
)
,
b
(
x
i
)
)
s(x 
i
​
 )= 
max(a(x 
i
​
 ),b(x 
i
​
 ))
b(x 
i
​
 )−a(x 
i
​
 )
​
 	Maximize (Better separation)
📂 Repository Structure
text
├── /data/            # Benchmark datasets (CSV format)  
├── /src/              
│   ├── CLFSTO.py     # Main algorithm implementation  
│   ├── utils.py      # Helper functions (fitness, Levy Flight, etc.)  
├── /results/         # Clustering outputs & performance logs  
├── /figures/         # Convergence plots, Silhouette analysis  
├── requirements.txt  # Python dependencies  
└── README.md  
🛠️ Installation & Usage
Clone the repository:

bash
git clone https://github.com/yourusername/CLFSTO-Clustering.git
cd CLFSTO-Clustering
Install dependencies:

bash
pip install -r requirements.txt
Run CLFSTO clustering:

python
from src.CLFSTO import CLFSTO_Clustering
data = load_dataset("iris.csv")  
best_centers, fitness = CLFSTO_Clustering(data, k=3, max_iter=100)
📜 Citation

bibtex
@misc{CLFSTO2024,
  author = {Hossain},
  title = {Chaotic Levy Flight Siberian Tiger Optimization (CLFSTO) for Data Clustering},
  year = {2025},
  publisher = {},
  journal = {},
  howpublished = {\url{}}
}
🤝 Contributing
Contributions are welcome! Open an Issue or submit a Pull Request for improvements.

📧 Contact
For questions or collaborations, contact: alaminh1411@gmail.com

🔹 Optimized Clustering with Metaheuristics | Built with Python & NumPy 🔹
