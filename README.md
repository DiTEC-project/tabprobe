# Association Rule Learning with Tabular Foundation Models

This repository contains the experimental source code for the paper **"Tabular Foundation Models Can Learn Association
Rules"**, TabProbe algorithm (Algorithm 2 in the paper) and the baselines.

In addition, it provides a reusable Python
wrapper that enables researchers and practitioners to mine association rules
from tabular foundation models without having to reproduce the full experimental pipeline.

The tabular foundation models supported both in the experiments and in the wrapper library are
**TabPFNv2.5** [1], **TabICL** [2], and **TabDPT** [3].

**Table of Contents**

1. [Project Overview and Structure](#project-overview-and-structure)
2. [Reproducing the Experiments](#reproducing-the-experiments)
3. [Learn Rules on Your Own Data](#learn-rules-on-your-own-data)
4. [References](#references)

## Project Overview and Structure

This section describes repository structure, datasets, tabular foundation models
and baselines used in the experiments.

### Repository structure

```
├── src/
│   ├── wrapper/                     # Reusable rule learning library from TFMs (TabPFNv2.5, TabICL, and TabDPT)
│   │   ├── __init__.py
│   │   └── tabprobe.py              # TabProbe: unified interface for all TFMs
│   ├── experiments/
│   │   ├── rule_mining/                # Association rule mining experiments
│   │   │   ├── tabpfn_experiments.py
│   │   │   ├── tabicl_experiments.py
│   │   │   ├── tabdpt_experiments.py
│   │   │   ├── aerial_experiments.py
│   │   │   └── fpgrowth_experiments.py
│   │   ├── itemset_mining/             # Frequent itemset mining (required for running CORELS)
│   │   ├── classification/             # CBA and CORELS classification experiments
│   │   │   ├── cba_experiments.py
│   │   │   └── corels_experiments.py
│   │   ├── scalability_experiments.py  # Scalability experiments
│   │   └── hyperparameter_analysis.py  # Hyper-parameter analysis experiment
│   └── utils/                   # Shared utilities
│       ├── data_loading.py      # UCI ML repo data loading
│       ├── data_prep.py         # Data encoding
│       ├── rule_extraction.py   # Rule extraction from reconstructions
│       └── rule_quality.py      # Quality metrics
├── requirements.txt             # Project requirements
```

### Datasets

10 datasets from the UCI ML repository [4] are used in the experiments. The datasets can be found
under the [data](data) folder, organized based on their size.

**Small tabular data**

- [Acute Inflammations](https://archive.ics.uci.edu/dataset/184)
- [Hepatitis](https://archive.ics.uci.edu/dataset/46)
- [Cervical Cancer Behavior Risk](https://archive.ics.uci.edu/dataset/537)
- [Autistic Spectrum Disorder Screening Data for Adolescent](https://archive.ics.uci.edu/dataset/420)
- [Fertility](https://archive.ics.uci.edu/dataset/244)

**Larger tabular data**

- [Breast Cancer](https://archive.ics.uci.edu/dataset/14)
- [Congressional Voting Records](https://archive.ics.uci.edu/dataset/105)
- [Mushroom](https://archive.ics.uci.edu/dataset/73)
- [Chess (King-Rook vs. King-Pawn)](https://archive.ics.uci.edu/dataset/22)
- [Spambase](https://archive.ics.uci.edu/dataset/94)

### Tabular foundation models and baselines

3 tabular foundation
models ([TabPFNv2.5](https://github.com/PriorLabs/TabPFN) [1], [TabICL](https://github.com/soda-inria/tabicl) [2],
and [TabDPT](https://github.com/layer6ai-labs/TabDPT-inference) [3]),
Aerial+ [9] and FP-Growth [5] are used in the experiments. Please follow the hyperlinks to access detailed instructions
on installations of individual tabular foundation models. Aerial+ is implemented
with [PyAerial](https://github.com/DiTEC-project/pyaerial) [10] and FP-Growth
is implemented with [Mlxtend](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/) [11].

In downstream classification experiments, CORELS [8] is implemented following its original
code [repository](https://github.com/corels/corels), and CBA [7] is implemented
with [pyARC](https://github.com/jirifilip/pyARC) [6].

## Reproducing the Experiments

The source code for association rule learning is entirely written in Python.

**1. Install project requirements**

```
pip install -r requirements.txt
```

**2. Access to tabular foundation models**

TabPFN requires users to approve its terms & conditions on [huggingface](https://huggingface.co/Prior-Labs/tabpfn_2_5).
And then sign in to huggingface on CLI as follows:

```
huggingface-cli login
```

TabDPT can be installed as instructed on its repository as follows:

```
git clone git@github.com:layer6ai-labs/TabDPT.git
cd TabDPT
pip install -e .
pip install --group dev
```

**3. Running Experiments**

All experiments run on the small datasets by default. Update the call

```
get_ucimlrepo_datasets(size="normal")
```

in each individual experiment files to

```
get_ucimlrepo_datasets(size="normal")
```

to be able run the experiments on larger datasets as well.

Experiments can be run easily as follows:

```bash
# Rule mining experiments
python src/experiments/rule_mining/tabpfn_experiments.py
python src/experiments/rule_mining/tabicl_experiments.py
python src/experiments/rule_mining/tabdpt_experiments.py
python src/experiments/rule_mining/aerial_experiments.py
python src/experiments/rule_mining/fpgrowth_experiments.py

# Classification experiments
python src/experiments/classification/cba_experiments.py
python src/experiments/classification/corels_experiments.py

# Scalability and hyperparameter analysis
python src/experiments/scalability_experiments.py
python src/experiments/hyperparameter_analysis.py
```

**4. Accessing the experimental results**

All experimental results are saved into an `out` directory.

Rule mining experiments generate both an excel sheet containing rule quality scores and other
experimental logs into `out` folder, and all rules are saved into `out/rules` per rule learner, dataset and seed.

Classification experiments also generate both an excel sheet containing classification performance of the tabular
foundation models and the baselines into `out` folder and the classifiers per method (CBA or CORELS), rule learner,
dataset and seed are saved into `out/classifiers`.

## Learn Rules on Your Own Data

Beyond reproducing the experiments, we also provide a `rulemining` wrapper for TabPFN, TabICL and TabDPT that can be
run on any given dataset in `pandas.Dataframe` form.

To be able to run the example code below, install `ucimlrepo` repository which the code uses to fetch a sample dataset.

```
pip install ucimlrepo
```

Rule learning with tabular foundation models can then be run on the categorical pandas dataframes:

```python
from ucimlrepo import fetch_ucirepo
from src.wrapper import TabProbe

# Load breast cancer dataset from UCI ML repository
dataset = fetch_ucirepo(id=14).data.features

# Mine rules with TabPFN
miner = TabProbe(method='tabicl', ant_similarity=0.5, cons_similarity=0.8)
rules = miner.mine_rules(dataset, metrics=["support", "confidence"])

# print rule quality statistics
print("Average rule quality statistics:\n", print(miner.get_statistics()))

print("3 sample association rules:")
# Print top 3 rules
for rule in rules[:3]:
    ant = ' & '.join([f"{a['feature']}={a['value']}" for a in rule['antecedents']])
    cons = f"{rule['consequent']['feature']}={rule['consequent']['value']}"
    print(f"{ant} -> {cons} (conf: {rule.get('confidence', 0):.3f})")

# Convert to DataFrame for analysis
rules_df = miner.to_dataframe()
print("Rules in dataframe form for further processing:\n", rules_df.head())
```

Output of the code above is as follows:

```
Average rule quality statistics: 
{'num_rules': 18, 'support': 0.47513597513597516, 'confidence': 0.8988067608986312, 'data_coverage': 0.9020979020979021}

3 sample association rules:
inv-nodes=0-2 -> node-caps=no (conf: 0.944)
inv-nodes=0-2 -> irradiat=no (conf: 0.859)
node-caps=no -> inv-nodes=0-2 (conf: 0.905)

Rules in dataframe form for further processing:
     antecedents     consequent   support  confidence
0  inv-nodes=0-2   node-caps=no  0.702797    0.943662
1  inv-nodes=0-2    irradiat=no  0.639860    0.859155
2   node-caps=no  inv-nodes=0-2  0.702797    0.905405
3    irradiat=no  inv-nodes=0-2  0.639860    0.839450
4    irradiat=no   node-caps=no  0.657343    0.862385
```

Numerical values needs to discretized before rule learning.
This can be done via one of the methods provided in e.g., PyAerial's discretization
module: https://pyaerial.readthedocs.io/en/latest/user_guide.html#running-aerial-for-numerical-values

Applying k-means discretization via pyaerial:

```
from ucimlrepo import fetch_ucirepo
from aerial import discretization
from src.wrapper import TabProbe

# Load fertility dataset from UCI ML repository
dataset = fetch_ucirepo(id=244).data.features

# auto identify numerical columns and discretize
discrete_df = discretization.equal_width_discretization(dataset, n_bins=5)

# Mine rules with TabICL
miner = TabProbe(method='tabicl', ant_similarity=0.5, cons_similarity=0.8)
rules = miner.mine_rules(discrete_df, metrics=["support", "confidence"])
```

**Parameters**

The `TabProbe` class exposes the following parameters:

```python
TabProbe(
    method='tabicl',  # Foundation model to use: 'tabpfn', 'tabicl', or 'tabdpt'
    max_antecedents=2,  # Maximum number of items allowed in the rule antecedent
    ant_similarity=0.5,  # Similarity threshold for validating antecedents (0.0–1.0)
    cons_similarity=0.8,  # Similarity threshold for extracting consequents (0.0–1.0)
    n_estimators=8,  # Number of ensemble estimators for prediction averaging
    noise_factor=0.5,  # Gaussian noise factor added to context data
    n_bins=5,  # Number of bins for discretizing numerical features
    random_state=42  # Random seed for reproducibility
)
```

## References

1. Grinsztajn, Léo, et al. "TabPFN-2.5: Advancing the state of the art in tabular foundation models." arXiv preprint
   arXiv:2511.08667 (2025).
2. Qu, Jingang, et al. "Tabicl: A tabular foundation model for in-context learning on large data." arXiv preprint arXiv:
   2502.05564 (2025).
3. Ma, Junwei, et al. "Tabdpt: Scaling tabular foundation models." arXiv preprint arXiv:2410.18164 (2024).
4. Kelly, Markelle, Rachel Longjohn, and Kolby Nottingham. "The UCI machine learning repository." Nov. 2023,
5. Jiawei Han, Jian Pei, and Yiwen Yin. Mining frequent patterns without candidate generation. ACM sigmod record, 29(2):
   1–12, 2000.
6. Jirı, Filip, and Tomáš Kliegr. "Classification based on associations (CBA)-a performance analysis." (2018).
7. Liu, Bing, Wynne Hsu, and Yiming Ma. "Integrating classification and association rule mining." Proceedings of the
   fourth international conference on knowledge discovery and data mining. 1998.
8. Angelino, Elaine, et al. "Learning certifiably optimal rule lists for categorical data." Journal of Machine Learning
   Research 18.234 (2018): 1-78.
9. Erkan Karabulut, Paul Groth, and Victoria Degeler. Neurosymbolic association rule mining from tabular data. In
   Proceedings of The 19th International Conference on Neurosymbolic Learning and Reasoning (NeSy 2025), volume 284 of
   Proceedings of Machine Learning Research, pages 565–588. PMLR, 08–10 Sep 2025.
10. Karabulut, Erkan, Paul Groth, and Victoria Degeler. "PyAerial: Scalable association rule mining from tabular data."
    SoftwareX 31 (2025): 102341.
11. Raschka, Sebastian. "MLxtend: Providing machine learning and data science utilities and extensions to Python’s
    scientific computing stack." Journal of open source software 3.24 (2018): 638.