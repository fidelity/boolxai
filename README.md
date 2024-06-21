# BoolXAI: Explainable AI using expressive Boolean formulas

*This repo is being made available as a static archive. It has been released by Fidelity Investments under the Apache 2.0 license, and will not receive updates.*
*If you have questions, please contact <opensource@fidelity.com>*.

[![ci](https://github.com/fidelity/boolxai/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/fidelity/boolxai/actions/workflows/ci.yml)
[![PyPI version fury.io](https://badge.fury.io/py/boolxai.svg)](https://pypi.python.org/pypi/boolxai/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Downloads](https://static.pepy.tech/personalized-badge/boolxai?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/boolxai)

BoolXAI ([MAKE'23](https://www.mdpi.com/2504-4990/5/4/86), [ArXiv'23](https://arxiv.org/pdf/2306.03976)) is a research library for Explainable AI (XAI) based
on expressive Boolean formulas that allow And, Or, Choose(k), AtMost(k), and AtLeast(k) operators.

The Boolean formula defines a rule with tunable
complexity (or interpretability), according to which input data are classified. Such a
formula can include any operator that can be applied to one or more Boolean variables,
thus providing higher expressiveness compared to more rigid rule-based and tree-based
approaches. The classifier is trained using native local optimization techniques,
efficiently searching the space of feasible formulas. For a high-level introduction, see the [Fidelity blogpost](https://fcatalyst.com/blog/june2023/explainable-ai-using-expressive-boolean-formulas), and for a technical introduction, see the [Amazon AWS blogpost](https://aws.amazon.com/blogs/quantum-computing/explainable-ai-using-expressive-boolean-formulas/). 

BoolXAI is developed by Amazon Quantum Solutions Lab, the FCAT Quantum and Future Computing Incubator, and the AI Center of Excellence at Fidelity Investments. Documentation is available at [fidelity.github.io/boolxai](https://fidelity.github.io/boolxai).

## Quick Start

The heart of BoolXAI is the rule classifier (`BoolXAI.RuleClassifier`), which can be
used as an interpretable ML model for binary classification. Note that the input data
must be binarized. Here's a simple example showing the basic usage. For additional
examples, including advanced usage, see the Usage Examples.

<!--- When updating the below, please update tests/test_quick_start.py as well --->

```python
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from boolxai import BoolXAI, Operator

# Create random toy data for binary classification. X and y must be binary! 
rng = np.random.default_rng(seed=42)
X = rng.choice([0, 1], size=(100, 10))
y = rng.choice([0, 1], size=100)

# Rule classifier with maximum depth, complexity, possible operators
rule_classifier = BoolXAI.RuleClassifier(max_depth=3,
                                         max_complexity=6, 
                                         operators=[Operator.And, Operator.Or, Operator.Choose, Operator.AtMost, Operator.AtLeast],
                                         random_state=42)

# Learn the best rule
rule_classifier.fit(X, y)

# Best rule and best score
best_rule = rule_classifier.best_rule_
best_score = rule_classifier.best_score_
print(f"{best_rule=} {best_score=:.2f}")

# The depth of a rule is the number of edges in the longest path from the root to any leaf/literal.
# The complexity of a rule is the total number of operators and literals.
print(f"depth={best_rule.depth()} complexity={best_rule.complexity()}")

# Predict and score
y_pred = rule_classifier.predict(X)
score = balanced_accuracy_score(y, y_pred)
print(f"{score=:.2f}")

# It is also possible to plot the best rule --requires installing plot dependencies
best_rule.plot()

# or get a networkx.DiGraph representation of the rule --requires installing plot dependencies
G = best_rule.to_graph()
print(G)
```

## Usage Examples

- [Training a rule classifier](http://fidelity.github.io/boolxai/docs/_examples/basic_usage.html#Training-a-rule-classifier)
- [Interpreting and visualizing the rules](http://fidelity.github.io/boolxai/docs/_examples/basic_usage.html#Making-sense-of-the-rules)
- [Number of starts (num_starts)](http://fidelity.github.io/boolxai/docs/_examples/advanced_usage.html#Number-of-starts)
- [Bagging (max_samples)](http://fidelity.github.io/boolxai/docs/_examples/advanced_usage.html#Bagging)
- [Parallelization (num_jobs)](http://fidelity.github.io/boolxai/docs/_examples/advanced_usage.html#Parallelization)
- [Cross-Validation](http://fidelity.github.io/boolxai/docs/_examples/advanced_usage.html#Cross-validation)
- [Pareto frontier (max_complexity)](http://fidelity.github.io/boolxai/docs/_examples/advanced_usage.html#Pareto-frontier)
- [Boolean operators](http://fidelity.github.io/boolxai/docs/_examples/advanced_usage.html#Changing-the-allowed-operators)
- [Custom operators](http://fidelity.github.io/boolxai/docs/_examples/advanced_usage.html#Adding-custom-operators)
- [Partial optimization (base_rule)](http://fidelity.github.io/boolxai/docs/_examples/advanced_usage.html#Optimizing-part-of-a-rule)
- [Boosting](http://fidelity.github.io/boolxai/docs/_examples/upstream_usage.html#Boosting)
- [Multi-label classification](http://fidelity.github.io/boolxai/docs/_examples/upstream_usage.html#Multilabel-classification)
- [Multi-class classification](http://fidelity.github.io/boolxai/docs/_examples/upstream_usage.html#Multiclass-classification)

## Installation

We recommend installing BoolXAI in a virtual environment.

It can be installed from PyPI using:

```pip install boolxai```

Alternatively, clone this repo and use:

```pip install -e .```

In order to plot the best rule and get its networkx graph, additional dependencies are required, which can be installed using:

```pip install boolxai[plot]```

In order to run the [Notebook Usage Examples](https://github.com/fidelity/boolxai/tree/master/notebooks/), additional
dependencies are required, which can be installed using:

```pip install -r notebooks/requirements.txt```

Note that it's recommended to create a Jupyter notebook kernel for this repository and
run the notebooks using it, for example:

```python3 -m ipykernel install --user --name boolxai```

### Requirements

This library requires **Python 3.8+**.
See [requirements.txt](requirements.txt) for dependencies. For plotting, see [requirements_plot.txt](requirements_plot.txt) and `graphviz` must be installed separately (see [instructions](https://graphviz.org/download)) - it cannot be installed using `pip`.

## Citation

If you use BoolXAI in a publication, please cite it as:

```bibtex
@Article{boolxai2023,
AUTHOR = {Rosenberg, Gili and Brubaker, John Kyle and Schuetz, Martin J. A. and Salton, Grant and Zhu, Zhihuai and Zhu, Elton Yechao and Kadıoğlu, Serdar and Borujeni, Sima E. and Katzgraber, Helmut G.},
TITLE = {Explainable Artificial Intelligence Using Expressive Boolean Formulas},
JOURNAL = {Machine Learning and Knowledge Extraction},
VOLUME = {5},
YEAR = {2023},
NUMBER = {4},
PAGES = {1760--1795},
URL = {https://www.mdpi.com/2504-4990/5/4/86},
ISSN = {2504-4990},
DOI = {10.3390/make5040086}
}
```

## Support

Please submit bug reports, questions, and feature requests as
[GitHub Issues](https://github.com/fidelity/boolxai/issues).

## License

BoolXAI is licensed under the [Apache License 2.0](LICENSE).

<br>
