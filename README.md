# Model Training

A repository for training and deploying machine learning models for sentiment analysis.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Local Setup](#local-setup)
- [Module Structure](#module-structure)
- [GitHub Actions & CI/CD](#️-github-actions--cicd)
- [Resources](#-resources)
- [Use of GenAI](#-use-of-genai)

## Features

- `sentiment_model_training`: A Python package for sentiment analysis model training
- Modular design with separate components for:
  - Data acquisition
  - Preprocessing
  - Model training
  - Model evaluation
- **GitHub Actions**:
  - Automatically publishes the trained model to **GitHub Releases**
  - Publishes the bag of words vectorizer to **GitHub Releases**
  - **Automatic versioning** with GitVersion

## Module Structure

The project follows the [Cookiecutter Data Science](https://github.com/drivendataorg/cookiecutter-data-science) project structure, a widely adopted standard template for data science projects. Our implementation is organized as follows:

```
├── LICENSE            <- MIT License
├── README.md          <- The top-level README for developers using this project
├── data               <- Directory containing all data files
│   ├── raw/           <- Directory for raw data
│   │   └── raw.tsv    <- The original, immutable restaurant reviews data
│   └── processed/     <- Directory for processed data
│       ├── processed.npy  <- The processed dataset for modeling
│       ├── labels.pkl     <- Labels for sentiment analysis
│       ├── X_test.pkl     <- Test data for evaluation
│       └── y_test.pkl     <- Test labels for evaluation
│
├── model              <- Trained and serialized models and model artifacts
│   ├── model.pkl      <- The trained Gaussian Naive Bayes model
│   └── bag_of_words.pkl <- The bag of words vectorizer for text preprocessing
│
├── notebooks          <- Jupyter notebooks for exploration and demonstration
│   ├── 1.0-g17-exploration_of_data.ipynb <- Exploratory data analysis, separate from production code
│   └── 2.0-g17-demonstration_of_data.ipynb <- Model demonstration using production code
│
├── pyproject.toml     <- Project configuration file with package metadata
│
├── requirements.txt   <- Requirements file for reproducing the analysis environment
│
├── sentiment_model_training <- Source code for use in this project
│   ├── __init__.py    <- Makes sentiment_model_training a Python package
│   │
│   └── modeling       <- Scripts to train models and perform analysis
│       ├── __init__.py
│       ├── get_data.py  <- Script to download or generate data (equivalent to dataset.py)
│       ├── preprocess.py <- Script to transform data (equivalent to features.py)
│       ├── train.py   <- Script to train the sentiment analysis model
│       └── evaluate.py <- Script to evaluate model performance (similar to predict.py)
```

Similar to the cookiecutter template, our project separates:

- Source code (`sentiment_model_training/`)
- Data (`data/`)
- Models (`model/`)
- Documentation (`README.md`)
- Dependencies (`requirements.txt`)
- Notebooks (`notebooks/`)

The naming conventions are adapted to fit our specific sentiment analysis use case, while maintaining the logical separation of concerns of the cookiecutter template.

## Requirements

- Python 3.10+
- Restaurant Reviews dataset (automatically downloaded by the script)

### Python Dependencies

The project requires several Python packages:

- pandas - For data manipulation
- scikit-learn - For machine learning algorithms
- numpy - For numerical operations
- lib-ml - Custom library for preprocessing (from remla25-team17)
- joblib - For model serialization
- requests - For data download
- nltk - For natural language processing

## Installation

Install all dependencies using the requirements.txt file:

```bash
pip install -r requirements.txt
```

Alternatively, you can install the package in development mode:

```bash
pip install -e .
```

## Local Setup

**Clone the repository:**

```bash
git clone git@github.com:remla25-team17/model-training.git
cd model-training
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Run the complete pipeline:**

```bash
# Step 1: Download the data
python sentiment_model_training/modeling/get_data.py

# Step 2: Preprocess the data
python sentiment_model_training/modeling/preprocess.py

# Step 3: Train the model
python sentiment_model_training/modeling/train.py

# Step 4: Evaluate the model (optional)
python sentiment_model_training/modeling/evaluate.py
```

**Explore and Demonstrate with Jupyter Notebooks:**

Following cookiecutter data science principles, we separate exploratory code from production code using Jupyter notebooks

- `notebooks/exploration.ipynb` - Contains data exploration, analysis, and visualization. This notebook helps understand the dataset characteristics and inform modeling decisions but keeps exploratory code separate from production.
- `notebooks/demonstration.ipynb` - Shows how to use the trained model for predictions and evaluation. This notebook demonstrates the application of the production code rather than developing new features.

## [⚙️ GitHub Actions & CI/CD](#️-github-actions--cicd)

- **Automated ML Pipeline:**

  - Every push triggers the GitHub workflow that builds and releases the model
  - The pipeline runs the complete ML workflow: get data → preprocess → train model
  - Trained model and bag of words are automatically released

- **GitHub App Authentication:**

  For this project, we use a **GitHub App** to handle authentication in our CI/CD pipeline. Instead of relying only on GitHub's default `GITHUB_TOKEN`, which can sometimes have limited access (e.g. to trigger pre-release), the GitHub App gives us:

  - **Better security:** We can control exactly what the app is allowed to do (e.g. creating releases) without giving it more access than necessary.
  - **Reliable access:** The app works well even when we need to push images or create releases across different repositories or teams in the same organization.
  - **Clear traceability:** Every action is marked as being done by the GitHub App, so it's easy to see where changes come from.

- **Versioning:**
  - We use **GitVersion** to handle versioning automatically. GitVersion analyzes the repository's Git history and branch structure to generate a **semantic version number** (SemVer) without needing manual tagging.
  - This ensures that every build and release is consistently versioned, reducing human error and making versioning fully traceable to Git history.
  - Versioning rules:
    - Merges to `main` bump a stable version (e.g., `1.0.0`)
    - Builds from feature branches or pre-release branches (i.e., `develop`) are marked as **pre-releases** (e.g., `1.1.0-canary.5`)
  - The version is automatically injected into the package's `pyproject.toml` file during CI/CD

---

## [Resources](#-resources)

- [Cookiecutter Data Science](https://github.com/drivendataorg/cookiecutter-data-science) - The project template that inspired our structure
- [GitVersion](https://gitversion.net/) - Tool for automated versioning
- [Semantic Versioning](https://semver.org/) - Version numbering guidelines
- [GitHub App Token](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-a-user-access-token-for-a-github-app) - Documentation for GitHub App authentication

## [Use of GenAI](#-use-of-genai)

- GenAI was used to generate the structure of the README.md file.
- GenAI was used to generate the demonstration and exploration notebooks as part of an example as to how exploratory code can be separated from production code, thus following the cookiecutter template and project rubric.
