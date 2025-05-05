# model-training

A repository for training and deploying machine learning models for sentiment analysis. 

## Table of Contents



## Features
- run.py: A script for training a gaussian naive bayes model on a dataset. This also saves the model and bag of words to disk such that it can be released in github releases.
- Fully containerized with **Docker** if you want to run it in a container.
- **GitHub Actions**:
    - pushes the model to **GitHub Releases**.
    - pushes the bag of words to **GitHub Releases**.
    - **Automatic versioning** with GitVersion.


## Requirements

- Python 3.12+
- Docker (only if you want to run it in a container)
- dataset.tsv (for training the model)

### python dependencies

- pandas
- scikit-learn
- lib-ml (git+https://github.com/remla25-team17/lib-ml.git)
- joblib

``` bash
pip install pandas scikit-learn git+https://github.com/remla25-team17/lib-ml.git joblib
```

Or you can use the requirements.txt file to install all the dependencies at once.


```bash 
pip install -r requirements.txt
```

## Local Setup
**Clone the repository:**

```bash
git clone git@github.com:remla25-team17/model-training.git
cd model-training
```

**Run the service:**

```bash
python run.py
```

## [📦 Running with Docker](#-running-with-docker)
Docker allows you to package the entire application, including its dependencies, into a single container, making it easy to deploy consistently across different environments.

**Build the Docker image:**
The following command builds the Docker image locally:

```bash
docker build -t model-training .
```

- `docker build`: This command tells Docker to create an image from the Dockerfile in the current directory.

- `-t`: sentiment-service: The -t flag tags the image with the name sentiment-service so it's easier to reference later.

- `.`: The . specifies the build context, meaning Docker will use the current directory (which should contain your Dockerfile and app code) to build the image.


**Run the container:**
Once the image is built, you can run it with: 

```bash
docker run model-training
```

- `docker run`: This starts a new container from the sentiment-service image.

- `model-training`: This specifies the image to run (the one you just built).


## [⚙️ GitHub Actions & CI/CD](#️-github-actions--cicd)

- **Build & Push:**
    - Every push to `main` or `develop/**` triggers GitHub Actions. 

- **GitHub App Authentication**

   For this project, we use a **GitHub App** to handle authentication in our CI/CD pipeline. Instead of relying only on GitHub’s default `GITHUB_TOKEN`, which can sometimes have limited access (e.g. to trigger pre-release), the GitHub App gives us:

    - **Better security:** We can control exactly what the app is allowed to do (e.g. creating releases) without giving it more access than necessary.
    - **Reliable access:** The app works well even when we need to push images or create releases across different repositories or teams in the same organization.
    - **Clear traceability:** Every action is marked as being done by the GitHub App, so it's easy to see where changes come from.

- **Versioning:**
    - We use **GitVersion** to handle versioning automatically. GitVersion analyzes the repository’s Git history and branch structure to generate a **semantic version number** (SemVer) without needing manual tagging.
    - This ensures that every build and release is consistently versioned, reducing human error and making versioning fully traceable to Git history. Moreover, it is fully automatic: commit messages simply need to specify if it is a major/minor/patch(default) and `GitVersion.yml` will automatically calculate the release version.
    - For example:
        - Merges to `main` bump a stable version (e.g., `1.0.0`).
        - Builds from feature branches or pre-release branches (i.e., `develop`) are marked as **pre-releases** (e.g., `1.1.0-canary.5`), making it clear they're not production-ready. The counter at the end of the pre-release version signifies the current number of a pre-release.
    - This approach allows us to **automate releases** and keeps versioning fully aligned with Git flow practices.

---


## [Resources](#-resources)
- [GitVersion](https://gitversion.net/)
- [Semantic Versioning](https://semver.org/)
- [GitHub App Token](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-a-user-access-token-for-a-github-app)