from setuptools import setup, find_packages

setup(
    name="model-training",
    version="1.0.0",
    packages=find_packages(),
    entry_points={
        "pylint.plugins": [
            "unnecessary-iteration-checker = lint.custom_rules"
        ]
    },
)