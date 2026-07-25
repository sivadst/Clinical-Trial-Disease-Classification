from setuptools import setup, find_packages

setup(
    name="clinical_trial_disease_classification",
    version="1.0.0",
    description="Clinical Trial Disease Category Classification",
    author="Google Jules",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)
