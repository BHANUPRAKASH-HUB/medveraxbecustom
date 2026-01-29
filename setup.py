from setuptools import setup, find_packages

setup(
    name="medverax",
    version="1.0.0",
    description="AI-based Health Misinformation Detection System using Machine Learning",
    author="Bhanu Prakash",
    author_email="bhanuprakash@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "fastapi",
        "uvicorn",
        "pydantic",
        "matplotlib",
        "seaborn",
        "shap"
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Education",
    ],
)
