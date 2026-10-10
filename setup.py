"""Setup configuration for the tesis_ia package."""

from pathlib import Path

from setuptools import find_packages, setup

BASE_DIR = Path(__file__).parent
README_PATH = BASE_DIR / "README.md"
REQUIREMENTS_PATH = BASE_DIR / "requirements.txt"

long_description = README_PATH.read_text(encoding="utf-8")
requirements = [
    line.strip()
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="tesis_ia",
    version="0.1.0",
    author="Edgardo Crespo y Felipe Calderon",
    description="Base estructural para un proyecto de tesis de inteligencia artificial",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/tesis_ia",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
