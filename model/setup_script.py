from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="korean-sign-language-recognition",
    version="1.0.0",
    author="Korean Sign Language Team",
    author_email="your-email@example.com",
    description="Korean Sign Language Recognition System using OpenHands and AIHub Dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/korean-sign-language-recognition",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "intel": ["intel-extension-for-pytorch>=2.0.0", "mkl>=2023.0.0"],
        "cuda": ["torch[cuda]>=2.0.0"],
        "dev": ["pytest>=7.4.0", "black>=23.0.0", "flake8>=6.0.0"],
        "vis": ["tensorboard>=2.13.0", "wandb>=0.15.0"],
    },
    entry_points={
        "console_scripts": [
            "korean-sign=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)