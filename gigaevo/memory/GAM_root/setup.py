#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安装脚本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="general-agentic-memory",
    version="0.1.0",
    author="GAM Team",
    author_email="zhengliu1026@gmail.com",
    description="A general memory system for agents, powered by deep-research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VectorSpaceLab/general-agentic-memory",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "gam-eval=eval.run:main",
        ],
    },
)