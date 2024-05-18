from setuptools import setup,find_packages

name = "ChatWithPDF"
version = "0.0.2"
description = "Chat with your PDF"
author = "Jaisurya"
email = "pjaisurya@gmail.com"
url = "https://github.com/JaiSuryaPrabu/ChatWithPDF"

setup(
    name = name,
    version = version,
    description = description,
    author = author,
    author_email = email,
    url = url,
    packages = find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "spacy",
        "tqdm",
        "PyMuPDF",
        "torch",
        "sentence_transformers",
        "transformers",
        "gradio"
    ],
    entry_points = {
        "console_scripts":[
            "chat=ChatWithPDF.app:main"
        ]
    },
    classifiers=[
       "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ]
)