# ChatWithPDF
> Upload the pdf to the LLM and start to chat with it 

# Installation
## Dependencies
ChatWithPDF requires :
1. numpy
2. pandas
3. spacy
4. tqdm
5. PyMuPDF
6. torch
7. sentence_transformers
8. transformers
9. gradio
ChatWithPDF is built on python 3.11.

## User Installation
ChatWithPDF is currently available in the github and it is planned to release as a pip package. Various tests and contribtution is required 🤗
To run this app in your computer locally,
1. Clone this github repository
```bash
git clone https://github.com/JaiSuryaPrabu/ChatWithPDF
```
2. Move to the `ChatWithPDF` folder
```bash
cd ChatWithPDF
```
3. To install the required packages
```bash
pip install -r requirements.txt
```
4. To build the app locally
```
python setup.py build sdist bdist
pip install -e .
```
5. To run the app locally
```
chat 
```
Once it started to execute, open the browser and enter this `http://127.0.0.1:7860/`.
And the app will starts to run on your browser locally.

# Contributions
I am very glad everyone from beginners to experts to contribute this project to make it better !
Below are some guidelines to help you get started:
## How to contribute 
1. **Fork this Repository** : Click the "Fork" button at the top right corner of this repository to create your own copy
2. **Follow the user installation steps** to install locally on your machine
3. **Create a new branch** : Create a new branch for your contribution. Use a descriptive name for the branch (e.g., `feature/new-feature-name`)
4. **Make Changes** : Makd your desired changes to the codebase. You can add new features, fix bugs or improve the performance and documentations.
5. **Commit your changes** : Commit your changes with a clear and consice commit message :
```
git add .
git commit -m "Add feature : [description]
```
6. **Push to your fork** : Push your changes to your forked repository
```
git push origin your-branch-name
```
7. **Create a pull request (PR)** : Go to the original repository and create a new pull request. Provide a detailed description of your changes and why they are valuable.
 