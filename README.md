# Intro to LangSmith

Welcome to Intro to LangSmith!

## Introduction
In this course we will walk through the fundamentals of LangSmith - exploring observability, prompt engineering, evaluations, feedback mechanisms, and production monitoring. Take a look at the setup instructions below so you can follow along with any of our notebook examples.

##### This version has been modified to be self-hosted, replacing all calls to langchain_openai with langchain_ollama and utilizing llama3.2.
---

## Setup
Follow these instructions to make sure you have all the resources necessary for this course!

### Sign up for LangSmith
* Sign up [here](https://smith.langchain.com/) 
* Navigate to the Settings page, and generate an API key in LangSmith.
* Create a .env file that mimics the provided .env.example. Set `LANGCHAIN_API_KEY` in the .env file.

### Set up Ollama & Llama 3.2
* Ensure you have `langchain_ollama` installed.
* Ensure you have Ollama installed locally (see [Ollama Quickstart Guide](https://github.com/ollama/ollama/blob/main/README.md#quickstart))


### Create an environment and install dependencies
```
$ cd intro-to-langsmith
$ python3 -m venv intro-to-ls
$ source intro-to-ls/bin/activate
$ pip install -r requirements.txt
```
Note: This fork was tested with [Ollama version 0.5.13](https://github.com/ollama/ollama/releases/tag/v0.5.13).
