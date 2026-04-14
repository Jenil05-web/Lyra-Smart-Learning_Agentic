# 🎓 Personal Study Assistant

An agentic learning platform built with **LangGraph** and **Streamlit** that transforms PDFs into interactive, personalized educational journeys.

---

## 🚀 Overview
This assistant leverages a multi-agent system to analyze documents, build a logical curriculum, teach concepts via RAG, and validate knowledge through dynamic assessments.

## 🧠 Agent Architecture
Powered by **LangGraph**, the system orchestrates three specialized agents:
1. **The Architect:** Analyzes PDFs and generates a step-by-step learning roadmap.
2. **The Tutor:** Delivers interactive lessons using PDF context and Wikipedia search.
3. **The Examiner:** Generates and evaluates quizzes to ensure concept mastery.

## ✨ Key Features
* **PDF Intelligence:** Upload study material to seed the knowledge base.
* **Adaptive Learning Paths:** Automatically plans steps based on content complexity.
* **Wikipedia Integration:** Fetches external context to supplement PDF data.
* **Interactive Chat:** Fluid, conversational interface for deep-diving into topics.
* **Knowledge Verification:** Built-in quiz module to test understanding.

## 🛠️ Tech Stack
* **Orchestration:** LangGraph, LangChain , Python
* **Frontend:** Streamlit
* **LLM Framework:** OpenAI , Gpt -3.5-pro
* **Tools:** Wikipedia API, FAISS

