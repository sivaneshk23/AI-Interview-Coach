# AI Interview Coach

AI Interview Coach is an end-to-end AI-powered interview preparation system built using **IBM watsonx.ai** and **IBM Granite** foundation models. The system uses **Retrieval-Augmented Generation (RAG)** and specialized AI agents to generate role-specific interview questions, evaluate candidate responses, and provide personalized improvement feedback.

The project was developed as part of the **IBM SkillsBuild · Edunet · AICTE internship programme**.

---

## What It Does

AI Interview Coach helps users prepare for job interviews through an adaptive interview experience.

### Key Features

- Supports **any job role** through free-form role input.
- Generates role-specific interview questions using **IBM Granite**.
- Uses **RAG with FAISS** to ground interview questions and evaluations using a local interview knowledge base.
- Conducts **adaptive multi-turn mock interviews**.
- Evaluates answers across multiple dimensions:
  - Overall performance
  - Technical knowledge
  - Relevance
  - Clarity
  - Communication
  - Completeness
- Provides the **next interview question based on the ongoing session**.
- Maintains interview sessions using **SQLite**.
- Generates a final performance report containing:
  - Overall score
  - Strengths
  - Weaknesses
  - Improvement suggestions
- Provides a simple browser-based interview interface.
- Includes REST APIs and interactive **FastAPI Swagger documentation**.
- Includes automated tests using **pytest**.

---

## Problem Statement

### Problem Statement No. 22 – Interview Trainer Agent

Preparing for job interviews can be difficult because candidates often struggle to find role-specific questions, understand industry expectations, practice realistic interview conversations, and receive meaningful feedback on their answers.

The **Interview Trainer Agent** addresses this problem by using Retrieval-Augmented Generation and IBM Granite models to create a personalized interview preparation system.

Users can provide their **name, experience level, and target job role**. The system retrieves relevant information from its interview knowledge base and generates tailored interview questions. During the mock interview, the candidate's responses are evaluated and personalized feedback is provided.

The objective is to help candidates improve their technical knowledge, communication, relevance, clarity, and completeness before attending real interviews.

---

## Solution

The proposed solution is an **AI-powered Interview Coach** that combines **Agentic AI, RAG, IBM Granite, and session-based interview management**.

The system consists of two specialized AI agents:

### 1. Interviewer Agent

The Interviewer Agent uses **IBM Granite through IBM watsonx.ai** to generate relevant interview questions based on:

- Target job role
- Experience level
- Previous answers
- Retrieved knowledge from the RAG system

The agent supports an adaptive multi-turn interview instead of generating unrelated questions independently.

### 2. Evaluator Agent

The Evaluator Agent analyzes each candidate response using IBM Granite and produces structured evaluation results.

It evaluates:

- Overall performance
- Technical knowledge
- Relevance
- Clarity
- Communication
- Completeness

It also provides actionable feedback that can help the candidate improve.

### 3. RAG Engine

The RAG Engine retrieves relevant information from the project's local interview knowledge base.

The documents are processed into embeddings and stored in a **FAISS vector index**. Relevant context is retrieved before generating interview questions or evaluations.

This helps the system produce more role-relevant and grounded responses.

### 4. Session Management

Interview sessions are persisted using **SQLite**.

This allows the system to maintain:

- Candidate information
- Interview questions
- Submitted answers
- Evaluation results
- Interview progress
- Final performance summary

---

## Architecture

```text
                    Browser / API Client
                           │
                           ▼
                    FastAPI Application
                      (app/main.py)
                           │
                           ▼
                    Interview Engine
                (app/interview_engine.py)
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
      Interviewer      Evaluator      RAG Engine
         Agent            Agent           │
             │             │              ▼
             │             │        FAISS Vector
             │             │           Store
             │             │              │
             │             │              ▼
             │             │       Interview Knowledge
             │             │          Base Documents
             │             │
             └─────────────┼──────────────┘
                           │
                           ▼
                     IBM watsonx.ai
                           │
                           ▼
                     IBM Granite
                           │
                           ▼
                     Session Store
                        SQLite

Built with IBM Bob — IBM SkillsBuild · Edunet · AICTE internship project
