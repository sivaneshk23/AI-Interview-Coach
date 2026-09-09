# System Architecture

## High-Level Architecture

User
  |
  v
Candidate Profile
  |
  v
Interview Trainer Agent
  |
  +--------------------+
  |                    |
  v                    v
Knowledge/RAG       Candidate Data
  |                    |
  +---------+----------+
            |
            v
      Interview Planner
            |
     +------+------+------+------+------+
     |      |      |      |      |      |
     v      v      v      v      v      v
 Aptitude Technical Communication HR Resume Evaluation
     |      |      |      |      |
     +------+------+------+------+------+
            |
            v
      Feedback Generator
            |
            v
      Performance Report

## Core Components

### 1. Candidate Profile

Stores information required to personalize the interview.

### 2. RAG Knowledge Layer

Retrieves relevant knowledge from approved interview-preparation sources.

### 3. Primary Interview Agent

Controls the overall conversation and determines the appropriate interview
workflow.

### 4. Specialized Agents

Each specialized component focuses on one interview domain.

### 5. Evaluation Layer

Evaluates candidate responses against defined criteria.

### 6. Feedback Layer

Converts evaluation results into actionable recommendations.

## Agent Flow

User Request
    |
    v
Primary Agent
    |
    +--> Identify interview type
    |
    +--> Retrieve relevant knowledge
    |
    +--> Select specialized capability
    |
    +--> Generate question
    |
    +--> Receive candidate answer
    |
    +--> Evaluate answer
    |
    +--> Generate feedback
    |
    +--> Determine next question
    |
    v
Final Performance Summary