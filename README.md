# Judgmental AI System - Graduation Project

## Overview
The Judgmental AI System is a cutting-edge application designed to assist in legal case analysis and decision-making. This project leverages advanced AI models and a robust architecture to process legal documents, analyze cases, and provide insights to support judicial processes.

## Features
- **Case Analysis**: Automates the analysis of legal cases using AI-driven pipelines.
- **Graph-Based Knowledge Representation**: Utilizes graph structures to represent and process legal knowledge.
- **Agent-Based Architecture**: Includes specialized agents for tasks such as data ingestion, evidence analysis, and procedural auditing.
- **FastAPI Integration**: Provides a RESTful API for seamless interaction with the system.

## Project Structure
```
GraduationProject/
├── main.py                # Entry point for the application
├── src/                   # Source code for the system
│   ├── agents/            # Specialized agents for various tasks
│   ├── Chunking/          # Chunking utilities for data processing
│   ├── Config/            # Configuration files
│   ├── Graph/             # Graph-based knowledge representation
│   ├── Graphstore/        # Graph storage utilities
│   ├── LLMs/              # Large Language Model integrations
│   ├── Prompts/           # Prompt templates for agents
│   ├── routers/           # API routers
│   └── Utils/             # Utility functions
├── Datasets/              # Structured and unstructured legal datasets
├── Experiments/           # Experimentation notebooks and scripts
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
└── README.md              # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mohamedGaber2004/EL-MOSTASHAR-Graduation-Project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd EL-MOSTASHAR-Graduation-Project
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the application:
   ```bash
   python main.py
   ```
2. Access the API documentation at:
   ```
   http://127.0.0.1:8000/docs
   ```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- **Team Members**: Mohamed Gaber, Ahmed Hamdy, Omar Youssef, and others.
- **Advisors**: Special thanks to our academic advisors for their guidance.
- **Resources**: Leveraged open-source libraries and datasets to build this system.
