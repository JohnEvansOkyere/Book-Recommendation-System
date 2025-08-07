# Book-Recommendation-System

## Workflow
- config.yaml
- entity
- config/configuration.py
- components
- pipeline
- main.py
- app.py

Here's a more appropriate README section for an ML project:

# How to Run the Machine Learning Project

## Prerequisites

Before running the code, ensure you have:

- Python 3.7+ (recommended 3.8/3.9)
- pip or conda package manager
- Recommended: Virtual environment (venv, conda, or virtualenv)

## Setup Instructions

### 1. Clone the Repository
```bash
git https://github.com/JohnEvansOkyere/Book-Recommendation-System.git

```

### 2. Create and Activate Virtual Environment (Recommended)

#### Option 1: Using venv
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

#### Option 2: Using conda
```bash
conda create -n ml_project python=3.8
conda activate ml_project
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

fisrt
use template.py to create project structure

second
write in setup.py to set up your project structure

third
call the setup from setup.py in requirements as -e .

4th
Write in exception.py - This script gives the line number, file name and specific line of error

5th
Move to logger folder-  and write script for log.py - Keeps tracks of all errors in production

6th
Move to utils - the functions we will be using mostly in development, means common functions we will be using, so that I will not have to write the function again and again

7th
Constant - Getting the paths of config.yaml files

8th
Config.yaml
1. Created data ingestion path

9th
Move to entity file - Return type of input/ Function

10th
Move to configuration.py - Return variable one-by-one

11th
Now we start with the ingestion process by moving to data_ingestio.py

12th
Move to the pipeline folder
All the process should 

13th
After the pipeline, call the pipeline in main.py
then run main.py for data extraction

14th
After ingestion, then data_validation

15th
Followed by data transformation