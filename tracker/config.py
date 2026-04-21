import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CURRICULUM_DIR = os.path.join(BASE_DIR, "curriculum")
EXERCISES_DIR = os.path.join(BASE_DIR, "exercises")
PROGRESS_FILE = os.path.join(DATA_DIR, "progress.json")
PROGRESS_TEMPLATE = os.path.join(DATA_DIR, "progress.template.json")
