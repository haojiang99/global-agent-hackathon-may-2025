#!/usr/bin/env python3
"""
Patient Record Generator for Radiation Therapy
This script generates random patient records for radiation therapy and stores them in a SQLite database.
It includes demographic information, medical history, treatment plans, and CT simulation details.

Author: Hao Jiang, UTSW Medical Center
"""

import os
import sqlite3
import random
import datetime
import uuid
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
import requests
import json

# Load API key from .env file if available
load_dotenv()

# DeepSeek API key - Will be overridden by environment variable if available
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "Use Your Key")

# Database setup
DB_FILE = "patient_records.db"

# Table definitions
TABLES = {
    "Patient": """
        CREATE TABLE IF NOT EXISTS Patient (
            patient_id TEXT PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL, 
            date_of_birth TEXT NOT NULL,
            gender TEXT NOT NULL,
            address TEXT,
            phone TEXT,
            email TEXT,
            insurance_provider TEXT,
            insurance_id TEXT,
            created_at TEXT NOT NULL
        )
    """,
    
    "Medical": """
        CREATE TABLE IF NOT EXISTS Medical (
            medical_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            primary_diagnosis TEXT NOT NULL,
            diagnosis_date TEXT NOT NULL,
            diagnosis_code TEXT,
            cancer_stage TEXT,
            tumor_location TEXT,
            tumor_size REAL,
            metastasis TEXT,
            allergies TEXT,
            medical_history TEXT,
            medications TEXT,
            previous_treatments TEXT,
            attending_physician TEXT,
            FOREIGN KEY (patient_id) REFERENCES Patient(patient_id)
        )
    """,
    
    "Plan": """
        CREATE TABLE IF NOT EXISTS Plan (
            plan_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            medical_id TEXT NOT NULL,
            treatment_type TEXT NOT NULL,
            treatment_purpose TEXT NOT NULL,
            treatment_course TEXT NOT NULL,
            num_fractions INTEGER NOT NULL,
            dose_per_fraction REAL NOT NULL,
            total_dose REAL NOT NULL,
            machine_type TEXT NOT NULL,
            machine_model TEXT,
            radiation_type TEXT NOT NULL,
            start_date TEXT,
            end_date TEXT,
            oncologist TEXT,
            notes TEXT,
            FOREIGN KEY (patient_id) REFERENCES Patient(patient_id),
            FOREIGN KEY (medical_id) REFERENCES Medical(medical_id)
        )
    """,
    
    "Simulate": """
        CREATE TABLE IF NOT EXISTS Simulate (
            simulation_id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL,
            patient_id TEXT NOT NULL,
            simulation_date TEXT NOT NULL,
            ct_scanner_vendor TEXT NOT NULL,
            ct_scanner_model TEXT,
            slice_thickness REAL NOT NULL,
            contrast_used BOOLEAN NOT NULL,
            immobilization_devices TEXT,
            reference_points TEXT,
            notes TEXT,
            FOREIGN KEY (plan_id) REFERENCES Plan(plan_id),
            FOREIGN KEY (patient_id) REFERENCES Patient(patient_id)
        )
    """,
    
    "Schedule": """
        CREATE TABLE IF NOT EXISTS Schedule (
            schedule_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            plan_id TEXT,
            appointment_date TEXT NOT NULL,
            appointment_time TEXT NOT NULL,
            appointment_type TEXT NOT NULL,
            is_treatment BOOLEAN NOT NULL,
            completed BOOLEAN NOT NULL DEFAULT 0,
            duration_minutes INTEGER,
            doctor_name TEXT,
            location TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES Patient(patient_id),
            FOREIGN KEY (plan_id) REFERENCES Plan(plan_id)
        )
    """
}

def setup_database() -> sqlite3.Connection:
    """Create SQLite database and tables if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    for table_name, create_statement in TABLES.items():
        cursor.execute(create_statement)
    
    conn.commit()
    return conn

def generate_random_date(start_year: int, end_year: int) -> str:
    """Generate a random date between start_year and end_year."""
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(end_year, 12, 31)
    days_between = (end_date - start_date).days
    random_days = random.randint(0, days_between)
    random_date = start_date + datetime.timedelta(days=random_days)
    return random_date.strftime("%Y-%m-%d")

def query_deepseek_api(prompt: str) -> str:
    """Query the DeepSeek API with a prompt and return the response."""
    url = "https://api.deepseek.com/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant skilled in generating realistic medical data."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error querying DeepSeek API: {e}")
        # Return fallback data
        return "Could not generate data from API"

def generate_patient_data() -> Dict[str, Any]:
    """Generate random patient demographic data."""
    prompt = """
    Generate JSON data for a fictional patient with the following fields:
    - first_name
    - last_name
    - date_of_birth (YYYY-MM-DD format, between 1940 and 2000)
    - gender (Male, Female, or Other)
    - address (a realistic US address)
    - phone (a realistic US phone number)
    - email
    - insurance_provider
    - insurance_id

    Return ONLY the JSON data without any additional text or explanation.
    """
    
    try:
        response = query_deepseek_api(prompt)
        data = json.loads(response)
        
        # Add patient_id and created_at
        data["patient_id"] = str(uuid.uuid4())
        data["created_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return data
    except:
        # Fallback if API fails or returns invalid JSON
        genders = ["Male", "Female", "Other"]
        insurance_providers = ["Medicare", "Medicaid", "Blue Cross", "UnitedHealth", "Cigna", "Aetna"]
        
        return {
            "patient_id": str(uuid.uuid4()),
            "first_name": random.choice(["John", "Jane", "Robert", "Mary", "Michael", "Linda"]),
            "last_name": random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller"]),
            "date_of_birth": generate_random_date(1940, 2000),
            "gender": random.choice(genders),
            "address": "123 Main St, Anytown, CA 90210",
            "phone": f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            "email": f"patient{random.randint(1000, 9999)}@example.com",
            "insurance_provider": random.choice(insurance_providers),
            "insurance_id": f"INS-{random.randint(10000000, 99999999)}",
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def generate_medical_data(patient_id: str, dob: str) -> Dict[str, Any]:
    """Generate random medical history data."""
    # Calculate age
    birth_date = datetime.datetime.strptime(dob, "%Y-%m-%d").date()
    today = datetime.date.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    
    # List of common oncologist names to use for attending physicians
    oncologists = [
        "Dr. Sarah Johnson",
        "Dr. Robert Chen",
        "Dr. Michael Rodriguez",
        "Dr. Emily Parker",
        "Dr. David Kim",
        "Dr. Lisa Wong",
        "Dr. James Wilson",
        "Dr. Amanda Taylor",
        "Dr. Richard Martinez",
        "Dr. Elizabeth Thompson"
    ]
    
    prompt = f"""
    Generate JSON data for a fictional cancer patient (age {age}) with the following fields:
    - primary_diagnosis (a specific cancer diagnosis)
    - diagnosis_date (YYYY-MM-DD format, within the last 2 years)
    - diagnosis_code (an ICD-10 code for the cancer)
    - cancer_stage (e.g., "Stage I", "Stage II", "Stage IIIA", etc.)
    - tumor_location (specific anatomical location)
    - tumor_size (in cm)
    - metastasis (Yes/No and if Yes, where)
    - allergies (comma-separated list or "None")
    - medical_history (comma-separated list of previous conditions)
    - medications (comma-separated list of current medications)
    - previous_treatments (comma-separated list of any cancer treatments already received, if any)
    - attending_physician (full name with title, e.g., "Dr. Jane Smith")

    Return ONLY the JSON data without any additional text or explanation.
    """
    
    try:
        response = query_deepseek_api(prompt)
        data = json.loads(response)
        
        # Add medical_id and patient_id
        data["medical_id"] = str(uuid.uuid4())
        data["patient_id"] = patient_id
        
        # Ensure attending_physician is set
        if not data.get("attending_physician"):
            data["attending_physician"] = random.choice(oncologists)
            
        return data
    except:
        # Fallback if API fails or returns invalid JSON
        cancer_types = ["Lung Adenocarcinoma", "Breast Ductal Carcinoma", "Prostate Adenocarcinoma", 
                        "Colorectal Adenocarcinoma", "Lymphoma", "Melanoma"]
        cancer_stages = ["Stage I", "Stage II", "Stage IIIA", "Stage IIIB", "Stage IV"]
        
        return {
            "medical_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "primary_diagnosis": random.choice(cancer_types),
            "diagnosis_date": generate_random_date(datetime.date.today().year - 2, datetime.date.today().year),
            "diagnosis_code": f"C{random.randint(10, 99)}.{random.randint(0, 9)}",
            "cancer_stage": random.choice(cancer_stages),
            "tumor_location": random.choice(["Left lung", "Right breast", "Prostate", "Sigmoid colon", "Lymph nodes", "Back"]),
            "tumor_size": round(random.uniform(0.5, 10.0), 1),
            "metastasis": random.choice(["No", "Yes, liver", "Yes, bone", "Yes, brain", "Yes, lymph nodes"]),
            "allergies": random.choice(["None", "Penicillin", "Contrast dye, Sulfa drugs", "Latex"]),
            "medical_history": "Hypertension, Type 2 Diabetes",
            "medications": "Metformin, Lisinopril, Simvastatin",
            "previous_treatments": random.choice(["None", "Surgery", "Chemotherapy", "Surgery, Chemotherapy"]),
            "attending_physician": random.choice(oncologists)
        }

def generate_plan_data(patient_id: str, medical_id: str, primary_diagnosis: str) -> Dict[str, Any]:
    """Generate random radiation treatment plan data."""
    prompt = f"""
    Generate JSON data for a radiation therapy treatment plan for a patient with {primary_diagnosis} with the following fields:
    - treatment_type (specific radiation therapy technique like IMRT, VMAT, 3DCRT, etc.)
    - treatment_purpose (curative, palliative, adjuvant, etc.)
    - treatment_course (initial, boost, re-treatment)
    - num_fractions (integer between 1 and 35)
    - dose_per_fraction (in Gy, typically between 1.8 and 8.0)
    - total_dose (calculated as num_fractions * dose_per_fraction)
    - machine_type (Linac, TomoTherapy, CyberKnife, Gamma Knife, etc.)
    - machine_model (a realistic model name for the machine_type)
    - radiation_type (photons, electrons, protons)
    - start_date (YYYY-MM-DD format, within the next month)
    - end_date (YYYY-MM-DD format, calculated based on fractions and start date)
    - oncologist (a fictional doctor name)
    - notes (a brief clinical note about the treatment plan)

    Return ONLY the JSON data without any additional text or explanation.
    """
    
    try:
        response = query_deepseek_api(prompt)
        data = json.loads(response)
        
        # Add plan_id, patient_id, and medical_id
        data["plan_id"] = str(uuid.uuid4())
        data["patient_id"] = patient_id
        data["medical_id"] = medical_id
        
        # Verify total_dose is correct
        data["total_dose"] = round(data["num_fractions"] * data["dose_per_fraction"], 1)
        
        return data
    except:
        # Fallback if API fails or returns invalid JSON
        treatment_types = ["IMRT", "VMAT", "3DCRT", "SBRT", "SRS"]
        purposes = ["Curative", "Palliative", "Adjuvant", "Neoadjuvant"]
        courses = ["Initial", "Boost", "Re-treatment"]
        machine_types = ["Linac", "TomoTherapy", "CyberKnife", "Gamma Knife"]
        machine_models = {
            "Linac": ["TrueBeam", "Halcyon", "Versa HD", "Edge"],
            "TomoTherapy": ["TomoHD", "TomoTherapy H"],
            "CyberKnife": ["CyberKnife M6", "CyberKnife S7"],
            "Gamma Knife": ["Gamma Knife Icon", "Gamma Knife Perfexion"]
        }
        radiation_types = ["Photons", "Electrons", "Protons"]
        
        # Choose machine type and then appropriate model
        machine_type = random.choice(machine_types)
        machine_model = random.choice(machine_models[machine_type])
        
        # Generate num_fractions and dose_per_fraction
        num_fractions = random.randint(1, 35)
        dose_per_fraction = round(random.uniform(1.8, 8.0), 1)
        total_dose = round(num_fractions * dose_per_fraction, 1)
        
        # Generate start_date and end_date
        today = datetime.date.today()
        start_date = today + datetime.timedelta(days=random.randint(7, 30))
        end_date = start_date + datetime.timedelta(days=num_fractions + random.randint(0, 10))
        
        return {
            "plan_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "medical_id": medical_id,
            "treatment_type": random.choice(treatment_types),
            "treatment_purpose": random.choice(purposes),
            "treatment_course": random.choice(courses),
            "num_fractions": num_fractions,
            "dose_per_fraction": dose_per_fraction,
            "total_dose": total_dose,
            "machine_type": machine_type,
            "machine_model": machine_model,
            "radiation_type": random.choice(radiation_types),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "oncologist": f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'])}",
            "notes": f"Standard {random.choice(treatment_types)} treatment for {primary_diagnosis}."
        }

def generate_simulation_data(patient_id: str, plan_id: str) -> Dict[str, Any]:
    """Generate random CT simulation data."""
    prompt = """
    Generate JSON data for a CT simulation for radiation therapy with the following fields:
    - simulation_date (YYYY-MM-DD format, within the last 2 weeks)
    - ct_scanner_vendor (a realistic CT scanner manufacturer)
    - ct_scanner_model (a realistic model for the vendor)
    - slice_thickness (in mm, typically between 1.0 and 5.0)
    - contrast_used (true or false)
    - immobilization_devices (comma-separated list of devices used)
    - reference_points (description of reference marks or tattoos placed)
    - notes (a brief clinical note about the simulation)

    Return ONLY the JSON data without any additional text or explanation.
    """
    
    try:
        response = query_deepseek_api(prompt)
        data = json.loads(response)
        
        # Add simulation_id, patient_id, and plan_id
        data["simulation_id"] = str(uuid.uuid4())
        data["patient_id"] = patient_id
        data["plan_id"] = plan_id
        
        # Convert contrast_used to integer for SQLite BOOLEAN
        if isinstance(data["contrast_used"], bool):
            data["contrast_used"] = 1 if data["contrast_used"] else 0
        elif isinstance(data["contrast_used"], str):
            data["contrast_used"] = 1 if data["contrast_used"].lower() == "true" else 0
        
        return data
    except:
        # Fallback if API fails or returns invalid JSON
        ct_vendors = ["Siemens", "GE Healthcare", "Philips", "Canon Medical", "United Imaging"]
        ct_models = {
            "Siemens": ["SOMATOM Force", "SOMATOM Edge Plus", "SOMATOM go.Top"],
            "GE Healthcare": ["Revolution CT", "Revolution Apex", "Optima CT660"],
            "Philips": ["Brilliance CT Big Bore", "IQon Spectral CT", "Access CT"],
            "Canon Medical": ["Aquilion ONE", "Aquilion Precision", "Aquilion Lightning"],
            "United Imaging": ["uCT 960+", "uCT 780", "uCT 530"]
        }
        
        immobilization_devices = [
            "Thermoplastic mask, Head and neck support",
            "Vacuum cushion, Knee support",
            "Wing board, Breast board",
            "Body fix vacuum cushion",
            "Ankle lock, Knee support"
        ]
        
        reference_points = [
            "Three tattoos placed: one anterior and two lateral",
            "BBs placed at laser intersection points",
            "Four reference marks with permanent tattoos",
            "Temporary marks with BBs for CBCT verification",
            "Three tattoos placed on body landmarks"
        ]
        
        # Choose vendor and then appropriate model
        vendor = random.choice(ct_vendors)
        model = random.choice(ct_models[vendor])
        
        # Generate simulation date
        today = datetime.date.today()
        sim_date = today - datetime.timedelta(days=random.randint(1, 14))
        
        return {
            "simulation_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "plan_id": plan_id,
            "simulation_date": sim_date.strftime("%Y-%m-%d"),
            "ct_scanner_vendor": vendor,
            "ct_scanner_model": model,
            "slice_thickness": round(random.uniform(1.0, 5.0), 1),
            "contrast_used": random.choice([0, 1]),
            "immobilization_devices": random.choice(immobilization_devices),
            "reference_points": random.choice(reference_points),
            "notes": "Standard CT simulation completed without complications."
        }

def insert_patient(conn: sqlite3.Connection, patient_data: Dict[str, Any]) -> None:
    """Insert patient data into the Patient table."""
    cursor = conn.cursor()
    
    placeholders = ", ".join(["?"] * len(patient_data))
    columns = ", ".join(patient_data.keys())
    values = tuple(patient_data.values())
    
    sql = f"INSERT INTO Patient ({columns}) VALUES ({placeholders})"
    cursor.execute(sql, values)
    conn.commit()

def insert_medical(conn: sqlite3.Connection, medical_data: Dict[str, Any]) -> None:
    """Insert medical data into the Medical table."""
    cursor = conn.cursor()
    
    placeholders = ", ".join(["?"] * len(medical_data))
    columns = ", ".join(medical_data.keys())
    values = tuple(medical_data.values())
    
    sql = f"INSERT INTO Medical ({columns}) VALUES ({placeholders})"
    cursor.execute(sql, values)
    conn.commit()

def insert_plan(conn: sqlite3.Connection, plan_data: Dict[str, Any]) -> None:
    """Insert plan data into the Plan table."""
    cursor = conn.cursor()
    
    placeholders = ", ".join(["?"] * len(plan_data))
    columns = ", ".join(plan_data.keys())
    values = tuple(plan_data.values())
    
    sql = f"INSERT INTO Plan ({columns}) VALUES ({placeholders})"
    cursor.execute(sql, values)
    conn.commit()

def insert_simulation(conn: sqlite3.Connection, simulation_data: Dict[str, Any]) -> None:
    """Insert simulation data into the Simulate table."""
    cursor = conn.cursor()
    
    placeholders = ", ".join(["?"] * len(simulation_data))
    columns = ", ".join(simulation_data.keys())
    values = tuple(simulation_data.values())
    
    sql = f"INSERT INTO Simulate ({columns}) VALUES ({placeholders})"
    cursor.execute(sql, values)
    conn.commit()

def generate_schedule_data(patient_id: str, plan_id: str, start_date: str, num_fractions: int) -> List[Dict[str, Any]]:
    """Generate random schedule data for a patient, including treatment appointments."""
    prompt = f"""
    Generate JSON array data for scheduling appointments for a radiation therapy patient with the following fields:
    - appointment_date (YYYY-MM-DD format, starting from {start_date} and scheduled for {num_fractions} treatments plus 2-3 doctor visits)
    - appointment_time (HH:MM format, each appointment should be at a specific time)
    - appointment_type (e.g., "Treatment", "Doctor Visit", "Follow-up", "Planning", etc.)
    - is_treatment (boolean, true for treatment appointments, false for doctor visits and follow-ups)
    - duration_minutes (integer, typically 15-30 minutes for treatments, 30-60 for doctor visits)
    - doctor_name (a realistic doctor name)
    - location (e.g., "Radiation Oncology - Room 102", "Main Hospital - Clinic 3", etc.)
    - notes (brief notes about the appointment)
    - completed (boolean, set the first few appointments to true and the rest to false)

    Return ONLY the JSON array data without any additional text or explanation.
    """
    
    try:
        response = query_deepseek_api(prompt)
        schedule_data_list = json.loads(response)
        
        # Convert to list if not already
        if isinstance(schedule_data_list, dict):
            schedule_data_list = [schedule_data_list]
        
        # Add IDs and patient_id, plan_id
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for schedule_data in schedule_data_list:
            schedule_data["schedule_id"] = str(uuid.uuid4())
            schedule_data["patient_id"] = patient_id
            
            # Only add plan_id to treatment appointments
            if schedule_data.get("is_treatment", False):
                schedule_data["plan_id"] = plan_id
            else:
                schedule_data["plan_id"] = None
                
            schedule_data["created_at"] = current_datetime
            
            # Convert boolean strings to actual booleans if needed
            if isinstance(schedule_data.get("is_treatment"), str):
                schedule_data["is_treatment"] = schedule_data["is_treatment"].lower() == "true"
            
            if isinstance(schedule_data.get("completed"), str):
                schedule_data["completed"] = schedule_data["completed"].lower() == "true"
        
        return schedule_data_list
    except:
        # Fallback if API fails or returns invalid JSON
        schedule_data_list = []
        
        # Parse the start date
        treatment_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        
        # Generate doctor appointment before treatment
        planning_date = treatment_date - datetime.timedelta(days=random.randint(7, 14))
        schedule_data_list.append({
            "schedule_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "plan_id": None,
            "appointment_date": planning_date.strftime("%Y-%m-%d"),
            "appointment_time": f"{random.randint(8, 16):02d}:{random.choice(['00', '15', '30', '45'])}",
            "appointment_type": "Planning Consultation",
            "is_treatment": False,
            "completed": True,
            "duration_minutes": random.choice([30, 45, 60]),
            "doctor_name": f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'])}",
            "location": f"Main Hospital - Clinic {random.randint(1, 5)}",
            "notes": "Initial planning consultation for radiation therapy",
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Generate treatment appointments
        for i in range(num_fractions):
            # Treatment every weekday
            while treatment_date.weekday() >= 5:  # Skip weekend
                treatment_date += datetime.timedelta(days=1)
                
            schedule_data_list.append({
                "schedule_id": str(uuid.uuid4()),
                "patient_id": patient_id,
                "plan_id": plan_id,
                "appointment_date": treatment_date.strftime("%Y-%m-%d"),
                "appointment_time": f"{random.randint(8, 16):02d}:{random.choice(['00', '15', '30', '45'])}",
                "appointment_type": "Treatment",
                "is_treatment": True,
                "completed": i < min(3, num_fractions),  # First few are completed
                "duration_minutes": random.choice([15, 20, 30]),
                "doctor_name": f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'])}",
                "location": f"Radiation Oncology - Room {random.randint(101, 110)}",
                "notes": f"Treatment session {i+1} of {num_fractions}",
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            treatment_date += datetime.timedelta(days=1)
            
        # Generate follow-up appointment after treatment
        followup_date = treatment_date + datetime.timedelta(days=random.randint(14, 28))
        schedule_data_list.append({
            "schedule_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "plan_id": None,
            "appointment_date": followup_date.strftime("%Y-%m-%d"),
            "appointment_time": f"{random.randint(8, 16):02d}:{random.choice(['00', '15', '30', '45'])}",
            "appointment_type": "Follow-up",
            "is_treatment": False,
            "completed": False,
            "duration_minutes": random.choice([30, 45, 60]),
            "doctor_name": f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'])}",
            "location": f"Main Hospital - Clinic {random.randint(1, 5)}",
            "notes": "Post-treatment follow-up appointment",
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return schedule_data_list

def insert_schedule(conn: sqlite3.Connection, schedule_data: Dict[str, Any]) -> None:
    """Insert schedule data into the Schedule table."""
    cursor = conn.cursor()
    
    placeholders = ", ".join(["?"] * len(schedule_data))
    columns = ", ".join(schedule_data.keys())
    values = tuple(schedule_data.values())
    
    sql = f"INSERT INTO Schedule ({columns}) VALUES ({placeholders})"
    cursor.execute(sql, values)
    conn.commit()

def generate_patient_record() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Generate a complete patient record."""
    patient_data = generate_patient_data()
    medical_data = generate_medical_data(patient_data["patient_id"], patient_data["date_of_birth"])
    plan_data = generate_plan_data(patient_data["patient_id"], medical_data["medical_id"], medical_data["primary_diagnosis"])
    simulation_data = generate_simulation_data(patient_data["patient_id"], plan_data["plan_id"])
    schedule_data_list = generate_schedule_data(
        patient_data["patient_id"], 
        plan_data["plan_id"], 
        plan_data["start_date"], 
        plan_data["num_fractions"]
    )
    
    return patient_data, medical_data, plan_data, simulation_data, schedule_data_list

def main():
    """Main function to generate and insert a patient record."""
    print("Setting up database...")
    conn = setup_database()
    
    print("Generating patient record...")
    patient_data, medical_data, plan_data, simulation_data, schedule_data_list = generate_patient_record()
    
    print("Inserting data into database...")
    insert_patient(conn, patient_data)
    insert_medical(conn, medical_data)
    insert_plan(conn, plan_data)
    insert_simulation(conn, simulation_data)
    
    print("Inserting schedule data...")
    for schedule_data in schedule_data_list:
        insert_schedule(conn, schedule_data)
    
    print(f"Patient record generated and stored in database: {DB_FILE}")
    print(f"Patient ID: {patient_data['patient_id']}")
    print(f"Patient Name: {patient_data['first_name']} {patient_data['last_name']}")
    print(f"Diagnosis: {medical_data['primary_diagnosis']}")
    print(f"Attending Physician: {medical_data.get('attending_physician', 'Not specified')}")
    print(f"Treatment: {plan_data['treatment_type']} with {plan_data['num_fractions']} fractions")
    print(f"Schedule: {len(schedule_data_list)} appointments created")
    
    conn.close()

if __name__ == "__main__":
    main()