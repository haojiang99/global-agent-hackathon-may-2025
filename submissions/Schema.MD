# Radiation Therapy Patient Database Schema

## Patient Table
```sql
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
```

## Medical Table
```sql
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
```

## Plan Table
```sql
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
```

## Simulate Table
```sql
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
```

## Schedule Table
```sql
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
```

## Common Queries

### Get all patients
```sql
SELECT * FROM Patient;
```

### Get patient diagnoses
```sql
SELECT 
    first_name, 
    last_name, 
    primary_diagnosis, 
    cancer_stage 
FROM Patient 
JOIN Medical ON Patient.patient_id = Medical.patient_id;
```

### Get treatment plans
```sql
SELECT 
    first_name, 
    last_name, 
    treatment_type, 
    num_fractions, 
    total_dose 
FROM Patient 
JOIN Plan ON Patient.patient_id = Plan.patient_id;
```

### Get simulation details
```sql
SELECT 
    p.first_name, 
    p.last_name, 
    s.simulation_date, 
    s.ct_scanner_vendor, 
    s.ct_scanner_model 
FROM Patient p
JOIN Simulate s ON p.patient_id = s.patient_id;
```

### Get complete patient record
```sql
SELECT 
    p.patient_id, 
    p.first_name, 
    p.last_name, 
    p.date_of_birth, 
    p.gender,
    m.primary_diagnosis, 
    m.cancer_stage,
    pl.treatment_type, 
    pl.treatment_purpose,
    pl.num_fractions,
    pl.total_dose,
    s.simulation_date,
    s.ct_scanner_vendor
FROM Patient p
LEFT JOIN Medical m ON p.patient_id = m.patient_id
LEFT JOIN Plan pl ON p.patient_id = pl.patient_id
LEFT JOIN Simulate s ON p.patient_id = s.patient_id
WHERE p.patient_id = ?;
```

### Get patient's upcoming appointments
```sql
SELECT 
    p.first_name, 
    p.last_name, 
    s.appointment_date,
    s.appointment_time,
    s.appointment_type,
    s.is_treatment,
    s.doctor_name,
    s.location
FROM Schedule s
JOIN Patient p ON s.patient_id = p.patient_id
WHERE s.patient_id = ? 
  AND s.appointment_date >= DATE('now')
  AND s.completed = 0
ORDER BY s.appointment_date, s.appointment_time;
```

### Get today's treatment appointments
```sql
SELECT 
    p.first_name, 
    p.last_name, 
    s.appointment_time,
    pl.treatment_type,
    s.location
FROM Schedule s
JOIN Patient p ON s.patient_id = p.patient_id
LEFT JOIN Plan pl ON s.plan_id = pl.plan_id
WHERE s.appointment_date = DATE('now')
  AND s.is_treatment = 1
  AND s.completed = 0
ORDER BY s.appointment_time;
```

### Get treatment history for patient
```sql
SELECT 
    s.appointment_date,
    s.appointment_time,
    pl.treatment_type,
    pl.machine_type,
    s.doctor_name,
    s.completed
FROM Schedule s
JOIN Plan pl ON s.plan_id = pl.plan_id
WHERE s.patient_id = ?
  AND s.is_treatment = 1
ORDER BY s.appointment_date DESC, s.appointment_time DESC;
```

### Get patients by attending physician
```sql
SELECT 
    p.patient_id,
    p.first_name,
    p.last_name,
    p.date_of_birth,
    m.primary_diagnosis,
    m.diagnosis_date,
    m.cancer_stage
FROM Patient p
JOIN Medical m ON p.patient_id = m.patient_id
WHERE m.attending_physician = ?
ORDER BY p.last_name, p.first_name;
```

### Get all attending physicians with patient counts
```sql
SELECT 
    m.attending_physician,
    COUNT(DISTINCT m.patient_id) AS patient_count
FROM Medical m
GROUP BY m.attending_physician
ORDER BY patient_count DESC;
```