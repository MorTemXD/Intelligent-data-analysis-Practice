import psycopg2
import random
from faker import Faker
from contextlib import contextmanager
from datetime import date, timedelta

DB_CONFIG = {
    "dbname": "health_records_db",
    "user": "postgres",
    "password": "1111",
    "host": "localhost",
    "port": "5432"
}

fake = Faker('uk_UA')
NUM_RECORDS = 1000
NUM_DOCTORS = 50
NUM_PATIENTS = 200

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
        conn.commit()
    except Exception as e:
        print(f"Помилка підключення до бази даних: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def create_tables(conn):
    print("Створення/очищення таблиць...")
    with conn.cursor() as cur:
        tables = [
            "health_records", 
            "registered_cases", 
            "patients", 
            "doctors", 
            "diseases"
        ]
        
        for table in tables:
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
        
        cur.execute("""
        CREATE TABLE diseases (
            disease_id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            category VARCHAR(50) 
        );
        """)
        
        cur.execute("""
        CREATE TABLE doctors (
            doctor_id SERIAL PRIMARY KEY,
            full_name VARCHAR(100) NOT NULL,
            specialization VARCHAR(100),
            region VARCHAR(100)
        );
        """)

        cur.execute("""
        CREATE TABLE patients (
            patient_id SERIAL PRIMARY KEY,
            full_name VARCHAR(100) NOT NULL,
            age INT NOT NULL,
            sex VARCHAR(20),
            address VARCHAR(255)
        );
        """)

        cur.execute("""
        CREATE TABLE health_records (
            record_id SERIAL PRIMARY KEY,
            patient_id INT REFERENCES patients(patient_id) ON DELETE CASCADE,
            blood_pressure VARCHAR(20),
            blood_sugar DECIMAL(5, 2),
            cholesterol INT,
            pulse INT,
            temperature DECIMAL(4, 1),
            weight_kg DECIMAL(5, 2),
            height_cm INT,
            bmi DECIMAL(4, 2),
            smoking_status VARCHAR(50),
            physical_activity VARCHAR(50)
        );
        """)

        cur.execute("""
        CREATE TABLE registered_cases (
            case_id SERIAL PRIMARY KEY,
            patient_id INT REFERENCES patients(patient_id) ON DELETE CASCADE,
            doctor_id INT REFERENCES doctors(doctor_id),
            disease_id INT REFERENCES diseases(disease_id),
            registration_date DATE NOT NULL,
            medical_facility VARCHAR(100),
            district VARCHAR(100) 
        );
        """)
    print("Схему БД успішно створено.")

def populate_health_records(cur, patient_ids):
    smoking = ['Ніколи', 'Так, щодня', 'Так, іноді', 'Кинув']
    activity = ['Низька', 'Середня', 'Висока']
    
    for patient_id in patient_ids:
        height_cm = random.randint(155, 195)
        weight_kg = round(random.uniform(45.0, 110.0), 2)
        bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
        
        systolic = random.randint(100, 160)
        diastolic = random.randint(60, 100)
        blood_pressure = f"{systolic}/{diastolic}"
        
        blood_sugar = round(random.uniform(3.5, 8.5), 2)
        cholesterol = random.randint(150, 300)
        pulse = random.randint(55, 95)
        temperature = round(random.uniform(36.2, 37.1), 1)

        cur.execute(
            """
            INSERT INTO health_records (
                patient_id, blood_pressure, blood_sugar, cholesterol, 
                pulse, temperature, weight_kg, height_cm, bmi, 
                smoking_status, physical_activity
            ) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                patient_id, blood_pressure, blood_sugar, cholesterol,
                pulse, temperature, weight_kg, height_cm, bmi,
                random.choice(smoking), random.choice(activity)
            )
        )

def populate_data(conn):
    print(f"Заповнення довідкових таблиць...")
    
    with conn.cursor() as cur:
        
        diseases_data = [
            ('ГРВІ', 'Респіраторні'), 
            ('Гіпертонія', 'Хронічні'), 
            ('Діабет 2 типу', 'Ендокринні'), 
            ('Астма', 'Респіраторні'), 
            ('Перелом руки', 'Травми'),
            ('Апендицит', 'Хірургічні')
        ]
        cur.executemany("INSERT INTO diseases (name, category) VALUES (%s, %s)", diseases_data)

        regions = ['Київ', 'Одеса', 'Львів', 'Харків', 'Дніпро']
        specializations = ['Терапевт', 'Кардіолог', 'Хірург', 'Педіатр', 'Невролог']
        for _ in range(NUM_DOCTORS):
            name = fake.name()
            spec = random.choice(specializations)
            reg = random.choice(regions)
            cur.execute("INSERT INTO doctors (full_name, specialization, region) VALUES (%s, %s, %s)", (name, spec, reg))
        
        for _ in range(NUM_PATIENTS):
            age = random.randint(0, 90)
            sex = random.choice(['Чоловік', 'Жінка'])
            name = fake.name_male() if sex == 'Чоловік' else fake.name_female()
            address = fake.address()
            
            cur.execute(
                "INSERT INTO patients (full_name, age, sex, address) VALUES (%s, %s, %s, %s)", 
                (name, age, sex, address)
            )
        
        patient_ids = [i for i in range(1, NUM_PATIENTS + 1)]

        populate_health_records(cur, patient_ids)

        doctor_ids = [i for i in range(1, NUM_DOCTORS + 1)]
        disease_ids = [i for i in range(1, len(diseases_data) + 1)]
        facilities = ['Міська поліклініка №1', 'Обласна лікарня', 'Приватна клініка "Здоров\'я"']
        districts = ['Дільниця №1', 'Дільниця №2', 'Дільниця №3', 'Дільниця №4']
        start_date = date.today() - timedelta(days=365)
        
        for _ in range(NUM_RECORDS):
            reg_date = start_date + timedelta(days=random.randint(0, 365))
            cur.execute(
                """
                INSERT INTO registered_cases (
                    patient_id, doctor_id, disease_id, registration_date, medical_facility, district
                ) 
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    random.choice(patient_ids),
                    random.choice(doctor_ids),
                    random.choice(disease_ids),
                    reg_date,
                    random.choice(facilities),
                    random.choice(districts)
                )
            )
    print("Дані успішно завантажено.")

def get_health_statistics(conn):
    print("\n--- ЗВЕДЕНА СТАТИСТИКА ---")
    with conn.cursor() as cur:
        
        print("\n--- СТАТИСТИКА ЗАХВОРЮВАНОСТІ ---")
        cur.execute("SELECT COUNT(*) FROM registered_cases;")
        total_cases = cur.fetchone()[0]
        print(f"\n1. Загальна кількість зареєстрованих випадків: {total_cases}")
        
        print("\n2. Розподіл випадків за регіоном лікаря:")
        cur.execute("""
            SELECT d.region, COUNT(rc.case_id) 
            FROM registered_cases rc JOIN doctors d ON rc.doctor_id = d.doctor_id
            GROUP BY d.region ORDER BY COUNT(rc.case_id) DESC;
        """)
        for region, count in cur.fetchall():
            print(f"   - {region}: {count} випадків")

        print("\n3. Топ-3 категорії захворювань:")
        cur.execute("""
            SELECT d.category, COUNT(rc.case_id) 
            FROM registered_cases rc JOIN diseases d ON rc.disease_id = d.disease_id
            GROUP BY d.category ORDER BY COUNT(rc.case_id) DESC LIMIT 3;
        """)
        for category, count in cur.fetchall():
            print(f"   - {category}: {count} випадків")
            
        print("\n4. Активність за медичними установами:")
        cur.execute("""
            SELECT medical_facility, COUNT(*) 
            FROM registered_cases GROUP BY medical_facility ORDER BY COUNT(*) DESC;
        """)
        for facility, count in cur.fetchall():
            print(f"   - {facility}: {count} випадків")
        
        print("\n5. Розподіл випадків за категоріями пацієнтів (за віком):")
        cur.execute("""
            SELECT 
                CASE
                    WHEN p.age <= 17 THEN 'Діти'
                    WHEN p.age <= 59 THEN 'Дорослі'
                    ELSE 'Пенсіонери'
                END AS patient_category,
                COUNT(rc.case_id) 
            FROM registered_cases rc JOIN patients p ON rc.patient_id = p.patient_id
            GROUP BY patient_category ORDER BY patient_category;
        """)
        for category, count in cur.fetchall():
            print(f"   - {category}: {count} випадків")


        print("\n--- СТАТИСТИКА МЕДИЧНИХ ПОКАЗНИКІВ ---")
        
        cur.execute("SELECT COUNT(*) FROM health_records;")
        print(f"\n6. Загальна кількість медичних записів: {cur.fetchone()[0]}")

        print("\n7. Середні показники за всіма записами:")
        cur.execute("""
            SELECT 
                AVG(bmi), 
                AVG(blood_sugar), 
                AVG(cholesterol) 
            FROM health_records;
        """)
        avg_stats = cur.fetchone()
        print(f"   - Середній BMI: {avg_stats[0]:.2f}")
        print(f"   - Середній цукор крові: {avg_stats[1]:.2f} ммоль/л")
        print(f"   - Середній холестерин: {avg_stats[2]:.0f} мг/дл")
        
        print("\n8. Кількість пацієнтів з високим рівнем цукру (> 7.0 ммоль/л):")
        cur.execute("SELECT COUNT(DISTINCT patient_id) FROM health_records WHERE blood_sugar > 7.0;")
        print(f"   - {cur.fetchone()[0]} пацієнтів")

def main():
    with get_db_connection() as conn:
        if conn:
            create_tables(conn)
            populate_data(conn)
            get_health_statistics(conn)
        else:
            print("Не вдалося підключитися до БД.")

if __name__ == "__main__":
    main()