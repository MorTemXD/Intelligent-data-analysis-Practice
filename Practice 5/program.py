import psycopg2
import random
from faker import Faker
from contextlib import contextmanager

DB_CONFIG = {
    "dbname": "health_records_db",
    "user": "postgres",
    "password": "1111",
    "host": "localhost",
    "port": "5432"
}

fake = Faker('uk_UA')
NUM_RECORDS = 1000

@contextmanager
def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
        conn.commit()
    except Exception as e:
        print(f"Помилка підключення до бази даних: {e}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def create_health_table(conn):
    print("Створення таблиці 'health_records'...")
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS health_records (
            id SERIAL PRIMARY KEY,
            name VARCHAR(150),
            age INT,
            sex VARCHAR(20),
            blood_pressure VARCHAR(20),
            blood_sugar DECIMAL(5, 2),
            cholesterol INT,
            pulse INT,
            temperature DECIMAL(4, 1),
            weight_kg DECIMAL(5, 2),
            height_cm INT,
            bmi DECIMAL(4, 2),
            smoking_status VARCHAR(50),
            physical_activity VARCHAR(50),
            chronic_diseases VARCHAR(255)
        );
        """)
        cur.execute("TRUNCATE TABLE health_records RESTART IDENTITY;")
    print("Таблицю успішно створено (або очищено).")

def populate_health_data(conn):
    print(f"Заповнення таблиці {NUM_RECORDS} записами...")
    with conn.cursor() as cur:
        sexes = ['Чоловік', 'Жінка']
        smoking = ['Ніколи', 'Так, щодня', 'Так, іноді', 'Кинув']
        activity = ['Низька', 'Середня', 'Висока']
        diseases = ['Немає', 'Гіпертонія', 'Діабет 2 типу', 'Астма', 'Гіпертонія, Діабет']

        for _ in range(NUM_RECORDS):
            sex = random.choice(sexes)
            if sex == 'Чоловік':
                name = fake.name_male()
                height_cm = random.randint(165, 195)
                weight_kg = round(random.uniform(60.0, 110.0), 2)
            else:
                name = fake.name_female()
                height_cm = random.randint(155, 180)
                weight_kg = round(random.uniform(45.0, 90.0), 2)
            
            age = random.randint(18, 80)
            
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
                    name, age, sex, blood_pressure, blood_sugar, cholesterol, 
                    pulse, temperature, weight_kg, height_cm, bmi, 
                    smoking_status, physical_activity, chronic_diseases
                ) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    name, age, sex, blood_pressure, blood_sugar, cholesterol,
                    pulse, temperature, weight_kg, height_cm, bmi,
                    random.choice(smoking), random.choice(activity), random.choice(diseases)
                )
            )
    print("Дані успішно завантажено.")

def get_health_statistics(conn):
    print("\n--- МЕДИЧНА СТАТИСТИКА ---")
    with conn.cursor() as cur:
        
        cur.execute("SELECT COUNT(*) FROM health_records;")
        print(f"\n1. Загальна кількість пацієнтів: {cur.fetchone()[0]}")

        print("\n2. Розподіл за статтю:")
        cur.execute("SELECT sex, COUNT(*) FROM health_records GROUP BY sex;")
        for row in cur.fetchall():
            print(f"   - {row[0]}: {row[1]} осіб")

        print("\n3. Середні показники:")
        cur.execute("""
            SELECT 
                AVG(age), 
                AVG(bmi), 
                AVG(blood_sugar), 
                AVG(cholesterol) 
            FROM health_records;
        """)
        avg_stats = cur.fetchone()
        print(f"   - Середній вік: {avg_stats[0]:.1f} років")
        print(f"   - Середній BMI: {avg_stats[1]:.2f}")
        print(f"   - Середній цукор крові: {avg_stats[2]:.2f} ммоль/л")
        print(f"   - Середній холестерин: {avg_stats[3]:.0f} мг/дл")

        print("\n4. Статус куріння:")
        cur.execute("SELECT smoking_status, COUNT(*) FROM health_records GROUP BY smoking_status ORDER BY COUNT(*) DESC;")
        for row in cur.fetchall():
            print(f"   - {row[0]:<15}: {row[1]} осіб")
            
        cur.execute("SELECT COUNT(*) FROM health_records WHERE blood_sugar > 7.0;")
        print(f"\n5. Кількість пацієнтів з цукром крові > 7.0 ммоль/л: {cur.fetchone()[0]}")

def main():
    with get_db_connection() as conn:
        if conn:
            create_health_table(conn)
            populate_health_data(conn)
            get_health_statistics(conn)
        else:
            print("Не вдалося підключитися до БД. Скрипт зупинено.")

if __name__ == "__main__":
    main()