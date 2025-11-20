import psycopg2
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

DB_CONFIG = {
    "dbname": "medical_analysis_db",
    "user": "postgres",
    "password": "1111",
    "host": "localhost",
    "port": "5432"
}

NUM_PATIENTS = 10000

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
        conn.commit()
    except Exception as e:
        print(f"Помилка підключення до БД: {e}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()

def create_tables(conn):
    print("Створення таблиць бази даних...")
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS dataset CASCADE;")
        
        cur.execute("""
        CREATE TABLE dataset (
            patient_id SERIAL PRIMARY KEY,
            age INT,
            sex VARCHAR(20),
            bmi DECIMAL(4, 2),
            blood_glucose DECIMAL(5,2),
            systolic_bp INT,
            cholesterol INT,
            smoking_status VARCHAR(20),
            target_disease VARCHAR(50)
        );
        """)
    print("Таблиці успішно створено.")

def populate_smart_data(conn):
    print("Генерація синтетичних медичних даних...")
    
    with conn.cursor() as cur:
        for _ in range(NUM_PATIENTS):
            age = random.randint(18, 90)
            sex = random.choice(['Чоловік', 'Жінка'])
            
            bmi = round(random.normalvariate(26, 5), 2)
            glucose = round(random.normalvariate(5.5, 1.5), 2)
            cholesterol = int(random.normalvariate(200, 40))
            systolic_bp = int(random.normalvariate(120, 15))
            smoking = random.choice(['Ні', 'Так'])
            
            disease = 'Здоровий'
            risk_score = 0
            
            if glucose > 7.0: risk_score += 3
            if bmi > 30: risk_score += 2
            if age > 50: risk_score += 1
            if systolic_bp > 140: risk_score += 2
            
            if glucose > 7.5 and bmi > 28:
                if random.random() < 0.90:
                    disease = 'Діабет 2 типу'
            elif (systolic_bp > 145 or (systolic_bp > 135 and age > 60)):
                if random.random() < 0.85:
                    disease = 'Гіпертонія'
            else:
                risk_score = 0
                if glucose > 6.5: risk_score += 2
                if bmi > 30: risk_score += 2
                if age > 50: risk_score += 1
                if systolic_bp > 135: risk_score += 2
                if cholesterol > 240: risk_score += 2
                if smoking == 'Так': risk_score += 2
                
                if risk_score >= 5:
                    if random.random() < 0.80:
                        disease = 'Хвороба серця'

            if disease == 'Здоровий' and random.random() < 0.05:
                disease = random.choice(['Діабет 2 типу', 'Гіпертонія', 'Хвороба серця'])

            if disease != 'Здоровий' and random.random() < 0.05:
                disease = 'Здоровий'

            cur.execute("""
                INSERT INTO dataset (age, sex, bmi, blood_glucose, systolic_bp, cholesterol, smoking_status, target_disease)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (age, sex, bmi, glucose, systolic_bp, cholesterol, smoking, disease))
            
    print(f"Успішно завантажено {NUM_PATIENTS} записів.")

def perform_prediction_analysis(conn):
    print("\n--- ЕТАП DATA MINING: Прогнозування та Аналітика ---")
    
    df = pd.read_sql_query("SELECT * FROM dataset", conn)
    
    le = LabelEncoder()
    df['sex_encoded'] = le.fit_transform(df['sex'])
    df['smoking_encoded'] = le.fit_transform(df['smoking_status'])
    
    feature_names = ['age', 'bmi', 'blood_glucose', 'systolic_bp', 'cholesterol', 'sex_encoded', 'smoking_encoded']
    ukr_feature_names = ['Вік', 'ІМТ', 'Глюкоза', 'Тиск', 'Холестерин', 'Стать', 'Куріння']
    
    X = df[feature_names]
    y = df['target_disease']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Навчання моделі Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("\n=== ЗВІТ ТОЧНОСТІ ПРОГНОЗУ ===")
    print(classification_report(y_test, y_pred))
    
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.countplot(x='target_disease', data=df, palette='pastel', edgecolor='black')
    plt.title('Розподіл діагнозів у вибірці', fontsize=14)
    plt.xlabel('Діагноз', fontsize=12)
    plt.ylabel('Кількість пацієнтів', fontsize=12)
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title('Матриця помилок', fontsize=14)
    plt.xlabel('Прогноз моделі', fontsize=12)
    plt.ylabel('Реальний діагноз', fontsize=12)
    plt.show()

    feature_imp = pd.Series(clf.feature_importances_, index=ukr_feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_imp, y=feature_imp.index, palette='viridis')
    plt.title('Топ факторів впливу на діагноз', fontsize=14)
    plt.xlabel('Коефіцієнт важливості', fontsize=12)
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='target_disease', y='blood_glucose', data=df, palette='Set2')
    plt.title('Рівень глюкози залежно від діагнозу', fontsize=14)
    plt.xlabel('Діагноз')
    plt.ylabel('Глюкоза (ммоль/л)')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='systolic_bp', hue='target_disease', data=df, palette='deep', alpha=0.7)
    plt.title('Кластеризація: Вік та Тиск', fontsize=14)
    plt.xlabel('Вік пацієнта')
    plt.ylabel('Систолічний тиск')
    plt.legend(title='Діагноз', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    df_corr = df[feature_names].copy()
    df_corr.columns = ukr_feature_names
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Кореляція між показниками', fontsize=14)
    plt.show()

def main():
    with get_db_connection() as conn:
        if conn:
            create_tables(conn)
            populate_smart_data(conn)
            perform_prediction_analysis(conn)
        else:
            print("Не вдалося підключитися до БД.")

if __name__ == "__main__":
    main()