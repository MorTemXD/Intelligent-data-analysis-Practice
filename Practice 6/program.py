import psycopg2
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
from datetime import datetime, timedelta

DB_CONFIG = {
    "dbname": "olap_db",
    "user": "postgres",
    "password": "1111",
    "host": "localhost",
    "port": "5432"
}

NUM_RECORDS = 10000

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
        conn.commit()
    except Exception as e:
        print(f"Error: {e}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()

def create_star_schema(conn):
    print("Creating database structure...")
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS fact_orders CASCADE;")
        cur.execute("DROP TABLE IF EXISTS dim_products CASCADE;")
        cur.execute("DROP TABLE IF EXISTS dim_clients CASCADE;")

        cur.execute("""
        CREATE TABLE dim_products (
            product_id SERIAL PRIMARY KEY,
            brand VARCHAR(50),
            category VARCHAR(50)
        );
        """)

        cur.execute("""
        CREATE TABLE dim_clients (
            client_id SERIAL PRIMARY KEY,
            surname VARCHAR(100),
            city VARCHAR(50),
            region VARCHAR(50),
            country VARCHAR(50)
        );
        """)

        cur.execute("""
        CREATE TABLE fact_orders (
            order_id SERIAL PRIMARY KEY,
            execution_date DATE,
            product_id INT REFERENCES dim_products(product_id),
            client_id INT REFERENCES dim_clients(client_id),
            price DECIMAL(10, 2),
            quantity INT,
            total_sum DECIMAL(12, 2)
        );
        """)
    print("Tables created successfully.")

def populate_data(conn):
    print("Generating data...")
    
    brands_data = {
        'Smartphones': ['Samsung', 'Apple', 'Xiaomi', 'Motorola'],
        'Laptops': ['Dell', 'HP', 'Lenovo', 'Asus'],
        'Appliances': ['Bosch', 'LG', 'Philips']
    }
    
    clients_data = [
        ('Petrenko', 'Kyiv', 'Kyivska', 'Ukraine'),
        ('Ivanenko', 'Lviv', 'Lvivska', 'Ukraine'),
        ('Sydorenko', 'Odesa', 'Odeska', 'Ukraine'),
        ('Koval', 'Kharkiv', 'Kharkivska', 'Ukraine'),
        ('Bondar', 'Dnipro', 'Dnipropetrovska', 'Ukraine'),
        ('Smith', 'London', 'Greater London', 'UK'),
        ('Muller', 'Munich', 'Bavaria', 'Germany'),
        ('Nowak', 'Warsaw', 'Masovian', 'Poland')
    ]

    with conn.cursor() as cur:
        product_ids = []
        for cat, brands in brands_data.items():
            for brand in brands:
                cur.execute("INSERT INTO dim_products (brand, category) VALUES (%s, %s) RETURNING product_id", (brand, cat))
                product_ids.append(cur.fetchone()[0])
        
        client_ids = []
        for client in clients_data:
            cur.execute("INSERT INTO dim_clients (surname, city, region, country) VALUES (%s, %s, %s, %s) RETURNING client_id", client)
            client_ids.append(cur.fetchone()[0])

        start_date = datetime(2025, 1, 1)
        for _ in range(NUM_RECORDS):
            date_exec = start_date + timedelta(days=random.randint(0, 365))
            pid = random.choice(product_ids)
            cid = random.choice(client_ids)
            
            qty = random.randint(1, 10)
            price = random.uniform(500, 40000)
            total = price * qty
            
            cur.execute("""
                INSERT INTO fact_orders (execution_date, product_id, client_id, price, quantity, total_sum)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (date_exec, pid, cid, round(price, 2), qty, round(total, 2)))
            
    print(f"Generated {NUM_RECORDS} records.")

def analyze_olap(conn):
    print("\nRunning OLAP analysis...")
    
    query = """
    SELECT 
        f.execution_date,
        p.category,
        p.brand,
        c.surname,
        c.region,
        c.country,
        f.price,
        f.quantity,
        f.total_sum
    FROM fact_orders f
    JOIN dim_products p ON f.product_id = p.product_id
    JOIN dim_clients c ON f.client_id = c.client_id
    """
    df = pd.read_sql_query(query, conn)
    
    pivot1 = pd.pivot_table(df, values='total_sum', index='region', columns='category', aggfunc='sum', fill_value=0)
    print("\nЗведена таблиця 1 (Продажі за регіонами та категоріями):")
    print(pivot1)
    
    pivot2 = pd.pivot_table(df, values='quantity', index='brand', columns='country', aggfunc='sum', fill_value=0)
    print("\nЗведена таблиця 2 (Кількість за брендом та країною):")
    print(pivot2)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot2, annot=True, fmt='d', cmap='Blues')
    plt.title('Теплова карта: Продажі Бренд vs Країна')
    plt.ylabel('Бренд')
    plt.xlabel('Країна')
    plt.tight_layout()
    plt.show()

    df['execution_date'] = pd.to_datetime(df['execution_date'])
    daily_sales = df.groupby('execution_date')['total_sum'].sum().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='execution_date', y='total_sum', data=daily_sales, color='green')
    plt.title('Динаміка продажів у часі')
    plt.xlabel('Дата виконання')
    plt.ylabel('Загальна сума продажів')
    plt.tight_layout()
    plt.show()

def main():
    with get_db_connection() as conn:
        if conn:
            create_star_schema(conn)
            populate_data(conn)
            analyze_olap(conn)

if __name__ == "__main__":
    main()