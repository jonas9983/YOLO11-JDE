import sqlite3
import argparse
import os

def check_distribution(db_path):
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("--- Athlete Distribution per Division ---")
    query_div = """
    SELECT division, COUNT(DISTINCT athlete_name) as num_athletes
    FROM athletes
    GROUP BY division
    ORDER BY num_athletes DESC;
    """
    cursor.execute(query_div)
    for row in cursor.fetchall():
        print(f"{row[0]:<30} | {row[1]:>5} athletes")

    print("\n--- Top 20 Contests by Athlete Count ---")
    query_contests = """
    SELECT year, contest_name, COUNT(DISTINCT athlete_name) as num_athletes
    FROM athletes
    GROUP BY year, contest_name
    ORDER BY num_athletes DESC
    LIMIT 20;
    """
    cursor.execute(query_contests)
    for row in cursor.fetchall():
        print(f"{row[0]} {row[1]:<50} | {row[2]:>5} athletes")

    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="./debug/npc_database.db", help="Path to npc_database.db")
    args = parser.parse_args()
    check_distribution(args.db)
