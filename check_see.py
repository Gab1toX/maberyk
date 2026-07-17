# check_see.py
import sqlite3
conn = sqlite3.connect("episodic_memory.sqlite3")
rows = conn.execute("SELECT COUNT(*) FROM conversations WHERE answer LIKE '% see %' OR answer LIKE 'see %' OR answer LIKE '% see'").fetchone()
print(f"Respuestas con 'see': {rows[0]}")