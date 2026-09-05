# check.py
import sqlite3
conn = sqlite3.connect("episodic_memory.sqlite3")
rows = conn.execute("SELECT question, answer, keywords FROM conversations ORDER BY timestamp DESC LIMIT 20").fetchall()
for q, a, k in rows:
    print(f"Q: {q}")
    print(f"A: {a}")
    print(f"K: {k}")
    print("---")