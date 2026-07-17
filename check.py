import sqlite3
conn = sqlite3.connect("episodic_memory.sqlite3")
rows = conn.execute("SELECT question, answer FROM conversations WHERE keywords LIKE '%claude%'").fetchall()
for q, a in rows:
    print(q, "->", a)