# fix_see.py
import sqlite3
conn = sqlite3.connect("episodic_memory.sqlite3")
row = conn.execute("SELECT id, answer FROM conversations WHERE answer LIKE '% see %' OR answer LIKE 'see %' OR answer LIKE '% see'").fetchone()
print(f"ID: {row[0]}, Answer: {row[1]}")
clean = " ".join(w for w in row[1].split() if w.lower() != "see")
conn.execute("UPDATE conversations SET answer = ? WHERE id = ?", (clean, row[0]))
conn.commit()
print(f"Corregida: {clean}")