# clean_see.py
import sqlite3
conn = sqlite3.connect("episodic_memory.sqlite3")
rows = conn.execute("SELECT id, answer FROM conversations WHERE answer LIKE '%see%'").fetchall()
print(f"Encontradas {len(rows)} respuestas con 'see'")
for id, answer in rows:
    clean = " ".join(w for w in answer.split() if w.lower() != "see")
    conn.execute("UPDATE conversations SET answer = ? WHERE id = ?", (clean, id))
conn.commit()
print("Listo")