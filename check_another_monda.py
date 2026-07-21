import sqlite3
conn = sqlite3.connect("episodic_memory.sqlite3")
rows = conn.execute("SELECT question, answer, source FROM conversations WHERE answer LIKE '%valorant%' ORDER BY timestamp DESC LIMIT 5").fetchall()
for q, a, s in rows:
    print(s, "|", q, "->", a)