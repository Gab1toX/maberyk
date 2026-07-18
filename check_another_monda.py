import sqlite3
conn = sqlite3.connect("episodic_memory.sqlite3")
rows = conn.execute("SELECT question, answer FROM conversations WHERE source='human_taught' AND (question LIKE '%nombre%' OR question LIKE '%maberyk%' OR question LIKE '%llamas%') LIMIT 10").fetchall()
for q, a in rows:
    print(q, "->", a)