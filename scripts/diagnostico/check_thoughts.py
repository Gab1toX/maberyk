import sqlite3
conn = sqlite3.connect('episodic_memory.sqlite3')
rows = conn.execute("""
    SELECT thought, COUNT(*) as cnt 
    FROM episodes 
    WHERE thought != '' AND thought IS NOT NULL
    GROUP BY thought 
    ORDER BY cnt DESC 
    LIMIT 20
""").fetchall()
for r in rows:
    print(f"{r[1]:6d}x  {r[0]}")
conn.close()