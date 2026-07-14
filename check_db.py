import sqlite3
conn = sqlite3.connect('episodic_memory.sqlite3')
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
for t in tables:
    print(f'Tabla: {t[0]}')
    cols = conn.execute(f"PRAGMA table_info({t[0]})").fetchall()
    for c in cols:
        print(f'  {c[1]} ({c[2]})')
conn.close()