import sqlite3

conn = sqlite3.connect('episodic_memory.sqlite3')
schema = conn.execute("SELECT sql FROM sqlite_master WHERE name='conversations'").fetchone()
print(f'Schema actual: {schema}')