import sqlite3
import os

path = os.path.abspath('episodic_memory.sqlite3')
print(f'Archivo: {path}')

conn = sqlite3.connect('episodic_memory.sqlite3')
total = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
print(f'Total conversations: {total}')

sample = conn.execute("SELECT question, answer FROM conversations ORDER BY timestamp DESC LIMIT 10").fetchall()
for q, a in sample:
    print(f'Q: {q}')
    print(f'A: {a}')
    print('---')