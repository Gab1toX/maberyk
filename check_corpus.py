import sqlite3

conn = sqlite3.connect('episodic_memory.sqlite3')
rows = conn.execute("SELECT answer FROM conversations WHERE source='agent_generated'").fetchall()
conn.close()

sentences = [r[0] for r in rows if r[0] and r[0].strip()]
with_prefix = sum(1 for s in sentences if s.lower().startswith('curious see'))

print(f'Total: {len(sentences)}')
print(f'Empiezan con curious see: {with_prefix} ({with_prefix/len(sentences)*100:.1f}%)')
print(f'Sin ese prefijo: {len(sentences)-with_prefix}')

others = [s for s in sentences if not s.lower().startswith('curious see')]
print('Muestra sin prefijo:')
for s in others[:5]:
    print(f'  - {s}')