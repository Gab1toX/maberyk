import sqlite3
c = sqlite3.connect('episodic_memory.sqlite3')
print('pares unicos:', c.execute("SELECT COUNT(*) FROM (SELECT DISTINCT question, answer FROM conversations WHERE source='human_taught')").fetchone())
print('thoughts unicos:', c.execute("SELECT COUNT(DISTINCT thought) FROM episodes WHERE thought != ''").fetchone())
print('seed presente:', c.execute("SELECT COUNT(*) FROM conversations WHERE question='quien eres' AND source='human_taught'").fetchone())

print('pares unicos por source:')
for row in c.execute(
    """
    SELECT source, COUNT(*) FROM (
        SELECT DISTINCT source, question, answer FROM conversations
    )
    GROUP BY source
    ORDER BY source
    """
):
    print(f"  {row[0]}: {row[1]}")
