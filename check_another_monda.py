import sqlite3
conn = sqlite3.connect("episodic_memory.sqlite3")
total = conn.execute("SELECT COUNT(*) FROM conversations WHERE source='human_taught'").fetchone()[0]
print(f"human_taught: {total}")
total2 = conn.execute("SELECT COUNT(*) FROM conversations WHERE source='agent_generated'").fetchone()[0]
print(f"agent_generated: {total2}")