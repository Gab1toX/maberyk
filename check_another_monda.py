python - << 'EOF'
import io
p = 'interfaz/web_mind.py'
s = io.open(p, encoding='utf-8').read()
old = "        self.agent.enable_voice()\n"
new = "        self.agent.enable_voice()\n        self.agent.enable_tutor()\n"
assert s.count(old) == 1, f"encontradas {s.count(old)} coincidencias"
io.open(p, 'w', encoding='utf-8').write(s.replace(old, new))
print("listo")
EOF