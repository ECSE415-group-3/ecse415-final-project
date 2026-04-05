import json
from pathlib import Path

p = Path(__file__).resolve().parent / "notebooks" / "01-feature-based-model-optionA.ipynb"
raw = p.read_text(encoding="utf-8")
try:
    data = json.loads(raw)
except json.JSONDecodeError as e:
    Path("notebooks/_json_decode_error.txt").write_text(
        f"{e}\nlineno={e.lineno} colno={e.colno} pos={e.pos}\n", encoding="utf-8"
    )
    raise
p.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
Path("notebooks/_json_ok.txt").write_text("ok\n", encoding="utf-8")
