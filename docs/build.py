"""Build the small static docs site: python docs/build.py (install the docs extra)."""

from html import escape
from pathlib import Path
import shutil
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent
PAGES = ("index", "getting-started", "model-and-outputs", "api", "reproducibility")
NAV = {
    "index": "Neuralign",
    "getting-started": "Start here",
    "model-and-outputs": "Outputs",
    "api": "API",
    "reproducibility": "Reproducibility",
}
renderer = MarkdownIt("default")
for page in PAGES:
    source = (ROOT / f"{page}.md").read_text()
    source = source.replace(
        "{{example}}",
        "```python\n" + (ROOT.parent / "examples/quickstart.py").read_text() + "\n```",
    )
    navigation = " ".join(
        f'<a href="{slug}.html" {"aria-current=page" if slug == page else ""}>{label}</a>'
        for slug, label in NAV.items()
    )
    content = renderer.render(source)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(NAV[page])} · Neuralign</title><link rel="stylesheet" href="style.css"></head>
<body><nav aria-label="Main navigation">{navigation}</nav><main>{content}</main>
<footer>Neuralign · research software · version 0.1.0</footer></body></html>"""
    (ROOT / f"{page}.html").write_text(html)
shutil.copy2(ROOT.parent / "examples/quickstart.py", ROOT / "assets/quickstart.py")
image = ROOT.parent / "examples/quickstart.png"
if image.exists():
    shutil.copy2(image, ROOT / "assets/quickstart.png")
print(f"Built {len(PAGES)} documentation pages in {ROOT}")
