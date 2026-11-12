"""Build the docs; the homepage follows README.md. Run: python docs/build.py."""

from html import escape
from pathlib import Path
import shutil
from urllib.parse import urlsplit, urlunsplit
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent
PAGES = ("index", "getting-started", "model-and-outputs", "api", "reproducibility")
NAV = {
    "index": "Neuralign",
    "getting-started": "Your recordings",
    "model-and-outputs": "Outputs",
    "api": "API",
    "reproducibility": "Save a fit",
}
SITE = "https://mikloskralik.github.io/neuralign/"
ASSETS = "https://raw.githubusercontent.com/MiklosKralik/neuralign/main/docs/assets/"
home = (ROOT.parent / "README.md").read_text()
home = home.replace(ASSETS, "assets/")
home = home.replace(f"[Documentation]({SITE})", "[Model and outputs](model-and-outputs.md)")
for page in PAGES:
    home = home.replace(f"{SITE}{page}.html", f"{page}.md")
(ROOT / "index.md").write_text(home)

renderer = MarkdownIt("default", {"html": True})
for page in PAGES:
    source = (ROOT / f"{page}.md").read_text()
    navigation = " ".join(
        f'<a href="{slug}.html" {"aria-current=page" if slug == page else ""}>{label}</a>'
        for slug, label in NAV.items()
    )
    tokens = renderer.parse(source)
    for token in tokens:
        for child in token.children or []:
            if child.type == "link_open":
                target = urlsplit(child.attrGet("href"))
                if not target.scheme and target.path.endswith(".md"):
                    child.attrSet("href", urlunsplit(target._replace(path=target.path[:-3] + ".html")))
    content = renderer.renderer.render(tokens, renderer.options, {})
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(NAV[page])} · Neuralign</title><link rel="icon" href="assets/neuralign-mark.svg" type="image/svg+xml"><link rel="stylesheet" href="style.css"></head>
<body><nav aria-label="Main navigation">{navigation}</nav><main>{content}</main></body></html>"""
    (ROOT / f"{page}.html").write_text(html)
shutil.copy2(ROOT.parent / "examples/quickstart.py", ROOT / "assets/quickstart.py")
image = ROOT.parent / "examples/quickstart.png"
if image.exists():
    shutil.copy2(image, ROOT / "assets/quickstart.png")
print(f"Built {len(PAGES)} documentation pages in {ROOT}")
