"""Decide whether a verified version is new, without hiding partial uploads."""

import json
import os
from pathlib import Path
import tomllib
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


def needs_publishing(name, version, filenames):
    """Return False only for an existing, complete, non-yanked release.

    PyPI files cannot be replaced. Ordinary main updates keep their version and
    skip publishing; incomplete releases require an explicit maintainer decision.
    Network failures and unexpected API responses must fail the workflow.
    """
    url = f"https://pypi.org/pypi/{quote(name, safe='')}/{quote(version, safe='')}/json"
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "neuralign-release-check"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            release = json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return True
        raise

    available = {
        item["filename"] for item in release["urls"] if not item.get("yanked", False)
    }
    missing = set(filenames) - available
    if missing:
        raise RuntimeError(
            f"PyPI already contains {name} {version}, but its release is incomplete "
            f"or yanked (missing: {', '.join(sorted(missing))}). Recover the release "
            "deliberately or bump the version; automatic publishing will not skip "
            "individual files."
        )
    return False


def main():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    name, version = project["name"], project["version"]
    wheels = list(Path("dist").glob("*.whl"))
    sources = list(Path("dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise RuntimeError("Expected exactly one verified wheel and source distribution")

    publish = needs_publishing(name, version, [path.name for path in wheels + sources])
    outputs = f"version={version}\npublish={str(publish).lower()}\n"
    if output_file := os.environ.get("GITHUB_OUTPUT"):
        with open(output_file, "a", encoding="utf-8") as stream:
            stream.write(outputs)
    status = "will be published" if publish else "is already published; skipping upload"
    message = f"{name} {version} {status}."
    print(message)
    if summary_file := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_file, "a", encoding="utf-8") as stream:
            stream.write(message + "\n")


if __name__ == "__main__":
    main()
