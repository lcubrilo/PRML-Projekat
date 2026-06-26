"""Build report.pdf from report.md.

Preprocesses a clean copy (strips working status tags, author/meta comments, and emoji
that xelatex can't render), forces a page break before each major section, then renders
with pandoc + wkhtmltopdf. Leaves report.md untouched so the team keeps its tracking tags.

Usage:  python report/build_report.py
"""
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "report.md"
BUILD = HERE / "_build.md"
OUT = HERE / "report.pdf"


def strip_emoji(s: str) -> str:
    def keep(ch):
        o = ord(ch)
        return not (o == 0xFE0F or 0x2600 <= o <= 0x27BF or 0x2B00 <= o <= 0x2BFF
                    or 0x23E9 <= o <= 0x23FA or 0x1F000 <= o <= 0x1FAFF)
    return "".join(c for c in s if keep(c))


def clean(md: str) -> str:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)        # drop HTML comments
    out = []
    first_section = True
    for line in md.splitlines():
        if line.startswith("> Authoring split"):                # drop the author blockquote
            continue
        if re.match(r"^#{1,4} ", line):                          # heading: cut working " - tag ..." suffix
            line = line.split(" - ", 1)[0].rstrip()
        if re.match(r"^## \d", line):                           # page break before each major section
            if not first_section:
                out.append('<div style="page-break-before: always;"></div>\n')
            first_section = False
        out.append(line)
    return strip_emoji("\n".join(out))


def main():
    BUILD.write_text(clean(SRC.read_text(encoding="utf-8")), encoding="utf-8")
    # run pandoc from report/ so relative figures/ paths resolve for wkhtmltopdf
    cmd = ["pandoc", BUILD.name, "-o", OUT.name, "--pdf-engine=wkhtmltopdf",
           "-V", "margin-top=12", "-V", "margin-bottom=14",
           "-V", "margin-left=14", "-V", "margin-right=14"]
    subprocess.run(cmd, check=True, cwd=HERE)
    BUILD.unlink(missing_ok=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
