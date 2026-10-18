"""lessons/theory/*.md → 정적 웹페이지(docs/)로 변환한다.

사용법:  python tools/build_theory.py
출력:
  docs/index.html                  학습 로드맵(시작 페이지)
  docs/theory/stage_00.html ...    단계별 이론 설명 페이지
  docs/theory/glossary.html        검색되는 용어 사전
  docs/theory/all.html             전체를 한 파일로 합친 판 (공유·미리보기용, 외부 파일 불필요)
수식은 MathJax(CDN)로 표시한다. 인터넷이 없으면 수식이 원문($...$) 그대로 보인다.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify_unicode

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "lessons" / "theory"
OUT = ROOT / "docs"
THEORY_OUT = OUT / "theory"

PHASES = [
    ("기초", [0, 1, 2, 3, 4], "Python·문제 정의·데이터 품질·통계·변수 진단"),
    ("로지스틱 CSS", [5, 6, 7], "비닝/WOE/IV → 로지스틱 회귀 → 스코어카드"),
    ("검증과 모니터링", [8, 9, 10], "분할·누수 방지 → 평가지표 → PSI·드리프트"),
    ("머신러닝", [11, 12, 13], "트리 부스팅 → 튜닝·보정 → 합성 데이터(GAN)"),
    ("통계 심화·운영·캡스톤", [14, 15, 16], "검정의 올바른 사용 → 운영·위험관리 → 종합 프로젝트"),
]

CSS = """
:root{--bg:#fbfaf7;--fg:#1f2328;--muted:#5d6670;--accent:#2b5c8a;--accent-soft:#e7eef6;--card:#ffffff;--border:#dcd8cf;
--code-bg:#f1efe9;--warn:#9a5b00;--warn-bg:#fff4df;--danger:#9b2c2c;--danger-bg:#fdeaea;--tip:#1d6b4f;--tip-bg:#e6f4ee;--note-bg:#e7eef6}
@media (prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e6e3dc;--muted:#a0a7b0;--accent:#8ab4e0;--accent-soft:#222c38;--card:#1d2026;--border:#33373f;
--code-bg:#242831;--warn:#f0b35a;--warn-bg:#3a2e17;--danger:#f19a9a;--danger-bg:#3a2020;--tip:#7fd1ae;--tip-bg:#1b3329;--note-bg:#222c38}}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI","Apple SD Gothic Neo","Noto Sans KR","Malgun Gothic",sans-serif}
a{color:var(--accent)}
.layout{display:grid;grid-template-columns:270px minmax(0,1fr);min-height:100vh}
nav.side{position:sticky;top:0;align-self:start;height:100vh;overflow:auto;padding:20px 16px;border-right:1px solid var(--border);background:var(--card);font-size:14px}
nav.side .brand{font-weight:700;font-size:15px;display:block;margin-bottom:14px;text-decoration:none;color:var(--fg)}
nav.side h4{margin:16px 0 6px;font-size:12px;color:var(--muted);letter-spacing:.04em}
nav.side a.item{display:block;padding:5px 8px;border-radius:6px;text-decoration:none;color:var(--fg);line-height:1.4}
nav.side a.item:hover{background:var(--accent-soft)}
nav.side a.item.current{background:var(--accent-soft);color:var(--accent);font-weight:600}
main{padding:32px 28px 80px;max-width:880px;width:100%;margin:0 auto}
main h1{font-size:30px;line-height:1.3;margin:.2em 0 .4em}
main h2{font-size:23px;margin:2.1em 0 .6em;padding-top:.6em;border-top:1px solid var(--border)}
main h3{font-size:18px;margin:1.6em 0 .4em}
blockquote{margin:1em 0;padding:.6em 1em;border-left:4px solid var(--accent);background:var(--accent-soft);border-radius:0 8px 8px 0}
blockquote p{margin:.3em 0}
code{font:13.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--code-bg);padding:.12em .35em;border-radius:4px}
pre{background:var(--code-bg);border:1px solid var(--border);border-radius:8px;padding:12px 14px;overflow:auto}
pre code{background:none;padding:0}
.table-wrap{overflow-x:auto;margin:1em 0}
table{border-collapse:collapse;font-size:14.5px;min-width:60%}
th,td{border:1px solid var(--border);padding:6px 10px;text-align:left;vertical-align:top}
th{background:var(--accent-soft)}
.admonition{margin:1.2em 0;padding:.7em 1em;border-radius:8px;border:1px solid var(--border);background:var(--note-bg)}
.admonition-title{font-weight:700;margin:0 0 .3em}
.admonition p{margin:.3em 0}
.admonition.warning{background:var(--warn-bg);border-color:var(--warn)}
.admonition.warning .admonition-title{color:var(--warn)}
.admonition.danger{background:var(--danger-bg);border-color:var(--danger)}
.admonition.danger .admonition-title{color:var(--danger)}
.admonition.tip{background:var(--tip-bg);border-color:var(--tip)}
.admonition.tip .admonition-title{color:var(--tip)}
details{margin:.8em 0;padding:.5em .9em;border:1px solid var(--border);border-radius:8px;background:var(--card)}
details summary{cursor:pointer;font-weight:600}
.pager{display:flex;justify-content:space-between;gap:12px;margin-top:48px;padding-top:18px;border-top:1px solid var(--border);font-size:15px}
.footer{margin-top:28px;color:var(--muted);font-size:13px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px;margin:12px 0 28px}
.card{display:block;padding:12px 14px;border:1px solid var(--border);border-radius:10px;background:var(--card);text-decoration:none;color:var(--fg)}
.card:hover{border-color:var(--accent)}
.card b{display:block;margin-bottom:3px}
.card span{color:var(--muted);font-size:14px;line-height:1.5}
.search{width:100%;padding:10px 12px;border:1px solid var(--border);border-radius:8px;background:var(--card);color:var(--fg);font-size:16px}
.mjx-chtml{outline:0}
@media (max-width:900px){.layout{grid-template-columns:1fr}nav.side{position:static;height:auto;border-right:0;border-bottom:1px solid var(--border)}main{padding:22px 16px 60px}}
@media print{nav.side,.pager{display:none}.layout{display:block}body{background:#fff;color:#000}}
"""

MATHJAX = """<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]},options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>"""

SEARCH_JS = """<script>
(function(){var box=document.getElementById('gl-search');if(!box)return;
box.addEventListener('input',function(){var q=box.value.trim().toLowerCase();
document.querySelectorAll('main table tbody tr').forEach(function(tr){tr.style.display=(!q||tr.textContent.toLowerCase().indexOf(q)>-1)?'':'none';});});})();
</script>"""


# ---------------------------------------------------------------- 변환
def _protect_math(text: str):
    """코드는 건드리지 않고 $...$, $$...$$ 만 토큰으로 바꿔 Markdown 처리에서 보호한다."""
    code_store: list[str] = []
    math_store: list[tuple[bool, str]] = []

    def mask_code(m):
        code_store.append(m.group(0))
        return f"CODETOKEN{len(code_store) - 1}END"

    def put(m, display):
        math_store.append((display, m.group(1).strip()))
        return f"MATHTOKEN{len(math_store) - 1}END"

    text = re.sub(r"```.*?```", mask_code, text, flags=re.S)
    text = re.sub(r"`[^`\n]+`", mask_code, text)
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: put(m, True), text, flags=re.S)
    text = re.sub(r"(?<![\\$\w])\$(?!\s)([^$\n]+?)(?<!\s)\$(?!\w)", lambda m: put(m, False), text)
    for i, c in enumerate(code_store):
        text = text.replace(f"CODETOKEN{i}END", c)
    return text, math_store


def _restore_math(body: str, math_store) -> str:
    for i, (display, content) in enumerate(math_store):
        esc = html.escape(content, quote=False)
        rendered = f"\\[{esc}\\]" if display else f"\\({esc}\\)"
        body = body.replace(f"MATHTOKEN{i}END", rendered)
    return body


def convert(md_text: str, id_prefix: str = "") -> tuple[str, list[dict]]:
    protected, math_store = _protect_math(md_text)
    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "admonition", "md_in_html", "sane_lists"],
        extension_configs={"toc": {"slugify": slugify_unicode, "toc_depth": "2-3"}},
    )
    body = md.convert(protected)
    toc = md.toc_tokens
    body = _restore_math(body, math_store)
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    if id_prefix:
        body = re.sub(r'(<h[1-6] id=")', lambda m: m.group(1) + id_prefix, body)
    return body, toc


def parse_meta(md_text: str) -> dict:
    title = re.search(r"^# (.+)$", md_text, re.M)
    summary = re.search(r"^> (.+)$", md_text, re.M)
    return {
        "title": title.group(1).strip() if title else "(제목 없음)",
        "summary": summary.group(1).strip() if summary else "",
    }


# ---------------------------------------------------------------- 페이지 조립
def sidebar(pages: list[dict], current: str | None, base: str, home: str) -> str:
    parts = [f'<a class="brand" href="{home}">📘 Credit Model Mastery Lab</a>']
    by_num = {p["num"]: p for p in pages if p["kind"] == "stage"}
    for phase_name, nums, _ in PHASES:
        parts.append(f"<h4>{html.escape(phase_name)}</h4>")
        for n in nums:
            p = by_num.get(n)
            if not p:
                continue
            cls = "item current" if p["slug"] == current else "item"
            parts.append(f'<a class="{cls}" href="{base}{p["slug"]}.html">{n}. {html.escape(p["short"])}</a>')
    parts.append("<h4>도구</h4>")
    cls = "item current" if current == "glossary" else "item"
    parts.append(f'<a class="{cls}" href="{base}glossary.html">용어 사전 (검색)</a>')
    parts.append(f'<a class="item" href="{base}all.html">전체를 한 페이지로</a>')
    return "<nav class=\"side\">" + "\n".join(parts) + "</nav>"


def wrap_page(title: str, side: str, content: str, pager: str = "", extra_js: str = "") -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · Credit Model Mastery Lab</title>
<style>{CSS}</style>
{MATHJAX}
</head>
<body>
<div class="layout">
{side}
<main>
{content}
{pager}
<p class="footer">교육용 자료입니다. 실제 은행의 내부등급법(IRB)·감독규정 적합성이나 규제 승인을 보증하지 않습니다.
실제 적용 전에는 해당 시점의 공식 규정·감독지침·내부 방법론 원문을 확인하세요. 참고자료 확인일: 2026-10-18.</p>
</main>
</div>
{extra_js}
</body>
</html>
"""


def load_pages() -> list[dict]:
    pages = []
    for path in sorted(SRC.glob("stage_*.md")):
        text = path.read_text(encoding="utf-8")
        meta = parse_meta(text)
        num = int(path.stem.split("_")[1])
        short = re.sub(r"^단계\s*\d+\.\s*", "", meta["title"])
        pages.append({"kind": "stage", "num": num, "slug": path.stem, "text": text, "short": short, **meta})
    gl = SRC / "glossary.md"
    if gl.exists():
        text = gl.read_text(encoding="utf-8")
        pages.append({"kind": "glossary", "num": 999, "slug": "glossary", "text": text, "short": "용어 사전", **parse_meta(text)})
    return pages


def build() -> int:
    pages = load_pages()
    if not pages:
        print("lessons/theory/ 에 stage_*.md 가 없습니다.")
        return 1
    stages = [p for p in pages if p["kind"] == "stage"]
    THEORY_OUT.mkdir(parents=True, exist_ok=True)

    # 1) 개별 페이지
    for i, p in enumerate(stages):
        body, _ = convert(p["text"])
        prev_ = stages[i - 1] if i > 0 else None
        next_ = stages[i + 1] if i + 1 < len(stages) else None
        pager = '<div class="pager"><span>' + (
            f'<a href="{prev_["slug"]}.html">← {prev_["num"]}. {html.escape(prev_["short"])}</a>' if prev_ else ""
        ) + "</span><span>" + (
            f'<a href="{next_["slug"]}.html">{next_["num"]}. {html.escape(next_["short"])} →</a>' if next_ else ""
        ) + "</span></div>"
        page = wrap_page(p["title"], sidebar(pages, p["slug"], "", "../index.html"), body, pager)
        (THEORY_OUT / f"{p['slug']}.html").write_text(page, encoding="utf-8")

    gl = next((p for p in pages if p["kind"] == "glossary"), None)
    if gl:
        body, _ = convert(gl["text"])
        body = body.replace("</h1>", '</h1>\n<input id="gl-search" class="search" type="search" placeholder="용어 검색 (예: WOE, PSI, 누수)" autofocus>', 1)
        page = wrap_page(gl["title"], sidebar(pages, "glossary", "", "../index.html"), body, "", SEARCH_JS)
        (THEORY_OUT / "glossary.html").write_text(page, encoding="utf-8")

    # 2) 전체 한 페이지
    sections = []
    for p in stages + ([gl] if gl else []):
        prefix = f"{p['slug']}-"
        body, _ = convert(p["text"], id_prefix=prefix)
        anchor = "glossary" if p["kind"] == "glossary" else f"stage-{p['num']:02d}"
        sections.append(f'<section id="{anchor}">{body}</section>')
    all_body = "\n".join(sections)
    all_body = re.sub(r'href="stage_(\d\d)\.html(#[^"]*)?"', r'href="#stage-\1"', all_body)
    all_body = all_body.replace('href="glossary.html"', 'href="#glossary"')
    toc_items = "".join(
        f'<li><a href="#stage-{p["num"]:02d}">{p["num"]}. {html.escape(p["short"])}</a></li>' for p in stages
    ) + '<li><a href="#glossary">용어 사전</a></li>'
    head = f"<h1>이론 설명 전체 (단계 0~16 + 용어 사전)</h1><ol start=\"0\">{toc_items}</ol>"
    side = '<nav class="side"><a class="brand" href="#top">📘 Credit Model Mastery Lab</a>' + "".join(
        f'<a class="item" href="#stage-{p["num"]:02d}">{p["num"]}. {html.escape(p["short"])}</a>' for p in stages
    ) + '<a class="item" href="#glossary">용어 사전</a></nav>'
    (THEORY_OUT / "all.html").write_text(wrap_page("이론 설명 전체", side, head + all_body), encoding="utf-8")

    # 3) 시작 페이지
    cards_by_phase = []
    for phase_name, nums, desc in PHASES:
        cards = []
        for n in nums:
            p = next((s for s in stages if s["num"] == n), None)
            if p:
                cards.append(
                    f'<a class="card" href="theory/{p["slug"]}.html"><b>{n}. {html.escape(p["short"])}</b>'
                    f'<span>{html.escape(p["summary"])}</span></a>'
                )
        cards_by_phase.append(f"<h2>{html.escape(phase_name)}</h2><p>{html.escape(desc)}</p><div class=\"cards\">{''.join(cards)}</div>")
    index_body = f"""
<h1>Credit Model Mastery Lab</h1>
<blockquote><p>신용평가모형을 <b>처음부터 끝까지 직접 만들어 보는</b> 학습 프로그램의 이론 설명 사이트입니다. 각 단계는
「왜 생겼는가 → 왜 필요한가 → 무엇을 하는가 → 어떻게 작동하는가 → 가정 → 한계 → 대안 → 실무 적용 → 지식 확인 → 참고자료」 순서로 구성되어 있습니다.</p></blockquote>
<div class="admonition warning"><p class="admonition-title">교육용 프로그램입니다</p>
<p>공개 교육 데이터 결과는 실제 은행의 운영 성능·규제 적합성을 의미하지 않습니다. 개인정보·실제 고객 데이터를 사용하지 마세요.</p></div>
<h2>이렇게 학습하세요</h2>
<ol><li>이 사이트에서 해당 단계의 <b>이론</b>을 읽는다.</li>
<li>Colab 에서 같은 단계의 <b>실습 노트북</b>(<code>notebooks/</code>)을 직접 실행한다. (예측 → 실행 → 비교)</li>
<li>퀴즈로 확인하고, 틀린 개념은 해당 단계로 돌아와 복습한다.</li></ol>
<p><a href="theory/glossary.html">용어 사전(검색)</a> · <a href="theory/all.html">전체를 한 페이지로 보기</a></p>
{''.join(cards_by_phase)}
"""
    (OUT / "index.html").write_text(wrap_page("학습 로드맵", sidebar(pages, None, "theory/", "index.html"), index_body), encoding="utf-8")
    print(f"built {len(stages)} stage pages + glossary + all.html + index.html → {OUT.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(build())
