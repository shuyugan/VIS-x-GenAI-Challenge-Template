# md_pdf.py
import os, re, io, html
from datetime import date
from typing import List, Optional
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, ListFlowable, ListItem, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch

try:
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import get_lexer_by_name, TextLexer
    HAVE_PYGMENTS = True
except Exception:
    HAVE_PYGMENTS = False

# --- 基础样式 ---
def _build_styles():
    styles = getSampleStyleSheet()
    styles["Heading1"].fontSize = 18; styles["Heading1"].spaceAfter = 8
    styles["Heading2"].fontSize = 14; styles["Heading2"].spaceAfter = 6
    styles["Normal"].leading = 16

    # Caption 若已存在就别重复添加
    if "Caption" not in styles:
        styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"],
                                  alignment=TA_CENTER, textColor="#666666", fontSize=9, spaceBefore=4))

    # ✅ 用新的名字，避免与内置 "Code" 冲突
    if "CodeBlock" not in styles:
        styles.add(ParagraphStyle(name="CodeBlock", parent=styles["Normal"],
                                  fontName="Courier", fontSize=9, leading=11,
                                  backColor="#F7F7F7"))
    return styles

IMG_MD = re.compile(r'!\[(?P<alt>.*?)\]\((?P<src>[^)]+)\)')
OL_MD  = re.compile(r'^\s*\d+\.\s+(.*)$')      # 1. item
UL_MD  = re.compile(r'^\s*[-*+]\s+(.*)$')      # - item
FENCE  = re.compile(r'^```(\w+)?\s*$')         # ```python / ``` 

def _inline_md_to_html(text: str) -> str:
    # 极简内联转换：**粗体**、*斜体*、`行内代码`
    # 先转义 HTML，再做替换
    t = html.escape(text, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", t)
    return t

def _add_paragraph(story: List, styles, lines_bucket: List[str]):
    if not lines_bucket: return
    para = " ".join(lines_bucket).strip()
    if para:
        story.append(Paragraph(_inline_md_to_html(para), styles["Normal"]))
        story.append(Spacer(1, 0.15 * inch))
    lines_bucket.clear()

def _add_list(story: List, styles, items: List[str], ordered: bool):
    if not items: return
    flow = ListFlowable([ListItem(Paragraph(_inline_md_to_html(i), styles["Normal"])) for i in items],
                        bulletType="1" if ordered else "bullet", start="1")
    story.append(flow); story.append(Spacer(1, 0.15 * inch))
    items.clear()

def _add_image(story, styles, src: str, alt: str, base_dir: Optional[str]):
    if base_dir and not (src.startswith("http://") or src.startswith("https://") or os.path.isabs(src)):
        src = os.path.join(base_dir, src)
    try:
        img = Image(src)                 # 不传 preserveAspectRatio
        img.hAlign = "CENTER"
        img._restrictSize(6.5*inch, 8*inch)  # 保持比例，限制最大宽高
        story.append(img)
        if alt.strip():
            story.append(Paragraph(html.escape(alt.strip()), styles["Caption"]))
        story.append(Spacer(1, 0.2*inch))
    except Exception as e:
        # 调试时也把异常带出来更直观
        story.append(Paragraph(f"[Image not found or load error: {html.escape(src)} | {html.escape(str(e))}]", styles["Normal"]))
        story.append(Spacer(1, 0.1*inch))

def _add_codeblock(story, styles, code: str, lang: Optional[str]):
    code = code.rstrip("\n")
    story.append(Preformatted(code, styles["CodeBlock"]))  # ← 这里改名
    story.append(Spacer(1, 0.15*inch))

def _draw_header_footer(c, doc):
    c.saveState()
    width, height = doc.pagesize
    c.setFont("Helvetica", 9)

    # ===== Header =====
    # 左上角标题
    c.drawString(doc.leftMargin, height - 0.5*inch, "Data Visualization Report")

    # 右上角日期（写死为生成日期）
    c.drawRightString(width - doc.rightMargin, height - 0.5*inch,
                      date.today().isoformat())

    # ===== Footer =====
    # 右下角页码
    c.drawRightString(width - doc.rightMargin, 0.5*inch,
                      f"Page {c.getPageNumber()}")

    c.restoreState()

def markdown_to_pdf(md_text: str, output_pdf: str, base_dir: Optional[str] = None, pagesize=letter):
    styles = _build_styles()
    doc = SimpleDocTemplate(output_pdf, pagesize=pagesize,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    story: List = []

    lines = md_text.splitlines()
    bucket: List[str] = []
    ul_items: List[str] = []
    ol_items: List[str] = []

    in_code = False
    code_lang = None
    code_buf: List[str] = []

    def flush_all():
        _add_paragraph(story, styles, bucket)
        _add_list(story, styles, ul_items, ordered=False)
        _add_list(story, styles, ol_items, ordered=True)

    for raw in lines:
        line = raw.rstrip("\n")

        # 代码围栏
        m_f = FENCE.match(line)
        if m_f:
            if not in_code:
                # 进入代码块
                flush_all()
                in_code = True
                code_lang = (m_f.group(1) or "").strip().lower() or None
                code_buf = []
            else:
                # 结束代码块
                _add_codeblock(story, styles, "\n".join(code_buf), code_lang)
                in_code = False
                code_lang = None
                code_buf = []
            continue

        if in_code:
            code_buf.append(line)
            continue

        # 空行：结束段落/列表
        if not line.strip():
            flush_all()
            continue

        # 图片
        m_img = IMG_MD.search(line)
        if m_img:
            flush_all()
            _add_image(story, styles, m_img.group("src").strip(), m_img.group("alt") or "", base_dir)
            continue

        # 标题
        if line.startswith("## "):
            flush_all()
            story.append(Paragraph(_inline_md_to_html(line[3:].strip()), styles["Heading2"]))
            story.append(Spacer(1, 0.12*inch))
            continue
        if line.startswith("# "):
            flush_all()
            story.append(Paragraph(_inline_md_to_html(line[2:].strip()), styles["Heading1"]))
            story.append(Spacer(1, 0.16*inch))
            continue

        # 列表
        m_ul = UL_MD.match(line)
        if m_ul:
            if ol_items: _add_list(story, styles, ol_items, ordered=True)
            ul_items.append(m_ul.group(1))
            continue
        m_ol = OL_MD.match(line)
        if m_ol:
            if ul_items: _add_list(story, styles, ul_items, ordered=False)
            ol_items.append(m_ol.group(1))
            continue

        # 普通段落（合并行）
        bucket.append(line)

    # 收尾
    if in_code:
        _add_codeblock(story, styles, "\n".join(code_buf), code_lang)
    flush_all()

    doc.build(story, onFirstPage=_draw_header_footer, onLaterPages=_draw_header_footer)
    return output_pdf


if __name__ == "__main__":
    markdown_text = """
    # Head
## Head2
### Head3
sddasdasdas
![dasda](plot_1.png)
    """
    markdown_to_pdf(markdown_text, "report.pdf", os.getcwd())