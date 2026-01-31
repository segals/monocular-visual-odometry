#!/usr/bin/env python3
"""Convert PROJECT_REPORT.md to HTML (open in browser and print to PDF)"""
import markdown
from pathlib import Path

# Read markdown
md_path = Path("PROJECT_REPORT.md")
md_content = md_path.read_text(encoding='utf-8')

# Convert to HTML
html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

# Wrap with styling
full_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    max-width: 800px;
    margin: 0 auto;
    padding: 40px;
    color: #333;
}}
h1 {{
    font-size: 24pt;
    color: #1a1a1a;
    border-bottom: 2px solid #333;
    padding-bottom: 10px;
    margin-top: 0;
}}
h2 {{
    font-size: 16pt;
    color: #2c3e50;
    margin-top: 30px;
    border-bottom: 1px solid #ddd;
    padding-bottom: 5px;
}}
h3 {{
    font-size: 13pt;
    color: #34495e;
    margin-top: 20px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
    font-size: 10pt;
}}
th, td {{
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}}
th {{
    background-color: #f5f5f5;
    font-weight: bold;
}}
tr:nth-child(even) {{
    background-color: #fafafa;
}}
code {{
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 10pt;
}}
pre {{
    background-color: #f4f4f4;
    padding: 15px;
    border-radius: 5px;
    overflow-x: auto;
    font-size: 9pt;
}}
pre code {{
    padding: 0;
    background: none;
}}
strong {{
    color: #1a1a1a;
}}
hr {{
    border: none;
    border-top: 1px solid #ddd;
    margin: 30px 0;
}}
p {{
    margin: 10px 0;
}}
ul, ol {{
    margin: 10px 0;
    padding-left: 25px;
}}
li {{
    margin: 5px 0;
}}
</style>
</head>
<body>
{html_content}
</body>
</html>
"""

# Save HTML for reference
html_path = Path("PROJECT_REPORT.html")
html_path.write_text(full_html, encoding='utf-8')
print(f"HTML saved: {html_path.absolute()}")
print(f"\nOpen in browser and use Ctrl+P to print to PDF")
