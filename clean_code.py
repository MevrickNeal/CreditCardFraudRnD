import os
import re

def remove_comments_python(code):
    # This is a bit simplified but usually enough for # based comments
    # We want to avoid removing # inside strings, but for this project there isn't much complexity
    lines = code.splitlines()
    clean_lines = []
    for line in lines:
        if '#' in line:
            # Check if # is inside quotes (basic check)
            # Find the first # that is not quoted
            in_quote = False
            quote_char = None
            comment_start = -1
            for i, char in enumerate(line):
                if char in ("'", '"'):
                    if not in_quote:
                        in_quote = True
                        quote_char = char
                    elif char == quote_char:
                        in_quote = False
                        quote_char = None
                elif char == '#' and not in_quote:
                    comment_start = i
                    break
            if comment_start != -1:
                line = line[:comment_start].rstrip()
        if line.strip() or not clean_lines or clean_lines[-1].strip(): # Keep some spacing but remove comments
            clean_lines.append(line)
    return "\n".join(clean_lines)

def remove_comments_js(code):
    # Remove // comments
    code = re.sub(r'//.*', '', code)
    # Remove /* */ comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code

def remove_comments_html(code):
    return re.sub(r'<!--.*?-->', '', code, flags=re.DOTALL)

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    ext = os.path.splitext(filepath)[1]
    if ext == '.py':
        new_content = remove_comments_python(content)
    elif ext == '.js' or ext == '.css':
        new_content = remove_comments_js(content)
    elif ext == '.html':
        new_content = remove_comments_html(content)
    else:
        return

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Cleaned {filepath}")

files_to_clean = [
    "data_pipeline.py",
    "train_engine.py",
    "generate_profiles.py",
    "generate_visuals.py",
    "backend/app.py",
    "models/generative.py",
    "models/ensemble.py",
    "frontend/app.js",
    "frontend/index.html"
]

for f in files_to_clean:
    if os.path.exists(f):
        process_file(f)
    else:
        print(f"File {f} not found")
