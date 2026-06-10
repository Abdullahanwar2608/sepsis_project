import os
import re
import glob

def clean_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        stripped = line.lstrip()
        # Keep docstrings and decorators
        
        # If it's a comment line
        if stripped.startswith('#'):
            # Keep important section headers
            if stripped.startswith('# ──') or stripped.startswith('# =') or stripped.startswith('# TODO') or stripped.startswith('# Note') or stripped.startswith('# PhysioNet'):
                new_lines.append(line)
            # Otherwise drop it to clean up the code
            continue
            
        # If there's an inline comment, strip it (basic check, might catch # in strings but we'll be careful)
        # For safety, let's just drop full line comments that aren't important headers
        new_lines.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

for py_file in glob.glob('*.py') + glob.glob('mimic_extraction/*.py') + glob.glob('deep_learning/*.py'):
    clean_file(py_file)
    print(f"Cleaned {py_file}")
