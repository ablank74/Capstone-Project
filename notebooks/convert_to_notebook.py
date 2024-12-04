import nbformat as nbf
import re

def py_to_notebook(py_file, ipynb_file):
    notebook = nbf.v4.new_notebook()
    cells = []
    
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by cell markers
    cell_blocks = re.split(r'# %% \[(\w+)\]', content)
    
    # First block might be empty
    if cell_blocks[0].strip():
        cells.append(nbf.v4.new_code_cell(cell_blocks[0]))
    
    # Process remaining blocks
    for i in range(1, len(cell_blocks), 2):
        cell_type = cell_blocks[i]
        content = cell_blocks[i + 1].strip()
        
        if cell_type == 'markdown':
            cells.append(nbf.v4.new_markdown_cell(content))
        else:
            cells.append(nbf.v4.new_code_cell(content))
    
    notebook.cells = cells
    
    # Write the notebook
    with open(ipynb_file, 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)

# Convert the file
py_to_notebook('TicketAnalysis.py', 'TicketAnalysis.ipynb') 