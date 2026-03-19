import sys
import json
import os

# List of notebooks to fix
notebooks = [
    'notebooks/02_preprocess_feature.ipynb',
    'notebooks/03_mining_or_clustering.ipynb',
    'notebooks/04_modeling.ipynb',
    'notebooks/04b_semi_supervised.ipynb',
    'notebooks/05_evaluation_report.ipynb'
]

for nb_path in notebooks:
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # Find first code cell with matplotlib import
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = cell['source']
                if isinstance(source, list):
                    source_str = ''.join(source)
                else:
                    source_str = source
                
                if 'matplotlib' in source_str and '%matplotlib' not in source_str:
                    # Add matplotlib config
                    if isinstance(source, list):
                        cell['source'] = source + ['\n# Plot settings\n%matplotlib inline\nplt.rcParams["figure.figsize"] = (10, 6)\n']
                    else:
                        cell['source'] = source + '\n\n# Plot settings\n%matplotlib inline\nplt.rcParams["figure.figsize"] = (10, 6)'
                    print(f"Fixed: {nb_path}")
                    break
        
        # Save
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        
    except Exception as e:
        print(f"Error with {nb_path}: {e}")

print("\nDone fixing notebooks!")
