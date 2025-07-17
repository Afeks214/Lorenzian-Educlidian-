#!/usr/bin/env python3
"""
Fix corrupted strategic notebooks by rebuilding them based on the working version
"""

import json
import shutil
import os

def fix_corrupted_notebooks():
    """Fix corrupted notebooks by using the working version as template"""
    
    # The working notebook
    working_notebook = "strategic_mappo_training.ipynb"
    
    # Corrupted notebooks to fix
    corrupted_notebooks = [
        "strategic_mappo_training_complete.ipynb",
        "strategic_mappo_training_fixed.ipynb"
    ]
    
    print("🔧 Fixing corrupted strategic notebooks...")
    
    # Load the working notebook
    try:
        with open(working_notebook, 'r') as f:
            working_data = json.load(f)
        print(f"✅ Loaded working notebook: {working_notebook}")
    except Exception as e:
        print(f"❌ Failed to load working notebook: {e}")
        return False
    
    # Fix each corrupted notebook
    for notebook_path in corrupted_notebooks:
        try:
            print(f"\nFixing {notebook_path}...")
            
            # Create backup
            backup_path = f"{notebook_path}.backup"
            if os.path.exists(notebook_path):
                shutil.copy2(notebook_path, backup_path)
                print(f"   Backup created: {backup_path}")
            
            # Write the working version to the corrupted file
            with open(notebook_path, 'w') as f:
                json.dump(working_data, f, indent=2, ensure_ascii=False)
            
            # Verify the fix
            with open(notebook_path, 'r') as f:
                test_data = json.load(f)
            
            print(f"✅ Fixed {notebook_path}: {len(test_data['cells'])} cells")
            
        except Exception as e:
            print(f"❌ Failed to fix {notebook_path}: {e}")
    
    return True

if __name__ == "__main__":
    fix_corrupted_notebooks()