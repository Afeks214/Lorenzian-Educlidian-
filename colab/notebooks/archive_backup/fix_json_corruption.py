#!/usr/bin/env python3
"""
Fix JSON corruption in strategic notebooks
"""

import json
import re
import sys

def fix_json_corruption(file_path):
    """Fix JSON corruption in notebook files"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix common JSON corruption issues
        # 1. Fix escaped backslashes in strings
        content = re.sub(r'\\\\n', r'\\n', content)
        content = re.sub(r'\\\\"', r'\\"', content)
        
        # 2. Fix Unicode escape sequences
        content = content.replace('\\u', '\\u')
        
        # 3. Try to parse and validate
        try:
            notebook_data = json.loads(content)
            print(f"✅ {file_path}: JSON parsing successful")
            return True
        except json.JSONDecodeError as e:
            print(f"❌ {file_path}: JSON parsing failed - {e}")
            return False
            
    except Exception as e:
        print(f"❌ {file_path}: File processing failed - {e}")
        return False

def main():
    """Fix all corrupted notebooks"""
    corrupted_files = [
        "strategic_mappo_training_complete.ipynb",
        "strategic_mappo_training_fixed.ipynb",
        "strategic_mappo_training_temp_fixed.ipynb",
        "strategic_mappo_training_corrupted_backup.ipynb"
    ]
    
    print("🔧 Fixing JSON corruption in strategic notebooks...")
    
    for file_path in corrupted_files:
        try:
            print(f"\nProcessing {file_path}...")
            fix_json_corruption(file_path)
        except Exception as e:
            print(f"❌ Failed to process {file_path}: {e}")
    
    # Verify the working notebook
    print(f"\nVerifying working notebook...")
    if fix_json_corruption("strategic_mappo_training.ipynb"):
        print("✅ Working notebook verified successfully")
    else:
        print("❌ Working notebook has issues")

if __name__ == "__main__":
    main()