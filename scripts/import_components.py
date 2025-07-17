#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

ALGOSPACE_PATH = Path("~/AlgoSpace").expanduser()  # Adjust this path
GRANDMODEL_PATH = Path.cwd()

def import_components():
    """Import selected components from AlgoSpace"""
    
    imports = {
        # Core
        "src/core/events.py": "src/core/events.py",
        "src/core/kernel.py": "src/core/kernel.py",
        "src/core/component_base.py": "src/core/component_base.py",
        
        # Data Pipeline
        "src/data/handlers.py": "src/data/data_handler.py",
        "src/data/bar_generator.py": "src/data/bar_generator.py",
        "src/data/validators.py": "src/data/validators.py",
        
        # Indicators
        "src/indicators/engine.py": "src/indicators/engine.py",
        "src/indicators/base.py": "src/indicators/base.py",
        "src/indicators/mlmi.py": "src/indicators/custom/mlmi.py",
        "src/indicators/nwrqk.py": "src/indicators/custom/nwrqk.py",
        "src/indicators/fvg.py": "src/indicators/custom/fvg.py",
        "src/indicators/lvn.py": "src/indicators/custom/lvn.py",
        "src/indicators/mmd.py": "src/indicators/custom/mmd.py",
        
        # Matrix Assemblers
        "src/matrix/base.py": "src/matrix/base.py",
        "src/matrix/assembler_30m.py": "src/matrix/assembler_30m.py",
        "src/matrix/assembler_5m.py": "src/matrix/assembler_5m.py",
        "src/matrix/normalizers.py": "src/matrix/normalizers.py",
        
        # Synergy Detector
        "src/agents/synergy/detector.py": "src/synergy/detector.py",
        "src/agents/synergy/patterns.py": "src/synergy/patterns.py",
        "src/agents/synergy/base.py": "src/synergy/base.py",
        "src/agents/synergy/sequence.py": "src/synergy/sequence.py",
    }
    
    success = 0
    for src, dst in imports.items():
        src_path = ALGOSPACE_PATH / src
        dst_path = GRANDMODEL_PATH / dst
        
        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            print(f"✅ Imported: {src}")
            success += 1
        else:
            print(f"❌ Not found: {src}")
    
    print(f"\nImported {success}/{len(imports)} files")

if __name__ == "__main__":
    import_components()