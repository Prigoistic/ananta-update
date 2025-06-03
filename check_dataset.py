"""
Dataset Structure Diagnostic Tool
=================================

This script helps diagnose the actual structure of your mathematics dataset
and provides guidance on how to fix any path or structure issues.

Usage: python check_dataset.py
"""

import os
from pathlib import Path

def check_dataset_structure():
    """Check and display the actual dataset structure."""
    print("🔍 Checking dataset structure...\n")
    
    # Check current directory
    current_dir = Path(".")
    print(f"Current directory: {current_dir.absolute()}")
    print("\nFiles and directories in current location:")
    
    for item in current_dir.iterdir():
        if item.is_dir():
            print(f"  📁 {item.name}/")
        else:
            print(f"  📄 {item.name}")
    
    print("\n" + "="*50)
    
    # Look for potential dataset directories
    potential_dataset_dirs = []
    
    for item in current_dir.iterdir():
        if item.is_dir():
            name_lower = item.name.lower()
            if any(keyword in name_lower for keyword in ['math', 'dataset', 'train', 'data']):
                potential_dataset_dirs.append(item)
    
    if potential_dataset_dirs:
        print("\n🎯 Found potential dataset directories:")
        for i, dir_path in enumerate(potential_dataset_dirs, 1):
            print(f"\n{i}. {dir_path.name}/")
            
            # Check contents of each potential directory
            try:
                contents = list(dir_path.iterdir())
                print(f"   Contents ({len(contents)} items):")
                
                for item in contents[:10]:  # Show first 10 items
                    if item.is_dir():
                        print(f"     📁 {item.name}/")
                    else:
                        print(f"     📄 {item.name}")
                
                if len(contents) > 10:
                    print(f"     ... and {len(contents) - 10} more items")
                
                # Check if this looks like the DeepMind dataset structure
                subdir_names = [item.name for item in contents if item.is_dir()]
                expected_dirs = ['train-easy', 'train-medium', 'train-hard', 'extrapolate', 'interpolate']
                
                if any(expected in subdir_names for expected in expected_dirs):
                    print(f"   ✅ This looks like the correct dataset structure!")
                elif any('.txt' in item.name for item in contents if item.is_file()):
                    print(f"   📝 Contains .txt files - might need restructuring")
                
            except PermissionError:
                print(f"   ❌ Permission denied accessing {dir_path}")
            except Exception as e:
                print(f"   ❌ Error accessing {dir_path}: {e}")
    
    # Check for .txt files in current directory
    txt_files = [f for f in current_dir.iterdir() if f.is_file() and f.suffix == '.txt']
    if txt_files:
        print(f"\n📝 Found {len(txt_files)} .txt files in current directory:")
        for txt_file in txt_files[:5]:  # Show first 5
            print(f"   📄 {txt_file.name}")
        if len(txt_files) > 5:
            print(f"   ... and {len(txt_files) - 5} more .txt files")
    
    print("\n" + "="*50)
    
    # Provide recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if not potential_dataset_dirs:
        print(" No dataset directories found.")
        print("   Please ensure you've extracted the mathematics_dataset.tar file")
        print("   Expected structure:")
        print("   mathematics_dataset-v1.0/")
        print("   ├── train-easy/")
        print("   ├── train-medium/")
        print("   ├── train-hard/")
        print("   ├── extrapolate/")
        print("   └── interpolate/")
    
    else:
        print(" Dataset directories found!")
        print("\nTo process your dataset:")
        
        for i, dir_path in enumerate(potential_dataset_dirs, 1):
            print(f"\n{i}. For {dir_path.name}/:")
            print(f"   python data_processor.py --dataset_dir \"{dir_path.name}\"")
    
    print("\n📋 DEBUGGING COMMANDS:")
    print("1. Check what's in a specific directory:")
    print("   ls -la mathematics_dataset-v1.0/  (Linux/Mac)")
    print("   dir mathematics_dataset-v1.0\\  (Windows)")
    
    print("\n2. Find all .txt files:")
    print("   find . -name '*.txt' | head -10  (Linux/Mac)")
    print("   dir /s *.txt  (Windows)")

if __name__ == "__main__":
    check_dataset_structure() 