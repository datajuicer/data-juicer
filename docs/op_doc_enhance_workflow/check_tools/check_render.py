#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check operators in markdown files that lack examples
"""

import os
import re
from pathlib import Path
from typing import List, Dict

def get_test_file_path(md_file_path: str) -> str:
    """
    Generate corresponding test file path from markdown file path
    
    Args:
        md_file_path: markdown file path, e.g. docs/operators/aggregator/entity_attribute_aggregator.md
        
    Returns:
        test file path, e.g. tests/ops/aggregator/test_entity_attribute_aggregator.py
    """
    try:
        path = Path(md_file_path)
        
        # Extract filename without extension
        filename = path.stem
        
        # Get path parts relative to docs/operators
        parts = path.parts
        if 'docs' in parts and 'operators' in parts:
            docs_idx = parts.index('docs')
            operators_idx = parts.index('operators')
            
            # Extract subdirectory structure (parts after operators, excluding filename)
            if operators_idx + 1 < len(parts) - 1:
                subdirs = parts[operators_idx + 1:-1]
                subdir_path = '/'.join(subdirs)
                test_path = f"tests/ops/{subdir_path}/test_{filename}.py"
            else:
                test_path = f"tests/ops/test_{filename}.py"
        else:
            # Fallback to default format if path structure doesn't match
            test_path = f"tests/ops/test_{filename}.py"
            
        return test_path
    except Exception:
        # Return default format on error
        filename = Path(md_file_path).stem
        return f"tests/ops/test_{filename}.py"

def check_missing_examples(file_path: str) -> Dict:
    """
    Check if a single markdown file lacks examples
    
    Args:
        file_path: markdown file path
        
    Returns:
        check result dictionary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract file title (operator name)
        title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        operator_name = title_match.group(1).strip() if title_match else "Unknown"
        
        # Extract operator type
        type_match = re.search(r'Type 算子类型: \*\*(.+?)\*\*', content)
        operator_type = type_match.group(1).strip() if type_match else "Unknown"
        
        # Extract tags
        tags_match = re.search(r'Tags 标签: (.+)', content)
        tags = tags_match.group(1).strip() if tags_match else ""
        
        # Check for examples section
        examples_section = re.search(r'## 📊 Effect demonstration 效果演示\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
        
        has_examples = False
        example_count = 0
        
        if examples_section:
            examples_content = examples_section.group(1).strip()
            
            # Check if it's just "not available"
            if examples_content == "not available 暂无":
                has_examples = False
            else:
                # Count example methods
                example_methods = re.findall(r'### (.+)', examples_content)
                example_count = len(example_methods)
                has_examples = example_count > 0
        
        # Generate corresponding test file path
        test_file_path = get_test_file_path(file_path)
        
        return {
            'operator': operator_name,
            'type': operator_type,
            'tags': tags,
            'file': file_path,
            'test_file': test_file_path,
            'has_examples': has_examples,
            'example_count': example_count
        }
    
    except Exception as e:
        print(f"❌ Error processing file {file_path}: {e}")
        return None

def scan_operators_directory() -> List[str]:
    """
    Scan all markdown files in docs/operators/ directory
    
    Returns:
        list of markdown file paths
    """
    # Get current script directory, then find project root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent  # Navigate back from demos/op_doc_enhance_workflow/check_tools/
    operators_dir = project_root / "docs" / "operators"
    
    if not operators_dir.exists():
        print(f"❌ Directory not found: {operators_dir}")
        return []
    
    # Recursively find all .md files
    md_files = list(operators_dir.rglob("*.md"))
    return [str(f) for f in md_files]

def sort_operators(operators: List[Dict], sort_by: str = 'name') -> List[Dict]:
    """
    Sort operator list
    
    Args:
        operators: operator list
        sort_by: sorting method ('name', 'type', 'file')
        
    Returns:
        sorted operator list
    """
    if sort_by == 'name':
        return sorted(operators, key=lambda x: x['operator'].lower())
    elif sort_by == 'type':
        return sorted(operators, key=lambda x: (x['type'].lower(), x['operator'].lower()))
    elif sort_by == 'file':
        return sorted(operators, key=lambda x: x['file'].lower())
    else:
        return operators

def output_text(operators: List[Dict], by_type: bool = True, show_all: bool = True, sort_by: str = 'name'):
    """Text format output"""
    if by_type:
        # Group by type, but sort within groups by specified method
        type_groups = {}
        for op in operators:
            op_type = op['type']
            if op_type not in type_groups:
                type_groups[op_type] = []
            type_groups[op_type].append(op)
        
        # Sort operators within each type group
        for op_type in sorted(type_groups.keys()):
            ops = sort_operators(type_groups[op_type], sort_by)
            missing_count = len([op for op in ops if not op['has_examples']])
            total_count = len(ops)
            
            print(f"\n📂 {op_type} ({total_count} operators, {missing_count} missing examples)")
            print("-" * 60)
            
            for i, op in enumerate(ops, 1):
                status = "✅" if op['has_examples'] else "❌"
                example_info = f"({op['example_count']} examples)" if op['has_examples'] else "(no examples)"
                print(f"   {i:2d}. {status} {op['operator']} {example_info}")
                print(f"       📁 {os.path.basename(op['file'])}")
                print(f"       🧪 {op['test_file']}")
                if op['tags']:
                    print(f"       🏷️  {op['tags']}")
    else:
        # Simple list, already sorted
        for i, op in enumerate(operators, 1):
            status = "✅" if op['has_examples'] else "❌"
            example_info = f"({op['example_count']} examples)" if op['has_examples'] else "(no examples)"
            
            print(f"\n{i:2d}. {status} {op['operator']} - {op['type']} {example_info}")
            if op['tags']:
                print(f"    🏷️  {op['tags']}")
            print(f"    📁 {op['file']}")
            print(f"    🧪 {op['test_file']}")

def main():
    """Main function"""
    print("🔍 Scanning docs/operators/ directory...")
    
    # Get all markdown files
    files_to_check = scan_operators_directory()
    
    if not files_to_check:
        print("❌ No markdown files found")
        return
    
    print(f"📄 Found {len(files_to_check)} markdown files")
    
    # Collect all operator information
    all_operators = []
    for file_path in files_to_check:
        result = check_missing_examples(file_path)
        if result:
            all_operators.append(result)
    
    if not all_operators:
        print("❌ No valid operator documentation found")
        return
    
    # Sort by operator type
    all_operators = sort_operators(all_operators, 'type')
    
    # Filter operators without examples
    missing_examples = [op for op in all_operators if not op['has_examples']]
    
    print(f"\n📊 Check Results:")
    print("=" * 80)
    
    # Display all operators, grouped by type
    output_text(all_operators, by_type=True, show_all=True, sort_by='name')
    
    # Statistics
    total_operators = len(all_operators)
    operators_with_examples = len([op for op in all_operators if op['has_examples']])
    operators_without_examples = len(missing_examples)
    
    print(f"\n📈 Statistics:")
    print(f"   - Total operators: {total_operators}")
    print(f"   - Operators with examples: {operators_with_examples}")
    print(f"   - Operators missing examples: {operators_without_examples}")
    print(f"   - Example coverage: {operators_with_examples/total_operators*100:.1f}%" if total_operators > 0 else "   - Example coverage: 0%")
    
    if missing_examples:
        print(f"\n❌ Operators missing examples:")
        for i, op in enumerate(missing_examples, 1):
            print(f"   {i:2d}. {op['operator']} ({op['type']})")
            print(f"       🧪 {op['test_file']}")

if __name__ == "__main__":
    main()