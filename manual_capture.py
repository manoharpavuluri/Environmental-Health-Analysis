#!/usr/bin/env python3
"""
Manual screenshot capture instructions for WebGL visualizations.
Since automated WebGL capture is failing, this provides a manual approach.
"""

import os
import glob

def list_html_files():
    """List all HTML visualization files."""
    html_files = []
    
    # Search for HTML files in analysis directories
    analysis_dirs = [
        "xenium_acta2_analysis",
        "xenium_dual_gene_analysis", 
        "xenium_expression_filtered_analysis",
        "xenium_all_transcripts_analysis"
    ]
    
    for directory in analysis_dirs:
        if os.path.exists(directory):
            html_files.extend(glob.glob(f"{directory}/*.html"))
    
    return html_files

def generate_manual_instructions():
    """Generate manual capture instructions."""
    html_files = list_html_files()
    
    print("=" * 80)
    print("MANUAL SCREENSHOT CAPTURE INSTRUCTIONS")
    print("=" * 80)
    print()
    print("Since automated WebGL capture is failing, please follow these steps:")
    print()
    print("1. Open each HTML file in a modern browser (Chrome, Firefox, Safari)")
    print("2. Wait for the 3D visualization to fully load")
    print("3. Take a screenshot using your system's screenshot tool")
    print("4. Save the screenshot with the appropriate name in the 'images' folder")
    print()
    print("HTML Files to capture:")
    print("-" * 50)
    
    for html_file in html_files:
        filename = os.path.basename(html_file)
        output_name = filename.replace('.html', '.png')
        print(f"• {html_file}")
        print(f"  → Save as: images/{output_name}")
        print()
    
    print("=" * 80)
    print("ALTERNATIVE: Use browser developer tools")
    print("=" * 80)
    print("1. Open HTML file in browser")
    print("2. Press F12 to open developer tools")
    print("3. Go to Console tab")
    print("4. Run: document.body.style.background = 'white'")
    print("5. Take screenshot")
    print()
    print("=" * 80)

def create_batch_open_script():
    """Create a script to open all HTML files in browser."""
    html_files = list_html_files()
    
    script_content = """#!/bin/bash
# Script to open all HTML visualization files in default browser

echo "Opening HTML visualization files in browser..."
"""
    
    for html_file in html_files:
        script_content += f'open "{html_file}"\n'
    
    script_content += """
echo "All files opened. Please capture screenshots manually."
echo "Save screenshots in the 'images' folder with appropriate names."
"""
    
    with open("open_visualizations.sh", "w") as f:
        f.write(script_content)
    
    # Make executable
    os.chmod("open_visualizations.sh", 0o755)
    
    print("Created 'open_visualizations.sh' script")
    print("Run: ./open_visualizations.sh to open all HTML files in browser")

def main():
    """Main function."""
    print("WebGL Screenshot Capture Helper")
    print("=" * 50)
    
    # Generate manual instructions
    generate_manual_instructions()
    
    # Create batch open script
    create_batch_open_script()
    
    print("\nNext steps:")
    print("1. Run: ./open_visualizations.sh")
    print("2. Capture screenshots manually")
    print("3. Save them in the 'images' folder")
    print("4. Update README.md if needed")

if __name__ == "__main__":
    main()