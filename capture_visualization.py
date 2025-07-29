#!/usr/bin/env python3
"""
Reusable script to capture screenshots of HTML visualizations.
Usage: python3 capture_visualization.py [html_file] [output_file]
"""

import os
import sys
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def setup_chrome_driver():
    """Setup Chrome driver with headless options and WebGL support."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-gpu-sandbox")
    chrome_options.add_argument("--enable-gpu")
    chrome_options.add_argument("--enable-webgl")
    chrome_options.add_argument("--enable-3d-apis")
    chrome_options.add_argument("--use-gl=desktop")
    chrome_options.add_argument("--ignore-gpu-blocklist")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-background-timer-throttling")
    chrome_options.add_argument("--disable-backgrounding-occluded-windows")
    chrome_options.add_argument("--disable-renderer-backgrounding")
    chrome_options.add_argument("--disable-features=TranslateUI")
    chrome_options.add_argument("--disable-ipc-flooding-protection")
    return webdriver.Chrome(options=chrome_options)

def capture_html_screenshot(html_file, output_file, wait_time=10):
    """Capture screenshot of an HTML file with extended wait for WebGL rendering."""
    if not os.path.exists(html_file):
        print(f"Error: HTML file not found: {html_file}")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    driver = setup_chrome_driver()
    try:
        # Convert relative path to absolute file URL
        abs_path = os.path.abspath(html_file)
        file_url = f"file://{abs_path}"
        
        print(f"Loading {html_file}...")
        driver.get(file_url)
        
        # Wait for page to load and WebGL to initialize
        time.sleep(wait_time)
        
        # Wait for any dynamic content to load
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except:
            print(f"Warning: Timeout waiting for {html_file}")
        
        # Additional wait for WebGL content to render
        print("Waiting for WebGL content to render...")
        time.sleep(5)
        
        # Capture screenshot
        print(f"Capturing screenshot to {output_file}...")
        driver.save_screenshot(output_file)
        print(f"✓ Screenshot saved: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error capturing {html_file}: {e}")
        return False
    finally:
        driver.quit()

def main():
    """Main function."""
    if len(sys.argv) != 3:
        print("Usage: python3 capture_visualization.py <html_file> <output_file>")
        print("Example: python3 capture_visualization.py xenium_acta2_analysis/acta2_3d_scatter.html images/new_screenshot.png")
        sys.exit(1)
    
    html_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = capture_html_screenshot(html_file, output_file)
    if success:
        print(f"Successfully captured {html_file} to {output_file}")
    else:
        print(f"Failed to capture {html_file}")
        sys.exit(1)

if __name__ == "__main__":
    main()