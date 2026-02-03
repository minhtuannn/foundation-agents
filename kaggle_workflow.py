import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- CONFIGURATION ---
# Add all the competition homepages you want to crawl here
URLS_TO_CRAWL = [
    "https://www.kaggle.com/competitions/playground-series-s5e12"
]

# --- DRIVER SETUP ---
def setup_driver():
    """Sets up a stable Chrome driver with stealth options."""
    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Use a persistent profile to avoid CAPTCHA on subsequent runs
    profile_path = os.path.join(os.getcwd(), "selenium_profile")
    options.add_argument(f"--user-data-dir={profile_path}")

    try:
        driver_path = ChromeDriverManager().install()
        if "THIRD_PARTY_NOTICES" in driver_path:
            driver_dir = os.path.dirname(driver_path)
            driver_path = os.path.join(driver_dir, "chromedriver")
        if os.path.exists(driver_path):
            os.chmod(driver_path, 0o755)
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except Exception as e:
        print(f"Fatal Error: Could not initialize driver: {e}")
        return None

# --- CRAWLING FUNCTIONS ---
def get_tab_content(driver, url):
    """Navigates to a specific tab URL and extracts the main text content."""
    print(f"  Crawling tab: {url}...")
    driver.get(url)
    try:
        # Wait for the main content area to load
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "site-content")))
        time.sleep(3)
        return driver.find_element(By.ID, "site-content").text
    except Exception as e:
        print(f"    -> Warning: Could not extract content from {url}: {e}")
        return f"Error extracting content from {url}."

def get_top_notebook_links(driver, base_url):
    """Navigates to the Code tab, sorts by 'Most Votes', and returns top 10 links."""
    code_url = f"{base_url}/code"
    print(f"  Getting notebook links from: {code_url}...")
    driver.get(code_url)
    
    links = []
    try:
        wait = WebDriverWait(driver, 30)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(5)

        # Sort by 'Most Votes'
        print("    -> Sorting by 'Most Votes'...")
        try:
            wait.until(EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Hotness')]"))).click()
            time.sleep(1)
            wait.until(EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Most Votes')]"))).click()
            time.sleep(5)
        except Exception as e:
             print(f"    -> Warning: Sorting failed (UI might differ), proceeding with default order. {e}")
        
        # Extract links
        elements = driver.find_elements(By.TAG_NAME, "a")
        for elem in elements:
            href = elem.get_attribute("href")
            if href and "/code/" in href and "competitions" not in href and "/new" not in href:
                # FIX: Remove /comments suffix to get the notebook URL
                if href.endswith("/comments"):
                    href = href.replace("/comments", "")
                
                if href not in links:
                    links.append(href)
        
        print(f"    -> Found {len(links)} links, taking top 10.")
        return links[:10]
    except Exception as e:
        print(f"    -> Error getting notebook links: {e}")
        return []

def get_single_notebook_content(driver, url):
    """Crawls the full content of a single notebook page, handling the iframe."""
    print(f"    Crawling notebook: {url}")
    driver.get(url)
    iframe_content = ""
    try:
        # Wait for and switch to the notebook iframe
        iframe = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "rendered-kernel-content")))
        driver.switch_to.frame(iframe)
        time.sleep(3)

        # Scroll inside iframe to load all cells
        last_height = driver.execute_script("return document.body.scrollHeight")
        for _ in range(20):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        iframe_content = driver.find_element(By.TAG_NAME, "body").text
    except Exception as e:
        iframe_content = f"Error extracting notebook content from iframe: {e}"
    finally:
        # Always switch back to the main page context
        driver.switch_to.default_content()
    
    return iframe_content

# --- HELPER FUNCTIONS ---
def slugify(url):
    """Creates a clean filename or directory name from a URL."""
    if url.endswith('/'):
        url = url[:-1]
    return url.split('/')[-1].replace('?','-').replace('=','-')

# --- MAIN WORKFLOW ---
def main():
    """Main workflow to crawl competitions and their notebooks."""
    driver = setup_driver()
    if not driver:
        return

    try:
        for base_url in URLS_TO_CRAWL:
            print(f"\n--- Starting Competition: {base_url} ---")
            
            # 1. Setup directories
            comp_slug = slugify(base_url)
            os.makedirs(comp_slug, exist_ok=True)
            notebooks_dir = os.path.join(comp_slug, "notebooks")
            os.makedirs(notebooks_dir, exist_ok=True)
            
            # 2. Crawl main competition tabs
            results = {}
            tabs_to_crawl = ["Overview", "Data", "Rules"]
            for tab in tabs_to_crawl:
                results[tab] = get_tab_content(driver, f"{base_url}/{tab.lower()}")
            
            # 3. Get notebook links from the "Code" tab
            top_links = get_top_notebook_links(driver, base_url)
            results["Code"] = "Top 10 Notebook Links:\n" + "\n".join(top_links)
            
            # 4. Save main competition file
            main_filename = os.path.join(comp_slug, f"{comp_slug}.txt")
            with open(main_filename, "w", encoding="utf-8") as f:
                for tab, content in results.items():
                    f.write(f"=== {tab.upper()} ===\n")
                    f.write(content + "\n\n")
            print(f"\nSaved main competition data to: {main_filename}")

            # 5. Crawl each notebook from the links found
            print(f"\n--- Crawling {len(top_links)} notebooks for {comp_slug} ---")
            for link in top_links:
                notebook_slug = slugify(link)
                notebook_filename = os.path.join(notebooks_dir, f"{notebook_slug}.txt")
                
                # Get content
                notebook_content = get_single_notebook_content(driver, link)
                
                # Save content
                with open(notebook_filename, "w", encoding="utf-8") as f:
                    f.write(notebook_content)
                print(f"      -> Saved to {notebook_filename}")

    finally:
        print("\nWorkflow finished. Closing driver.")
        driver.quit()

if __name__ == "__main__":
    main()
