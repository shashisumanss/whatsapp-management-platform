from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import base64
import time
import threading
import io
from PIL import Image

# Global variables for WhatsApp Web session
whatsapp_driver = None
current_qr_code = None
whatsapp_connected = False
qr_refresh_needed = False

def initialize_whatsapp_web():
    global whatsapp_driver, current_qr_code, whatsapp_connected, qr_refresh_needed
    
    try:
        print("Starting WhatsApp Web initialization...")
        
        # Chrome options for server environment
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in background
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--remote-debugging-port=9222')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--allow-running-insecure-content')
        
        # Initialize Chrome driver with automatic driver management
        service = Service(ChromeDriverManager().install())
        whatsapp_driver = webdriver.Chrome(service=service, options=chrome_options)
        
        print("Navigating to WhatsApp Web...")
        # Navigate to WhatsApp Web
        whatsapp_driver.get('https://web.whatsapp.com')
        
        # Wait for page to load
        time.sleep(5)
        
        # Wait for QR code to appear and capture it
        capture_qr_code()
        
        # Monitor for connection
        def monitor_connection():
            global whatsapp_connected
            try:
                print("Monitoring for WhatsApp connection...")
                # Wait for successful login (chat list appears)
                WebDriverWait(whatsapp_driver, 120).until(
                    EC.any_of(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='chat-list']")),
                        EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='side']"))
                    )
                )
                whatsapp_connected = True
                current_qr_code = None  # Clear QR code after connection
                print("WhatsApp Web connected successfully!")
            except Exception as e:
                print(f"WhatsApp Web connection timeout or error: {e}")
        
        # Start connection monitoring in background
        threading.Thread(target=monitor_connection, daemon=True).start()
        
    except Exception as e:
        print(f"Error initializing WhatsApp Web: {e}")
        if whatsapp_driver:
            whatsapp_driver.quit()
            whatsapp_driver = None

def capture_qr_code():
    global current_qr_code, whatsapp_driver
    
    try:
        print("Looking for QR code...")
        
        # Multiple selectors to try for QR code
        qr_selectors = [
            "canvas[aria-label='Scan me!']",
            "canvas[aria-label*='Scan']",
            "div[data-testid='qr-code'] canvas",
            "canvas",
            "[data-ref] canvas"
        ]
        
        qr_element = None
        for selector in qr_selectors:
            try:
                qr_element = WebDriverWait(whatsapp_driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                print(f"Found QR code with selector: {selector}")
                break
            except:
                continue
        
        if qr_element:
            # Take screenshot of QR code element
            qr_screenshot = qr_element.screenshot_as_png
            current_qr_code = base64.b64encode(qr_screenshot).decode()
            print("QR code captured successfully!")
            
            # Start QR refresh timer (WhatsApp QR codes expire)
            def refresh_qr():
                time.sleep(15)  # Refresh every 15 seconds
                if not whatsapp_connected and whatsapp_driver:
                    capture_qr_code()
            
            threading.Thread(target=refresh_qr, daemon=True).start()
        else:
            print("QR code element not found")
            
    except Exception as e:
        print(f"Error capturing QR code: {e}")

def get_whatsapp_status():
    global whatsapp_driver, whatsapp_connected, current_qr_code
    
    if whatsapp_connected:
        return {
            'success': True,
            'connected': True,
            'message': 'WhatsApp Web is connected',
            'qr_code': None
        }
    elif current_qr_code:
        return {
            'success': True,
            'connected': False,
            'qr_code': current_qr_code,
            'message': 'QR code ready for scanning'
        }
    elif whatsapp_driver:
        return {
            'success': True,
            'connected': False,
            'qr_code': None,
            'message': 'Loading WhatsApp Web...'
        }
    else:
        return {
            'success': False,
            'connected': False,
            'qr_code': None,
            'message': 'WhatsApp Web not initialized'
        }

def stop_whatsapp_web():
    global whatsapp_driver, current_qr_code, whatsapp_connected
    
    try:
        if whatsapp_driver:
            whatsapp_driver.quit()
            whatsapp_driver = None
        current_qr_code = None
        whatsapp_connected = False
        print("WhatsApp Web session stopped")
        return True
    except Exception as e:
        print(f"Error stopping WhatsApp Web: {e}")
        return False
