"""
Sleepy Dull Stories - ENHANCED DEBUG Midjourney Visual Generator
🔍 FULL DEBUG MODE: Every request/response logged in detail
"""

import requests
import os
import json
import pandas as pd
import time
import sys
import sqlite3
import signal
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple
import urllib.request
from pathlib import Path
import logging

# Load environment first
load_dotenv()

class EnhancedDebugMidjourneyGenerator:
    """Enhanced debug version with full API request/response logging"""

    def __init__(self):
        # ... (keep existing initialization) ...
        self.debug_log = []
        self.debug_mode = True

        # API setup
        self.api_key = os.getenv('PIAPI_KEY')
        self.base_url = "https://api.piapi.ai/api/v1"

        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        print(f"🔍 DEBUG MODE ENABLED")
        print(f"🔑 API Key: {self.api_key[:8] if self.api_key else 'NOT FOUND'}...")
        print(f"🌐 Base URL: {self.base_url}")

    def debug_log_api_call(self, method: str, url: str, headers: dict, payload: dict = None,
                           response: requests.Response = None):
        """Log detailed API call information"""
        timestamp = datetime.now().isoformat()

        print(f"\n🔍 DEBUG API CALL [{timestamp}]")
        print(f"📡 Method: {method}")
        print(f"🌐 URL: {url}")
        print(
            f"📝 Headers: {json.dumps({k: v[:20] + '...' if k == 'x-api-key' and len(v) > 20 else v for k, v in headers.items()}, indent=2)}")

        if payload:
            print(f"📦 Request Payload:")
            print(json.dumps(payload, indent=2))

        if response:
            print(f"📊 Response Status: {response.status_code}")
            print(f"📋 Response Headers: {json.dumps(dict(response.headers), indent=2)}")
            print(f"📄 Response Body:")
            try:
                response_json = response.json()
                print(json.dumps(response_json, indent=2))
            except:
                print(f"Raw text: {response.text}")

        print("🔍" + "=" * 80)

    def test_api_connection_debug(self) -> bool:
        """Enhanced debug version of API connection test"""
        print(f"\n🔍 TESTING PIAPI CONNECTION - FULL DEBUG")
        print(f"🔑 API Key: {self.api_key}")
        print(f"🌐 Base URL: {self.base_url}")

        test_prompt = "red apple on white table --ar 1:1 --v 6.1"

        payload = {
            "model": "midjourney",
            "task_type": "imagine",
            "input": {
                "prompt": test_prompt,
                "aspect_ratio": "1:1",
                "process_mode": "relax"
            }
        }

        url = f"{self.base_url}/task"

        print(f"\n🎯 PREPARING REQUEST:")
        print(f"URL: {url}")
        print(f"Headers: {json.dumps(self.headers, indent=2)}")
        print(f"Payload: {json.dumps(payload, indent=2)}")

        try:
            print(f"\n🚀 SENDING REQUEST...")
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=10
            )
            # Log the full API call
            self.debug_log_api_call("POST", f"{self.base_url}/task", self.headers, payload, response)

            print(f"\n📊 RESPONSE ANALYSIS:")
            print(f"Status Code: {response.status_code}")
            print(f"Content Type: {response.headers.get('content-type', 'unknown')}")
            print(f"Content Length: {response.headers.get('content-length', 'unknown')}")

            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"✅ JSON Response Parsed Successfully")
                    print(f"Response Code: {result.get('code', 'missing')}")
                    print(f"Response Message: {result.get('message', 'missing')}")

                    if result.get("code") == 200:
                        print(f"✅ API CONNECTION TEST SUCCESSFUL")
                        task_data = result.get("data", {})
                        print(f"Task ID received: {task_data.get('task_id', 'missing')}")
                        return True
                    else:
                        print(f"❌ API returned error code: {result.get('code')}")
                        print(f"Error details: {result}")
                        return False

                except json.JSONDecodeError as e:
                    print(f"❌ JSON DECODE ERROR: {e}")
                    print(f"Raw response: {response.text}")
                    return False
            else:
                print(f"❌ HTTP ERROR: {response.status_code}")
                print(f"Response text: {response.text}")
                return False

        except requests.exceptions.Timeout:
            print(f"❌ REQUEST TIMEOUT (10 seconds)")
            return False
        except requests.exceptions.ConnectionError as e:
            print(f"❌ CONNECTION ERROR: {e}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"❌ REQUEST EXCEPTION: {e}")
            return False
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

    def submit_midjourney_task_debug(self, prompt: str, aspect_ratio: str = "16:9") -> Optional[str]:
        """Enhanced debug version of task submission"""
        print(f"\n🔍 SUBMITTING MIDJOURNEY TASK - FULL DEBUG")
        print(f"📝 Prompt: {prompt}")
        print(f"📐 Aspect Ratio: {aspect_ratio}")

        payload = {
            "model": "midjourney",
            "task_type": "imagine",
            "input": {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "process_mode": "relax",  # Could be "fast", "turbo", "relax"
                "skip_prompt_check": False
            }
        }

        url = f"{self.base_url}/task"

        print(f"\n🎯 REQUEST DETAILS:")
        print(f"URL: {url}")
        print(f"Method: POST")
        print(f"Headers: {json.dumps(self.headers, indent=2)}")
        print(f"Payload: {json.dumps(payload, indent=2)}")

        try:
            print(f"\n🚀 SENDING REQUEST...")
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=30
            )

            # Log the full API call
            self.debug_log_api_call("POST", url, self.headers, payload, response)

            print(f"\n📊 RESPONSE ANALYSIS:")
            print(f"Status Code: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")

            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"✅ JSON Response Parsed")
                    print(f"Full Response: {json.dumps(result, indent=2)}")

                    response_code = result.get("code")
                    response_message = result.get("message", "")

                    print(f"Response Code: {response_code}")
                    print(f"Response Message: {response_message}")

                    if response_code == 200:
                        task_data = result.get("data", {})
                        task_id = task_data.get("task_id")

                        print(f"✅ TASK SUBMITTED SUCCESSFULLY")
                        print(f"Task ID: {task_id}")
                        print(f"Task Status: {task_data.get('status', 'unknown')}")
                        print(f"Task Type: {task_data.get('task_type', 'unknown')}")
                        print(f"Model: {task_data.get('model', 'unknown')}")

                        return task_id
                    else:
                        print(f"❌ API ERROR RESPONSE")
                        print(f"Error Code: {response_code}")
                        print(f"Error Message: {response_message}")
                        print(f"Full Error Data: {result}")
                        return None

                except json.JSONDecodeError as e:
                    print(f"❌ JSON DECODE ERROR: {e}")
                    print(f"Raw Response Text: {response.text}")
                    return None

            elif response.status_code == 400:
                print(f"❌ BAD REQUEST (400)")
                print(f"Response: {response.text}")
                try:
                    error_data = response.json()
                    print(f"Error Details: {json.dumps(error_data, indent=2)}")
                except:
                    pass
                return None

            elif response.status_code == 401:
                print(f"❌ UNAUTHORIZED (401) - Check API Key")
                print(f"API Key used: {self.api_key[:8]}..." if self.api_key else "NO API KEY")
                print(f"Response: {response.text}")
                return None

            elif response.status_code == 500:
                print(f"❌ SERVER ERROR (500) - PiAPI Internal Error")
                print(f"Response: {response.text}")
                try:
                    error_data = response.json()
                    print(f"Server Error Details: {json.dumps(error_data, indent=2)}")
                except:
                    pass
                return None

            else:
                print(f"❌ UNEXPECTED HTTP STATUS: {response.status_code}")
                print(f"Response: {response.text}")
                return None

        except requests.exceptions.Timeout:
            print(f"❌ REQUEST TIMEOUT (30 seconds)")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"❌ CONNECTION ERROR: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ REQUEST EXCEPTION: {e}")
            return None
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None

    def check_task_status_debug(self, task_id: str) -> Optional[Dict]:
        """Enhanced debug version of task status checking"""
        print(f"\n🔍 CHECKING TASK STATUS - FULL DEBUG")
        print(f"🆔 Task ID: {task_id}")

        url = f"{self.base_url}/task/{task_id}"

        print(f"\n🎯 REQUEST DETAILS:")
        print(f"URL: {url}")
        print(f"Method: GET")
        print(f"Headers: {json.dumps(self.headers, indent=2)}")

        try:
            print(f"\n🚀 SENDING STATUS REQUEST...")
            response = requests.get(url, headers=self.headers, timeout=10)

            # Log the full API call
            self.debug_log_api_call("GET", url, self.headers, None, response)

            print(f"\n📊 STATUS RESPONSE ANALYSIS:")
            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"✅ JSON Response Parsed")
                    print(f"Full Response: {json.dumps(result, indent=2)}")

                    if result.get("code") == 200:
                        task_data = result.get("data", {})
                        status = task_data.get("status", "").lower()
                        output = task_data.get("output", {})

                        print(f"📊 TASK STATUS DETAILS:")
                        print(f"Status: {status}")
                        print(f"Progress: {output.get('progress', 'unknown')}%")
                        print(f"Model: {task_data.get('model', 'unknown')}")
                        print(f"Task Type: {task_data.get('task_type', 'unknown')}")

                        if status == "completed":
                            print(f"\n✅ TASK COMPLETED - ANALYZING OUTPUT")

                            # Analyze all available image URLs
                            temp_urls = output.get("temporary_image_urls", [])
                            image_url = output.get("image_url", "")
                            image_urls = output.get("image_urls", [])
                            discord_url = output.get("discord_image_url", "")

                            print(f"🖼️ IMAGE URLS ANALYSIS:")
                            print(f"temporary_image_urls: {len(temp_urls)} URLs")
                            for i, url in enumerate(temp_urls):
                                print(f"  [{i}]: {url}")

                            print(f"image_url: {image_url}")
                            print(f"image_urls: {len(image_urls)} URLs")
                            for i, url in enumerate(image_urls):
                                print(f"  [{i}]: {url}")
                            print(f"discord_image_url: {discord_url}")

                            # Select best URL
                            if temp_urls and len(temp_urls) > 0:
                                selected_url = temp_urls[1] if len(temp_urls) >= 2 else temp_urls[0]
                                print(f"✅ SELECTED URL: {selected_url} (from temporary_image_urls)")
                                return {"url": selected_url, "source": "temporary_image_urls"}
                            elif image_url:
                                print(f"✅ SELECTED URL: {image_url} (from image_url)")
                                return {"url": image_url, "source": "image_url"}
                            elif image_urls and len(image_urls) > 0:
                                selected_url = image_urls[0]
                                print(f"✅ SELECTED URL: {selected_url} (from image_urls)")
                                return {"url": selected_url, "source": "image_urls"}
                            else:
                                print(f"❌ NO VALID IMAGE URLS FOUND")
                                return False

                        elif status == "failed":
                            print(f"❌ TASK FAILED")
                            error_info = task_data.get("error", {})
                            print(f"Error Info: {json.dumps(error_info, indent=2)}")
                            return False

                        elif status in ["pending", "processing"]:
                            print(f"⏳ TASK STILL {status.upper()}")
                            return None

                        else:
                            print(f"❓ UNKNOWN STATUS: {status}")
                            return None

                    else:
                        print(f"❌ API ERROR RESPONSE")
                        print(f"Error Code: {result.get('code')}")
                        print(f"Error Message: {result.get('message', '')}")
                        return False

                except json.JSONDecodeError as e:
                    print(f"❌ JSON DECODE ERROR: {e}")
                    print(f"Raw Response: {response.text}")
                    return None

            else:
                print(f"❌ HTTP ERROR: {response.status_code}")
                print(f"Response: {response.text}")
                return None

        except Exception as e:
            print(f"❌ STATUS CHECK ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_debug_log(self, output_dir: str):
        """Save all debug information to file"""
        debug_file = Path(output_dir) / "api_debug_log.json"

        debug_report = {
            "debug_session_start": datetime.now().isoformat(),
            "total_api_calls": len(self.debug_log),
            "api_calls": self.debug_log
        }

        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(debug_report, f, indent=2, ensure_ascii=False)

        print(f"🔍 Debug log saved: {debug_file}")

    def run_debug_test(self):
        """Run a simple debug test"""
        print("🔍" * 60)
        print("ENHANCED DEBUG TEST MODE")
        print("🔍" * 60)

        # Test 1: API Connection
        print(f"\n🧪 TEST 1: API CONNECTION")
        connection_ok = self.test_api_connection_debug()

        if not connection_ok:
            print(f"❌ API connection failed - stopping debug test")
            return False

        # Test 2: Simple Task Submission
        print(f"\n🧪 TEST 2: SIMPLE TASK SUBMISSION")
        simple_prompt = "a cute cat sitting on a table --ar 1:1 --v 6.1"
        task_id = self.submit_midjourney_task_debug(simple_prompt, "1:1")

        if not task_id:
            print(f"❌ Task submission failed")
            return False

        # Test 3: Status Monitoring
        print(f"\n🧪 TEST 3: STATUS MONITORING (5 cycles)")
        for cycle in range(5):
            print(f"\n📊 Status Check Cycle {cycle + 1}/5")
            result = self.check_task_status_debug(task_id)

            if result and isinstance(result, dict):
                print(f"✅ Task completed! URL: {result['url']}")
                break
            elif result is False:
                print(f"❌ Task failed")
                break
            else:
                print(f"⏳ Still processing... waiting 30 seconds")
                time.sleep(30)

        # Save debug log
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)
        self.save_debug_log(str(debug_dir))

        print(f"\n🔍 DEBUG TEST COMPLETED")
        return True


# Quick test function
def run_debug_test():
    """Run debug test"""
    try:
        generator = EnhancedDebugMidjourneyGenerator()
        generator.run_debug_test()
    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--debug':
        run_debug_test()
    else:
        print("🔍 Enhanced Debug Visual Generator")
        print("Usage: python script.py --debug")
        print("This will run comprehensive API debugging")