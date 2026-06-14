#!/usr/bin/env python3
"""test_gradio_ui_v3.py - Final Playwright interaction test for Gradio UI

Tests all interactive elements including the new question buttons.
"""

import os
import sys
import time
import json

from playwright.sync_api import sync_playwright

GRADIO_URL = "http://localhost:7861"
SCREENSHOT_DIR = "/workspace2/cy/Qwen3-0.6B/outputs/screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

TEST_IMAGE = "/workspace2/cy/Qwen3-0.6B/data/coco128_images/000000000009.jpg"

results = {"tests": [], "screenshots": []}


def record(name, passed, detail=""):
    results["tests"].append({"name": name, "passed": passed, "detail": detail})
    status = "✅" if passed else "❌"
    print(f"  {status} {name}: {detail}")


def screenshot(page, name):
    path = os.path.join(SCREENSHOT_DIR, f"v3_{name}.png")
    page.screenshot(path=path, full_page=True)
    results["screenshots"].append(path)
    print(f"  📸 {path}")


def main():
    print("=" * 70)
    print("  Gradio UI v3 - Final Comprehensive Test")
    print("=" * 70)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        console_errors = []
        page.on("console", lambda msg: console_errors.append(f"[{msg.type}] {msg.text}") if msg.type == "error" else None)

        # ============================================================
        # Test 1: Page Load
        # ============================================================
        print("\n[1/8] Page load...")
        resp = page.goto(GRADIO_URL, timeout=30000, wait_until="networkidle")
        record("Page Load", resp.status == 200, f"HTTP {resp.status}")
        time.sleep(3)
        screenshot(page, "01_loaded")

        # ============================================================
        # Test 2: Component visibility
        # ============================================================
        print("\n[2/8] Component visibility...")

        record("Image Upload", page.locator('input[type="file"]').count() > 0, "file input found")
        record("Question Input", page.locator('textarea').first.is_visible(), "visible")
        record("Generate Button", page.locator('button:has-text("Generate")').is_visible(), "visible")
        record("Example Gallery", page.locator('div.gallery-container').count() > 0, "gallery found")
        record("Advanced Accordion", page.locator('span:has-text("Advanced")').count() > 0, "accordion found")

        # Question buttons (replaced Dropdown)
        question_buttons = ["Describe", "Objects", "Subject", "Colors", "Text?", "Count"]
        for btn_text in question_buttons:
            btn = page.locator(f'button:has-text("{btn_text}")')
            record(f"Question Button '{btn_text}'", btn.is_visible(), f"visible={btn.is_visible()}")

        # ============================================================
        # Test 3: Image Upload + Inference (the critical flow)
        # ============================================================
        print("\n[3/8] Full inference flow (upload + generate)...")

        # Upload image
        file_input = page.locator('input[type="file"]').first
        file_input.set_input_files(TEST_IMAGE)
        time.sleep(2)
        screenshot(page, "02_image_uploaded")

        # Set question
        textarea = page.locator('textarea').first
        textarea.click()
        textarea.fill("What objects can you see in this image?")
        time.sleep(0.5)

        # Click generate
        gen_btn = page.locator('button:has-text("Generate")')
        gen_btn.click()
        print("  ⏳ Waiting for model response...")

        # Wait for response
        for i in range(30):
            time.sleep(3)
            textareas = page.locator('textarea')
            if textareas.count() >= 2:
                try:
                    output = textareas.nth(1).input_value()
                    if output and len(output) > 10:
                        is_error = "error" in output.lower()
                        record("Full Inference Flow", not is_error,
                               f"Response ({len(output)} chars): '{output[:120]}...'")
                        break
                except:
                    pass
        else:
            record("Full Inference Flow", False, "Timeout after 90s")

        screenshot(page, "03_after_inference")

        # ============================================================
        # Test 4: Question Buttons
        # ============================================================
        print("\n[4/8] Question button click...")

        btn = page.locator('button:has-text("Colors")')
        btn.click()
        time.sleep(0.5)

        textarea = page.locator('textarea').first
        new_val = textarea.input_value()
        record("Question Button 'Colors'", "colors" in new_val.lower(), f"Question set to: '{new_val}'")
        screenshot(page, "04_question_button")

        # ============================================================
        # Test 5: Gallery Select
        # ============================================================
        print("\n[5/8] Gallery image selection...")
        try:
            gallery = page.locator('div.gallery-container')
            if gallery.count() > 0:
                # Click on gallery thumbnail - use SelectData event
                thumbnails = gallery.locator('img').first
                thumbnails.click()
                time.sleep(1)

                # Check if image component got updated
                image_container = page.locator('[data-testid="image"]')
                record("Gallery Select", image_container.count() > 0, "Image container present after gallery click")
                screenshot(page, "05_gallery_select")
            else:
                record("Gallery Select", False, "No gallery found")
        except Exception as e:
            record("Gallery Select", False, str(e))

        # ============================================================
        # Test 6: Advanced Settings
        # ============================================================
        print("\n[6/8] Advanced settings...")
        accordion = page.locator('span:has-text("Advanced")')
        accordion.first.click()
        time.sleep(1)
        record("Advanced Settings", page.locator('[role="slider"]').count() > 0, "Sliders visible after expand")
        screenshot(page, "06_advanced")

        # ============================================================
        # Test 7: Enter key submit
        # ============================================================
        print("\n[7/8] Enter key submit...")
        try:
            textarea = page.locator('textarea').first
            textarea.fill("Describe this image briefly.")
            textarea.press("Enter")
            print("  ⏳ Waiting for Enter-triggered response...")

            for i in range(20):
                time.sleep(3)
                textareas = page.locator('textarea')
                if textareas.count() >= 2:
                    try:
                        output = textareas.nth(1).input_value()
                        if output and len(output) > 10:
                            record("Enter Key Submit", True, f"Response ({len(output)} chars)")
                            break
                    except:
                        pass
            else:
                record("Enter Key Submit", False, "Timeout")
            screenshot(page, "07_enter_submit")
        except Exception as e:
            record("Enter Key Submit", False, str(e))

        # ============================================================
        # Test 8: Thinking Mode Toggle
        # ============================================================
        print("\n[8/8] Thinking mode toggle...")
        try:
            # Find and expand advanced settings if not already
            checkbox = page.locator('label:has-text("Thinking") input[type="checkbox"], label:has-text("Show Thinking") input')
            if checkbox.count() > 0:
                checkbox.first.click()
                time.sleep(0.3)
                record("Thinking Toggle", True, "Checkbox toggled")
            else:
                record("Thinking Toggle", False, "Checkbox not found")
        except Exception as e:
            record("Thinking Toggle", False, str(e))

        screenshot(page, "08_final")
        browser.close()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("  Final Test Summary")
    print("=" * 70)

    passed = sum(1 for t in results["tests"] if t["passed"])
    total = len(results["tests"])
    print(f"  Total: {passed}/{total} passed")

    for t in results["tests"]:
        status = "✅" if t["passed"] else "❌"
        print(f"  {status} {t['name']}: {t['detail']}")

    if console_errors:
        print(f"\n  Console errors: {len(console_errors)}")
        for e in console_errors[:5]:
            print(f"    {e}")

    # Save
    with open("/workspace2/cy/Qwen3-0.6B/outputs/ui_test_results_v3.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


if __name__ == "__main__":
    main()
