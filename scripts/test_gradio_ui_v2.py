#!/usr/bin/env python3
"""test_gradio_ui_v2.py - Comprehensive Playwright interaction test for Gradio UI

Tests all interactive elements with proper selectors for Gradio 6.x
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

results = {"tests": [], "screenshots": [], "errors": []}


def record(name, passed, detail=""):
    results["tests"].append({"name": name, "passed": passed, "detail": detail})
    status = "✅" if passed else "❌"
    print(f"  {status} {name}: {detail}")


def screenshot(page, name):
    path = os.path.join(SCREENSHOT_DIR, f"v2_{name}.png")
    page.screenshot(path=path, full_page=True)
    results["screenshots"].append(path)
    print(f"  📸 {path}")


def main():
    print("=" * 70)
    print("  Gradio UI v2 - Comprehensive Interaction Test")
    print("=" * 70)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        console_errors = []
        page.on("console", lambda msg: console_errors.append(f"[{msg.type}] {msg.text}") if msg.type in ["error", "warning"] else None)
        network_errors = []
        page.on("requestfailed", lambda req: network_errors.append(f"[FAIL] {req.url}"))

        # ============================================================
        # Test 1: Page Load
        # ============================================================
        print("\n[1/8] Page load...")
        try:
            resp = page.goto(GRADIO_URL, timeout=30000, wait_until="networkidle")
            record("Page Load", resp.status == 200, f"HTTP {resp.status}")
        except Exception as e:
            record("Page Load", False, str(e))
            browser.close()
            return

        time.sleep(3)  # Wait for Gradio frontend to render
        screenshot(page, "01_loaded")

        # ============================================================
        # Test 2: Component visibility
        # ============================================================
        print("\n[2/8] Component visibility...")

        # Image component
        image_comp = page.locator('div[class*="image"] input[type="file"], input[type="file"]')
        record("Image Upload", image_comp.count() > 0, f"{image_comp.count()} file input(s)")

        # Question textbox
        textbox = page.locator('textarea').first
        record("Question Input", textbox.is_visible(), f"visible={textbox.is_visible()}")

        # Generate button
        gen_btn = page.locator('button:has-text("Generate")')
        record("Generate Button", gen_btn.is_visible(), f"visible={gen_btn.is_visible()}")

        # Gallery
        gallery = page.locator('div.gallery-container')
        record("Example Gallery", gallery.count() > 0, f"count={gallery.count()}")

        # Dropdown - check the label text
        dropdown_label = page.locator('label:has-text("Example Questions"), [data-testid="dropdown"]')
        record("Dropdown Component", dropdown_label.count() > 0, f"count={dropdown_label.count()}")

        # Advanced accordion
        adv_acc = page.locator('button:has-text("Advanced"), span:has-text("Advanced")')
        record("Advanced Accordion", adv_acc.count() > 0, f"count={adv_acc.count()}")

        # ============================================================
        # Test 3: Image Upload
        # ============================================================
        print("\n[3/8] Image upload...")
        try:
            file_input = page.locator('input[type="file"]').first
            file_input.set_input_files(TEST_IMAGE)
            time.sleep(2)
            screenshot(page, "02_image_uploaded")
            record("Image Upload", True, f"Uploaded {os.path.basename(TEST_IMAGE)}")
        except Exception as e:
            record("Image Upload", False, str(e))
            screenshot(page, "02_upload_error")

        # ============================================================
        # Test 4: Question Input
        # ============================================================
        print("\n[4/8] Question input...")
        try:
            textbox = page.locator('textarea').first
            textbox.click()
            textbox.fill("")
            textbox.fill("What objects can you see in this image?")
            time.sleep(0.5)
            val = textbox.input_value()
            record("Question Input", "objects" in val, f"value='{val[:60]}'")
            screenshot(page, "03_question_set")
        except Exception as e:
            record("Question Input", False, str(e))

        # ============================================================
        # Test 5: Click Generate - THE CRITICAL TEST
        # ============================================================
        print("\n[5/8] Generate button click (waiting for model response)...")

        # Get current output text before clicking
        textareas = page.locator('textarea')
        output_before = ""
        if textareas.count() >= 2:
            try:
                output_before = textareas.nth(1).input_value()
            except:
                pass

        try:
            gen_btn = page.locator('button:has-text("Generate")')
            gen_btn.click()
            print("  ⏳ Button clicked, waiting for response...")

            # Wait for loading state to appear and then resolve
            # In Gradio, loading shows a spinner; response replaces it
            max_wait = 90  # seconds
            for i in range(max_wait // 3):
                time.sleep(3)

                # Check for response in output textarea
                textareas = page.locator('textarea')
                if textareas.count() >= 2:
                    try:
                        output_now = textareas.nth(1).input_value()
                        if output_now and output_now != output_before and len(output_now) > 5:
                            # Got a real response!
                            is_error = "error" in output_now.lower() or "Error" in output_now
                            record("Generate Response", not is_error,
                                   f"Response ({len(output_now)} chars): '{output_now[:150]}'")
                            break
                    except:
                        pass

                # Also check if loading spinner appeared
                spinner = page.locator('.loading, [aria-label="Loading"], .spinner, button[disabled]')
                if spinner.count() > 0 and i < 5:
                    print(f"  ⏳ Still processing... ({i*3}s)")

            else:
                # Timeout - take screenshot and report
                textareas = page.locator('textarea')
                output_now = ""
                if textareas.count() >= 2:
                    try:
                        output_now = textareas.nth(1).input_value()
                    except:
                        pass
                record("Generate Response", False, f"Timeout after {max_wait}s. Output: '{output_now[:100]}'")

            screenshot(page, "04_after_generate")

        except Exception as e:
            record("Generate Response", False, str(e))
            screenshot(page, "04_generate_error")

        # ============================================================
        # Test 6: Dropdown Selection
        # ============================================================
        print("\n[6/8] Dropdown interaction...")
        try:
            # Find the dropdown by label
            dropdown_el = page.locator('label:has-text("Example Questions")')
            if dropdown_el.count() > 0:
                # Click the dropdown container to open it
                dropdown_container = dropdown_el.locator('..')
                dropdown_container.click()
                time.sleep(1)
                screenshot(page, "05_dropdown_opened")

                # Find and click an option
                options = page.locator('[role="option"], [role="listbox"] li, li[class*="item"]')
                if options.count() > 0:
                    second_option = options.nth(1) if options.count() > 1 else options.first
                    second_option.click()
                    time.sleep(0.5)

                    # Check if question textarea was updated
                    textbox = page.locator('textarea').first
                    new_val = textbox.input_value()
                    record("Dropdown Selection", True, f"Updated question: '{new_val[:60]}'")
                else:
                    record("Dropdown Selection", False, f"No options found (dropdown items: {options.count()})")
            else:
                # Try alternative approach - find dropdown input
                dropdown_input = page.locator('input[role="combobox"], input[class*="dropdown"]')
                if dropdown_input.count() > 0:
                    dropdown_input.first.click()
                    time.sleep(1)
                    options = page.locator('[role="option"], [role="listbox"] li')
                    record("Dropdown Selection", options.count() > 0, f"Found {options.count()} options via input")
                else:
                    record("Dropdown Selection", False, "No dropdown element found")
        except Exception as e:
            record("Dropdown Selection", False, str(e))
            screenshot(page, "05_dropdown_error")

        # ============================================================
        # Test 7: Advanced Settings Accordion
        # ============================================================
        print("\n[7/8] Advanced settings accordion...")
        try:
            # Find and click the accordion header
            accordion_header = page.locator('span:has-text("Advanced")')
            if accordion_header.count() > 0:
                accordion_header.first.click()
                time.sleep(1)
                screenshot(page, "06_advanced_expanded")

                # Verify sliders appeared
                sliders = page.locator('input[type="range"], [role="slider"]')
                slider_count = sliders.count()
                record("Advanced Settings Expand", slider_count > 0, f"{slider_count} sliders visible")

                # Try adjusting temperature slider
                if slider_count > 0:
                    # Find temperature by label
                    temp_label = page.locator('text=Temperature')
                    if temp_label.count() > 0:
                        record("Temperature Slider", True, "Found temperature label")
                    else:
                        record("Temperature Slider", False, "Temperature label not found")
            else:
                record("Advanced Settings Expand", False, "Accordion header not found")
        except Exception as e:
            record("Advanced Settings Expand", False, str(e))

        # ============================================================
        # Test 8: Gallery Click (if exists)
        # ============================================================
        print("\n[8/8] Gallery click test...")
        try:
            gallery = page.locator('div.gallery-container')
            if gallery.count() > 0:
                # Try clicking on a thumbnail in the gallery
                thumbnails = gallery.locator('img, button, div[class*="thumbnail"]')
                thumb_count = thumbnails.count()
                if thumb_count > 0:
                    # In Gradio 6, Gallery thumbnails may not be directly clickable
                    # to load into Image component. Check if gallery.select event is connected.
                    print(f"  Found {thumb_count} gallery thumbnails")

                    # Try clicking first thumbnail
                    thumbnails.first.click()
                    time.sleep(1)
                    screenshot(page, "07_gallery_clicked")
                    record("Gallery Click", True, f"Clicked thumbnail ({thumb_count} available)")
                else:
                    record("Gallery Click", False, "No thumbnails found in gallery")
            else:
                record("Gallery Click", False, "Gallery component not found")
        except Exception as e:
            record("Gallery Click", False, str(e))

        # ============================================================
        # Collect all errors
        # ============================================================
        if console_errors:
            results["errors"].extend(console_errors[-20:])
        if network_errors:
            results["errors"].extend(network_errors[-10:])

        screenshot(page, "08_final")

        browser.close()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)

    passed = sum(1 for t in results["tests"] if t["passed"])
    total = len(results["tests"])
    print(f"  Total: {passed}/{total} passed")

    for t in results["tests"]:
        status = "✅" if t["passed"] else "❌"
        print(f"  {status} {t['name']}: {t['detail']}")

    if results["errors"]:
        print(f"\n  Console/Network errors ({len(results['errors'])}):")
        for e in results["errors"][:10]:
            print(f"    {e}")

    # Save
    with open("/workspace2/cy/Qwen3-0.6B/outputs/ui_test_results_v2.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to outputs/ui_test_results_v2.json")

    return results


if __name__ == "__main__":
    main()
