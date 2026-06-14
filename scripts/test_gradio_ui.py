#!/usr/bin/env python3
"""test_gradio_ui.py - Playwright-based Gradio UI interaction test

Tests:
1. Page loads correctly
2. Image upload works
3. Question input works
4. Generate button click triggers response
5. Example dropdown works
6. Advanced settings accordion works
7. Screenshot capture for visual inspection
"""

import os
import sys
import time
import json

from playwright.sync_api import sync_playwright

GRADIO_URL = "http://localhost:7861"
SCREENSHOT_DIR = "/workspace2/cy/Qwen3-0.6B/outputs/screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Test image path
TEST_IMAGE = "/workspace2/cy/Qwen3-0.6B/data/coco128_images/000000000009.jpg"

results = {
    "tests": [],
    "screenshots": [],
    "errors": [],
}


def record(name, passed, detail=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    results["tests"].append({"name": name, "passed": passed, "detail": detail})
    print(f"  {status} {name}: {detail}")


def screenshot(page, name):
    path = os.path.join(SCREENSHOT_DIR, f"{name}.png")
    page.screenshot(path=path, full_page=True)
    results["screenshots"].append(path)
    print(f"  📸 Screenshot saved: {path}")


def main():
    print("=" * 70)
    print("  Gradio UI Interaction Test with Playwright")
    print("=" * 70)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="en-US",
        )
        page = context.new_page()

        # Collect console errors
        console_errors = []
        page.on("console", lambda msg: console_errors.append(f"[{msg.type}] {msg.text}") if msg.type in ["error", "warning"] else None)

        # Collect network errors
        network_errors = []
        page.on("requestfailed", lambda req: network_errors.append(f"[FAIL] {req.url} - {req.failure}"))

        # ============================================================
        # Test 1: Page Load
        # ============================================================
        print("\n[1/7] Testing page load...")
        try:
            response = page.goto(GRADIO_URL, timeout=30000, wait_until="networkidle")
            if response and response.status == 200:
                record("Page Load", True, f"Status {response.status}")
            else:
                record("Page Load", False, f"Status {response.status}")
        except Exception as e:
            record("Page Load", False, str(e))
            print("  ⚠️ Cannot continue - page failed to load")
            browser.close()
            return

        time.sleep(2)  # Wait for Gradio JS to fully render
        screenshot(page, "01_page_loaded")

        # Check page title
        title = page.title()
        record("Page Title", "Qwen3" in title or "Vision" in title, f"Title: '{title}'")

        # ============================================================
        # Test 2: Check UI Components Exist
        # ============================================================
        print("\n[2/7] Checking UI components...")

        # Image upload component
        image_input = page.locator('input[type="file"]')
        image_visible = image_input.count() > 0
        record("Image Upload Component", image_visible, f"Found {image_input.count()} file inputs")

        # Text input for question
        question_input = page.locator('textarea, input[type="text"]').first
        question_visible = question_input.is_visible()
        record("Question Input Component", question_visible, f"Visible: {question_visible}")

        # Generate button
        generate_btn = page.locator('button:has-text("Generate")')
        btn_visible = generate_btn.is_visible()
        record("Generate Button", btn_visible, f"Visible: {btn_visible}")

        # Example dropdown
        dropdown = page.locator('label:has-text("Example") + *, [data-testid="dropdown"]')
        dropdown_count = dropdown.count()
        record("Example Dropdown", dropdown_count > 0, f"Found {dropdown_count} dropdowns")

        # Advanced settings accordion
        accordion = page.locator('span:has-text("Advanced")')
        accordion_visible = accordion.count() > 0
        record("Advanced Settings Accordion", accordion_visible, f"Found: {accordion_visible}")

        # Gallery for example images
        gallery = page.locator('[data-testid="gallery"], .gallery')
        gallery_count = gallery.count()
        record("Example Gallery", gallery_count > 0, f"Found {gallery_count} galleries")

        # ============================================================
        # Test 3: Image Upload
        # ============================================================
        print("\n[3/7] Testing image upload...")

        if os.path.exists(TEST_IMAGE):
            try:
                # Find the file input (may be hidden behind a button)
                file_inputs = page.locator('input[type="file"]')
                if file_inputs.count() > 0:
                    # Upload the test image
                    file_inputs.first.set_input_files(TEST_IMAGE)
                    time.sleep(2)  # Wait for image preview to load
                    screenshot(page, "02_after_image_upload")
                    record("Image Upload", True, f"Uploaded {TEST_IMAGE}")
                else:
                    record("Image Upload", False, "No file input found")
            except Exception as e:
                record("Image Upload", False, str(e))
                screenshot(page, "02_image_upload_error")
        else:
            record("Image Upload", False, f"Test image not found: {TEST_IMAGE}")

        # ============================================================
        # Test 4: Question Input
        # ============================================================
        print("\n[4/7] Testing question input...")

        try:
            # Find the question textarea
            textareas = page.locator('textarea')
            if textareas.count() > 0:
                # Clear and type a question
                textareas.first.click()
                textareas.first.fill("What objects can you see in this image?")
                time.sleep(0.5)
                input_value = textareas.first.input_value()
                record("Question Input", "objects" in input_value, f"Value: '{input_value[:50]}...'")
                screenshot(page, "03_after_question_input")
            else:
                record("Question Input", False, "No textarea found")
        except Exception as e:
            record("Question Input", False, str(e))

        # ============================================================
        # Test 5: Generate Button Click
        # ============================================================
        print("\n[5/7] Testing generate button click...")

        try:
            generate_btn = page.locator('button:has-text("Generate")')
            if generate_btn.is_visible():
                generate_btn.click()
                print("  ⏳ Waiting for model response (up to 60s)...")

                # Wait for response to appear - look for text in the output area
                # The output textbox should get content
                try:
                    # Wait for loading indicator to appear then disappear
                    loading = page.locator('.loading, [aria-label="Loading"]')
                    if loading.count() > 0:
                        print("  Loading indicator detected...")

                    # Wait for response text to appear (check output textbox)
                    output_area = page.locator('[data-testid="textbox"]').last
                    if output_area.count() == 0:
                        # Try alternative selectors
                        output_area = page.locator('textarea[readonly], .output-textbox').last

                    # Wait up to 60 seconds for response
                    time.sleep(10)  # Initial wait for model inference
                    screenshot(page, "04_generating")

                    # Check periodically for response
                    for i in range(10):
                        time.sleep(5)
                        # Check if there's any response text
                        page_text = page.inner_text('body')
                        if len(page_text) > 500:  # Response should add significant text
                            break

                    screenshot(page, "05_after_generation")

                    # Check for response content
                    all_textboxes = page.locator('textarea')
                    if all_textboxes.count() >= 2:
                        response_text = all_textboxes.last.input_value()
                        has_response = len(response_text) > 10
                        record("Generate Response", has_response,
                               f"Response length: {len(response_text)} chars, preview: '{response_text[:100]}...'")
                    else:
                        # Try to find response in any element
                        body_text = page.inner_text('body')
                        has_response = len(body_text) > 200
                        record("Generate Response", has_response,
                               f"Body text length: {len(body_text)}")

                except Exception as e:
                    record("Generate Response", False, f"Error waiting for response: {e}")
                    screenshot(page, "05_generation_error")
            else:
                record("Generate Response", False, "Generate button not visible")
        except Exception as e:
            record("Generate Response", False, str(e))

        # ============================================================
        # Test 6: Example Dropdown
        # ============================================================
        print("\n[6/7] Testing example dropdown...")

        try:
            # Find and click the dropdown
            dropdowns = page.locator('[data-testid="dropdown"], select')
            if dropdowns.count() > 0:
                dropdowns.first.click()
                time.sleep(1)
                screenshot(page, "06_dropdown_opened")

                # Try to select an option
                options = page.locator('[role="option"], [data-testid="dropdown-option"]')
                if options.count() > 0:
                    options.first.click()
                    time.sleep(0.5)
                    record("Example Dropdown Select", True, f"Selected first option, {options.count()} options available")
                else:
                    # Try clicking a list item
                    list_items = page.locator('li')
                    if list_items.count() > 0:
                        list_items.first.click()
                        record("Example Dropdown Select", True, "Selected via list item")
                    else:
                        record("Example Dropdown Select", False, "No options found in dropdown")
            else:
                record("Example Dropdown", False, "No dropdown found")
        except Exception as e:
            record("Example Dropdown", False, str(e))

        # ============================================================
        # Test 7: Advanced Settings Accordion
        # ============================================================
        print("\n[7/7] Testing advanced settings accordion...")

        try:
            # Click the accordion to expand
            accordion_btn = page.locator('button:has-text("Advanced"), summary:has-text("Advanced"), span:has-text("Advanced")')
            if accordion_btn.count() > 0:
                accordion_btn.first.click()
                time.sleep(1)
                screenshot(page, "07_advanced_settings")

                # Check if sliders are now visible
                sliders = page.locator('input[type="range"], [role="slider"]')
                slider_count = sliders.count()
                record("Advanced Settings", slider_count > 0, f"Found {slider_count} sliders after expanding")
            else:
                record("Advanced Settings", False, "Accordion not found")
        except Exception as e:
            record("Advanced Settings", False, str(e))

        # ============================================================
        # Collect Errors
        # ============================================================
        if console_errors:
            results["errors"].extend([f"Console: {e}" for e in console_errors[-20:]])
            print(f"\n⚠️ Console errors/warnings: {len(console_errors)}")
            for e in console_errors[-5:]:
                print(f"  {e}")

        if network_errors:
            results["errors"].extend([f"Network: {e}" for e in network_errors[-10:]])
            print(f"\n⚠️ Network errors: {len(network_errors)}")
            for e in network_errors[-5:]:
                print(f"  {e}")

        # Final screenshot
        screenshot(page, "08_final_state")

        browser.close()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)

    passed = sum(1 for t in results["tests"] if t["passed"])
    total = len(results["tests"])
    print(f"  Passed: {passed}/{total}")

    for t in results["tests"]:
        status = "✅" if t["passed"] else "❌"
        print(f"  {status} {t['name']}: {t['detail']}")

    if results["errors"]:
        print(f"\n  Errors captured: {len(results['errors'])}")
        for e in results["errors"][:10]:
            print(f"    {e}")

    print(f"\n  Screenshots saved to: {SCREENSHOT_DIR}")

    # Save results
    results_path = "/workspace2/cy/Qwen3-0.6B/outputs/ui_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    main()
