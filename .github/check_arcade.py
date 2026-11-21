"""Run only on hosted CI, against the iframe from the executed notebook."""
from pathlib import Path
import shutil
import sys

from playwright.sync_api import expect, sync_playwright

markup = Path(sys.argv[1]).read_text(encoding="utf-8")
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
chrome = shutil.which("google-chrome")
assert chrome, "Hosted runner Chrome is required"
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path=chrome)
    for width in (900, 390):
        page = browser.new_page(viewport={"width": width, "height": 520}, reduced_motion="reduce")
        page.set_content(markup)
        arcade = page.frame_locator('iframe[title="Spike Arcade"]')
        expect(arcade.get_by_role("heading", name="Spike Arcade")).to_be_visible()
        expect(arcade.get_by_role("status")).to_have_text("Tile 1 / 7")
        arcade.get_by_role("button", name="Jump", exact=True).click()
        expect(arcade.get_by_role("status")).to_have_text("Tile 4 / 7")
        arcade.get_by_role("button", name="Watch policy").click()
        arcade.get_by_role("button", name="Pause").click()
        arcade.get_by_role("button", name="Reset").click()
        expect(arcade.locator("#window b")).to_have_count(5)
        expect(arcade.locator("#decision")).to_contain_text("Next move: Jump")
        arcade.get_by_role("button", name="Step once").click()
        expect(arcade.get_by_role("status")).to_have_text("Tile 4 / 7")
        arcade.get_by_role("button", name="Step once").click()
        expect(arcade.get_by_role("status")).to_have_text("Track cleared!")
        arcade.get_by_role("combobox", name="Policy").select_option("short")
        expect(arcade.locator("#window b")).to_have_count(3)
        expect(arcade.get_by_role("button", name="Step once")).to_be_in_viewport()
        assert arcade.locator("main").evaluate("el => el.scrollWidth <= el.clientWidth")
        arcade.locator("main").screenshot(path=str(output / f"arcade-{width}.png"))
        page.close()
    browser.close()
