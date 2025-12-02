Automated Grafana Dashboard Image Capture (Playwright MCP)
=========================================================

Goal
----
Use Playwright (as an MCP action) to log into Grafana with a viewer account, open specific dashboards or panels, and save fresh screenshots for LLM agents (Copilot/Codex) to read.

Prerequisites
-------------
- Grafana reachable from this machine (local URL or VPN).
- A viewer-level account or service account. Avoid admin keys.
- Playwright installed (Node/TS/JS). Example assumes `npx playwright ...` works.
- Node 18+.
- A writable folder where agents can read outputs (e.g., `./_grafana_exports`).

Grafana prep (one-time)
-----------------------
1) Create a viewer account or service account with least privilege.  
2) Note your target dashboard URLs (or panel direct URLs) and their UIDs.  
3) If Grafana uses SSO only, do one manual login and export the Playwright storage state JSON (see below). Otherwise, plan to log in with username/password or basic auth token on each run.

Project layout (example)
------------------------
```
scripts/grafana/
  capture.js           # Playwright runner
  state.json           # Optional saved login session (kept locally)
_grafana_exports/      # Output PNGs for agents
```

Environment variables
---------------------
Set these in your shell or a local `.env` (do not commit secrets):
- `GRAFANA_URL` (e.g., `https://grafana.local`)
- `GRAFANA_USER`
- `GRAFANA_PASS`
- `GRAFANA_DASH_URL` (full dashboard or panel URL; include time range if desired)
- `GRAFANA_OUT_DIR` (default `_grafana_exports`)
- Optional: `GRAFANA_STATE_PATH` (e.g., `scripts/grafana/state.json`) if using saved cookies.

Playwright script (JavaScript)
------------------------------
Place in `scripts/grafana/capture.js`.

```js
// Run with: node scripts/grafana/capture.js
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

async function main() {
  const baseUrl = process.env.GRAFANA_URL;
  const dashUrl = process.env.GRAFANA_DASH_URL;
  const outDir = process.env.GRAFANA_OUT_DIR || path.resolve("_grafana_exports");
  const statePath = process.env.GRAFANA_STATE_PATH;

  if (!baseUrl || !dashUrl) throw new Error("Set GRAFANA_URL and GRAFANA_DASH_URL");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext(
    statePath && fs.existsSync(statePath) ? { storageState: statePath } : {}
  );
  const page = await context.newPage();

  // Login if no state is present
  if (!statePath || !fs.existsSync(statePath)) {
    await page.goto(`${baseUrl}/login`, { waitUntil: "networkidle" });
    await page.fill('input[name="user"]', process.env.GRAFANA_USER || "");
    await page.fill('input[name="password"]', process.env.GRAFANA_PASS || "");
    await page.click('button[type="submit"]');
    await page.waitForLoadState("networkidle");
    if (statePath) await context.storageState({ path: statePath });
  }

  await page.goto(dashUrl, { waitUntil: "networkidle" });
  await page.waitForTimeout(2000); // allow panels to render

  const fileName = `grafana_${Date.now()}.png`;
  const outPath = path.join(outDir, fileName);
  await page.screenshot({ path: outPath, fullPage: true });

  await browser.close();
  console.log(`Saved screenshot: ${outPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
```

Capturing a saved login session (optional)
------------------------------------------
If your Grafana uses SSO or MFA, do one manual login and capture cookies:
```
GRAFANA_URL=https://grafana.local npx playwright codegen
# Login once in the opened browser, then:
# context.storageState({ path: 'scripts/grafana/state.json' });
```
Then rerun the script with `GRAFANA_STATE_PATH=scripts/grafana/state.json` to avoid re-authenticating.

Running manually
----------------
```
$env:GRAFANA_URL="https://grafana.local"
$env:GRAFANA_USER="viewer"
$env:GRAFANA_PASS="password"
$env:GRAFANA_DASH_URL="https://grafana.local/d/uid/dashboard?orgId=1&from=now-6h&to=now"
node scripts/grafana/capture.js
```
Output PNG will land in `_grafana_exports/`.

Automate (Windows Task Scheduler)
---------------------------------
1) Create a basic task that runs under your user.  
2) Action: `Program/script`: `node`  
   Args: `scripts/grafana/capture.js`  
   Start in: repo root.  
3) Set needed env vars in a wrapper `.cmd` or via Task Scheduler `Start in` + `setx` beforehand.  
4) Schedule e.g., every 15 minutes. Ensure `_grafana_exports` remains readable by agents.

Feeding images to agents
------------------------
- Point the MCP action to the output folder `_grafana_exports` and return the latest PNG path(s).  
- For Copilot/Codex, pass the screenshot path(s) or embed the PNG bytes as needed.  
- Keep filenames time-stamped for ordering.

Variations
----------
- Panel-only capture: use Grafana solo panel URL and `page.locator(...)` with `screenshot({ clip })` to trim the image.  
- Data export: script can click panel menu > Inspect > Data > Download CSV; save beside the PNG for structured analysis.  
- Multiple dashboards: loop over an array of URLs and write one PNG per URL.

Troubleshooting
---------------
- Blank panels: increase `waitForTimeout`, or wait on a panel selector: `page.waitForSelector('[data-testid="panel-title"]')`.  
- Auth loops: regenerate `state.json` or verify user/pass.  
- 403/404: verify orgId and permissions for the viewer account.  
- Large pages: consider `viewport` size in `newContext` or clip specific panels to reduce file size.

Hardening (for later)
---------------------
- Rotate viewer credentials; keep tokens out of source control.  
- Restrict MCP to a whitelist of allowed dashboard URLs.  
- Add basic redaction if screenshots could show secrets.
