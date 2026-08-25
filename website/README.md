# SimpleNAV Project Page

This directory contains the source for the static, interactive project page. It is intentionally separate from the repository README so the landing page can remain easy to read and the website can evolve without generating committed HTML output.

## Local preview

From the repository root:

```bash
python3 website/build.py
python3 -m http.server 8000 --directory _site
```

Open <http://127.0.0.1:8000/>. The page uses a small JSON data file and therefore needs an HTTP server; opening `website/index.html` directly with `file://` will block the data request in most browsers.

## Structure

- `index.html`: page structure and reserved section slots;
- `assets/styles.css`: restrained responsive layout and component styles;
- `assets/app.js`: language toggle, featured-rollout carousel, framework/model tabs, benchmark switcher, filtered video gallery, and scroll navigation;
- `data/site.json`: bilingual page copy, framework data, Release 01 metrics, demos, roadmap, and document links;
- `build.py`: copies the source into `_site/` and adds the project logo, rollout gallery, README GIF previews, and trajectory-augmentation comparisons.

The generated `_site/` directory is a deployment artifact and should not be committed.

## First-time GitHub Pages setup

The workflow checks whether Pages is enabled before deploying. If it is disabled, the build still succeeds and uploads a `github-pages` artifact, while the deploy job is skipped with a warning. A repository administrator must select **Settings → Pages → Build and deployment → GitHub Actions** once. The `SimpleNav` workflow uses a separate `github-pages-SimpleNav` environment so it is not blocked by an existing `dev`-only deployment policy. After Pages is enabled, rerun the workflow from the Actions page; later pushes to `SimpleNav` deploy automatically.
