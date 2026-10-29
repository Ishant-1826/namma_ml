# Deployment Guide — Namma Space Viewer

## 1. Live Deployment Information

- **Production URL**: [https://namma-ml-viewer.pages.dev/](https://namma-ml-viewer.pages.dev/)
- **Deployment Hash URL**: [https://75352a0d.namma-ml-viewer.pages.dev/](https://75352a0d.namma-ml-viewer.pages.dev/)
- **Hosting Platform**: Cloudflare Pages (free tier, zero payment method required)
- **Project Name**: `namma-ml-viewer`
- **Branch**: `main`

---

## 2. Architecture & What Goes Where

All static web assets and compressed Gaussian splat assets are hosted directly on Cloudflare Pages:
- Every file is strictly under Cloudflare Pages' **25 MiB per-file limit** (the largest asset `scene.v1.spz` is 8.64 MB, well below the limit).
- Heavy uncompressed splat files (`scene.splat` at 13.4 MB, and `scene.ply` at 99 MB) are stripped from `dist/` by `viewer/vite.config.ts`.
- The R2 setup (`scripts/deploy_r2.ps1` and `docs/deploy/r2_cors.json`) is preserved in the repository as an optional fallback if offloading to Cloudflare R2 is desired in the future.

### Deployed File Inventory (`viewer/dist/`)

| Relative Path | Content Type | Size (Bytes) | Size (MB) | Cache-Control |
| :--- | :--- | :---: | :---: | :--- |
| `index.html` | `text/html; charset=utf-8` | 7,708 B | 0.01 MB | `public, max-age=0, must-revalidate` |
| `_headers` | `text/plain` | 120 B | < 0.01 MB | Cloudflare Pages rule file |
| `assets/index-*.js` | `application/javascript` | 3,221,925 B | 3.07 MB | Immutable / hashed |
| `assets/index-*.css` | `text/css` | 10,856 B | 0.01 MB | Immutable / hashed |
| `scene/scene.json` | `application/json` | 494 B | < 0.01 MB | `public, max-age=60` |
| `scene/scene.v1.spz` | `application/octet-stream` | 9,064,553 B | 8.64 MB | `public, max-age=31536000, immutable` |
| `scene/occupancy.json` | `application/json` | 406,189 B | 0.39 MB | Standard cache |
| `scene/occupancy.png` | `image/png` | 326,318 B | 0.31 MB | Standard cache |

**Total Transfer Size**: ~12.4 MB uncompressed (~10.5 MB over the wire with brotli/gzip CDN compression).

---

## 3. Cache & Header Configuration (`_headers`)

Cloudflare Pages applies caching directives from [`viewer/public/_headers`](file:///e:/namma_ml-main/viewer/public/_headers):
```
/scene/scene.json
  Cache-Control: public, max-age=60
/scene/*.spz
  Cache-Control: public, max-age=31536000, immutable
```

Verified via HTTP HEAD on the live deployment:
- `scene.json`: returns `Cache-Control: public, max-age=60`, ensuring updates propagate in <= 60 seconds.
- `scene.v1.spz`: returns `Cache-Control: public, max-age=31536000, immutable`, allowing browsers and edge caches to aggressively cache the 8.64 MB model without revalidation.

---

## 4. How to Redeploy

To build and deploy an update to Cloudflare Pages:

```powershell
# 1. Inside viewer directory:
cd viewer
npm run build

# 2. Deploy dist directory to Pages:
npx wrangler pages deploy dist --project-name namma-ml-viewer --branch main
```

Or run the automated scheduled task script:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_and_deploy_pages.ps1
```

---

## 5. How to Swap in a New Splat

To publish a newly trained or post-processed splat model:
1. Export the new model as a `.spz` (e.g. `scene.v2.spz`).
2. Copy it into `viewer/public/scene/scene.v2.spz`.
3. Update `viewer/public/scene/scene.json`:
   - Change `"splat_files": { "spz": "scene.v2.spz" }`
   - Update `"splat_count"` with the new Gaussian count.
4. Run `npm run build` in `viewer/`.
5. Deploy to Pages: `npx wrangler pages deploy dist --project-name namma-ml-viewer --branch main`.

Because the file name is versioned (`scene.v2.spz`), all returning visitors will immediately download the new model when `scene.json` expires (within 60 seconds), with zero CDN stale-cache conflicts.

