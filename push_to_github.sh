#!/usr/bin/env bash
# push_to_github.sh
# Run this once from the PEAT/ directory to initialize and push to KhareLab/PEAT.
set -e

REMOTE="https://github.com/KhareLab/PEAT.git"

echo "==> Initializing git repo..."
git init
git checkout -b main

echo "==> Staging all files..."
git add .
git commit -m "feat: PEAT Q1 roadmap changes

- Replace OpenAI with OpenRouter (OpenAI-compatible, cost-free dev)
- Add LLM_BASE_URL env var for HPC-hosted prod endpoint
- Fix Unpaywall RAG: use best_oa_location PDF URL
- Add library auth cookie fallback (cf. Zotero) for prod
- Add Sci-Hub fallback (dev only, SCIHUB_ENABLED=true flag)
- Add hpc_tools.py: gmx/HPC command execution with SSH routing
- Add HPC sidebar widget in app.py
- Add docs/index.html: GitHub Pages landing site
- Add .github/workflows/pages.yml: auto-deploy Pages on push
- Update requirements.txt: remove unused heavy deps, add openai>=1.30
- Update .gitignore: exclude secrets, temp.pdb/pdf, venv"

echo "==> Adding remote..."
git remote add origin "$REMOTE" 2>/dev/null || git remote set-url origin "$REMOTE"

echo "==> Pushing to main..."
git push -u origin main

echo ""
echo "✅ Done! Next steps:"
echo "   1. Go to https://github.com/KhareLab/PEAT/settings/pages"
echo "   2. Under 'Build and deployment', select:"
echo "      Source: GitHub Actions"
echo "   3. Your site will be live at https://kharelab.github.io/PEAT"
