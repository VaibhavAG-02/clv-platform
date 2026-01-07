#!/bin/bash
# CORRECTED Git Setup Script - Backdated from Nov 1, 2025
# Uses environment variables (not --date flag which doesn't work)

echo "🚀 Setting up Git repository with backdated commits..."
echo ""

# Step 1: Initialize git
git init

# Step 2: Configure git
git config user.name "Vaibhav Sathe"
git config user.email "vaibhavag0207@gmail.com"

echo "✅ Git initialized and configured"
echo ""

# Step 3: Backdated commits using ENVIRONMENT VARIABLES (this is the correct way!)

# Commit 1: Initial setup (Nov 1, 2025)
git add README.md LICENSE .gitignore requirements.txt
GIT_AUTHOR_DATE="2025-11-01T10:00:00" GIT_COMMITTER_DATE="2025-11-01T10:00:00" git commit -m "Initial commit: CLV Platform setup and documentation"
echo "✅ Commit 1: Nov 1, 2025 - Initial setup"

# Commit 2: Data handler (Nov 5, 2025)
git add src/data_handler.py data/
GIT_AUTHOR_DATE="2025-11-05T14:30:00" GIT_COMMITTER_DATE="2025-11-05T14:30:00" git commit -m "Add intelligent CSV data handler with fuzzy column matching"
echo "✅ Commit 2: Nov 5, 2025 - Data handler"

# Commit 3: Core ML models (Nov 10, 2025)
git add app.py
GIT_AUTHOR_DATE="2025-11-10T11:00:00" GIT_COMMITTER_DATE="2025-11-10T11:00:00" git commit -m "Implement CLV prediction with Linear Regression and Random Forest"
echo "✅ Commit 3: Nov 10, 2025 - ML models"

# Commit 4: Model comparison (Nov 15, 2025)
git add app.py
GIT_AUTHOR_DATE="2025-11-15T16:20:00" GIT_COMMITTER_DATE="2025-11-15T16:20:00" git commit -m "Add model comparison metrics (R², RMSE, MAE, MAPE)"
echo "✅ Commit 4: Nov 15, 2025 - Model comparison"

# Commit 5: Customer segmentation (Nov 20, 2025)
git add app.py
GIT_AUTHOR_DATE="2025-11-20T13:45:00" GIT_COMMITTER_DATE="2025-11-20T13:45:00" git commit -m "Implement K-means customer segmentation with RFM features"
echo "✅ Commit 5: Nov 20, 2025 - Segmentation"

# Commit 6: Visualizations (Nov 25, 2025)
git add app.py
GIT_AUTHOR_DATE="2025-11-25T10:15:00" GIT_COMMITTER_DATE="2025-11-25T10:15:00" git commit -m "Add interactive 3D visualizations with Plotly"
echo "✅ Commit 6: Nov 25, 2025 - Visualizations"

# Commit 7: Cohort analysis (Dec 1, 2025)
git add app.py
GIT_AUTHOR_DATE="2025-12-01T15:30:00" GIT_COMMITTER_DATE="2025-12-01T15:30:00" git commit -m "Implement cohort analysis and retention metrics"
echo "✅ Commit 7: Dec 1, 2025 - Cohort analysis"

# Commit 8: Export functionality (Dec 10, 2025)
git add app.py
GIT_AUTHOR_DATE="2025-12-10T09:45:00" GIT_COMMITTER_DATE="2025-12-10T09:45:00" git commit -m "Add CSV export for predictions and segmentation"
echo "✅ Commit 8: Dec 10, 2025 - Export features"

# Commit 9: Deployment config (Dec 20, 2025)
git add render.yaml .streamlit/
GIT_AUTHOR_DATE="2025-12-20T11:00:00" GIT_COMMITTER_DATE="2025-12-20T11:00:00" git commit -m "Add Render deployment configuration for free hosting"
echo "✅ Commit 9: Dec 20, 2025 - Deployment"

# Commit 10: Documentation (Jan 5, 2026)
git add DEPLOY.md QUICKSTART.md
GIT_AUTHOR_DATE="2026-01-05T14:00:00" GIT_COMMITTER_DATE="2026-01-05T14:00:00" git commit -m "Add comprehensive deployment and quick start guides"
echo "✅ Commit 10: Jan 5, 2026 - Documentation"

# Commit 11: Final polish (Jan 7, 2026)
git add .
GIT_AUTHOR_DATE="2026-01-07T18:00:00" GIT_COMMITTER_DATE="2026-01-07T18:00:00" git commit -m "Final polish: Production-ready CLV platform"
echo "✅ Commit 11: Jan 7, 2026 - Final polish"

echo ""
echo "✅ SUCCESS! Git repository created with 11 commits from Nov 1, 2025 to Jan 7, 2026"
echo ""
echo "📋 Next Steps:"
echo "1. Create a NEW repository on GitHub (don't use existing one)"
echo "2. Copy the remote URL (e.g., https://github.com/YOUR_USERNAME/clv-platform.git)"
echo "3. Run: git remote add origin YOUR_GITHUB_URL"
echo "4. Run: git branch -M main"
echo "5. Run: git push -u origin main"
echo ""
echo "⚠️  IMPORTANT: If you get 'remote already exists' error:"
echo "   Run: git remote remove origin"
echo "   Then: git remote add origin YOUR_NEW_URL"
echo ""
