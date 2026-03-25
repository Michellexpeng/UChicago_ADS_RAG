#!/bin/bash
set -e

# =============================================================================
# UChicago ADS RAG — GCP Deployment Script
# =============================================================================
# Prerequisites:
#   1. gcloud CLI installed & logged in (gcloud auth login)
#   2. Firebase CLI installed (npm install -g firebase-tools && firebase login)
#   3. GCP project set (gcloud config set project uchicago-ads-rag)
#   4. APIs enabled (gcloud services enable run.googleapis.com artifactregistry.googleapis.com)
#   5. backend/.env file with GOOGLE_API_KEY
# =============================================================================

PROJECT_ID="uchicago-ads-rag"
REGION="us-central1"
SERVICE_NAME="uchicago-rag-backend"

# Load GOOGLE_API_KEY from backend/.env
if [ -f backend/.env ]; then
  GOOGLE_API_KEY=$(grep GOOGLE_API_KEY backend/.env | cut -d '=' -f2)
else
  echo "ERROR: backend/.env not found. Please create it with GOOGLE_API_KEY=your-key"
  exit 1
fi

echo "=========================================="
echo "Step 1: Deploying backend to Cloud Run..."
echo "=========================================="
gcloud run deploy "$SERVICE_NAME" \
  --source ./backend \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --memory 4Gi \
  --cpu 4 \
  --cpu-boost \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 300 \
  --set-env-vars "GOOGLE_API_KEY=$GOOGLE_API_KEY" \
  --allow-unauthenticated

echo "=========================================="
echo "Step 2: Getting backend URL..."
echo "=========================================="
BACKEND_URL=$(gcloud run services describe "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format 'value(status.url)')
echo "Backend URL: $BACKEND_URL"

# Update CORS to allow Firebase Hosting domain
echo "Updating CORS settings..."
gcloud run services update "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --set-env-vars "^::^GOOGLE_API_KEY=$GOOGLE_API_KEY::ALLOWED_ORIGINS=https://${PROJECT_ID}.web.app,https://${PROJECT_ID}.firebaseapp.com"

echo "=========================================="
echo "Step 3: Building frontend..."
echo "=========================================="
cd frontend
VITE_API_URL="$BACKEND_URL" npm run build
echo "Frontend built successfully."

echo "=========================================="
echo "Step 4: Deploying frontend to Firebase..."
echo "=========================================="
firebase deploy --only hosting --project "$PROJECT_ID"

cd ..

echo "=========================================="
echo "Step 5: Setting up Cloud Scheduler keepalive..."
echo "=========================================="
# Ping /health every 5 minutes to avoid cold starts (instead of min-instances=1)
if gcloud scheduler jobs describe rag-keepalive --project "$PROJECT_ID" --location "$REGION" &>/dev/null; then
  gcloud scheduler jobs update http rag-keepalive \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --schedule="*/5 * * * *" \
    --uri="$BACKEND_URL/health" \
    --http-method=GET
else
  gcloud scheduler jobs create http rag-keepalive \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --schedule="*/5 * * * *" \
    --uri="$BACKEND_URL/health" \
    --http-method=GET
fi
echo "Keepalive scheduler: every 5 min → $BACKEND_URL/health"

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo "Backend:  $BACKEND_URL"
echo "Frontend: https://${PROJECT_ID}.web.app"
echo ""
echo "Test backend: curl $BACKEND_URL/health"
