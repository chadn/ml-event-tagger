# Deployment Guide - Render

This guide walks through deploying the ML Event Tagger API to Render.

## Prerequisites

-   GitHub account
-   Render account (free tier works)
-   Code pushed to a GitHub repository

## Deployment Steps

### 1. Prepare Repository

Ensure your code is pushed to GitHub:

```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

### 2. Connect to Render

1. Go to [https://render.com](https://render.com)
2. Sign in with GitHub
3. Click "New +" → "Web Service"
4. Connect your `ml-event-tagger` repository

### 3. Configure Service

Render will auto-detect the `render.yaml` configuration file. Verify settings:

-   **Name:** `ml-event-tagger`
-   **Runtime:** Docker
-   **Region:** Oregon (or closest to you)
-   **Branch:** main
-   **Plan:** Free
-   **Health Check Path:** `/health`

### 4. Environment Variables

These are pre-configured in `render.yaml`:

-   `PORT=8000`
-   `PYTHONUNBUFFERED=1`

### 5. Deploy

Click "Create Web Service" and wait for:

-   ✅ Docker build to complete (~5-10 minutes first time)
-   ✅ Health check to pass
-   ✅ Service to go live

### 6. Test Deployment

Once live, your API will be at:

```
https://ml-event-tagger-<random-id>.onrender.com
```

Test with:

```bash
# Health check
curl https://ml-event-tagger-<your-id>.onrender.com/health

# Prediction
curl -X POST https://ml-event-tagger-<your-id>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "events": [{
      "name": "House Music Night",
      "description": "DJ performance with dancing",
      "location": "Oakland, CA"
    }]
  }'
```

## Render Free Tier Limitations

⚠️ **Important for Free Tier:**

-   Service spins down after 15 minutes of inactivity
-   First request after spin-down takes ~30-60 seconds (cold start)
-   750 hours/month free (enough for a demo)
-   Suitable for demos and portfolios

## Monitoring

View logs in Render dashboard:

1. Go to your service
2. Click "Logs" tab
3. Monitor startup and requests

## Troubleshooting

### Build Fails

Check:

-   `Dockerfile` is valid
-   All required files are in repo (especially `models/`)
-   Dependencies install correctly

### Health Check Fails

-   Verify `/health` endpoint returns 200
-   Check model files are included in Docker image
-   Review logs for startup errors

### Slow Cold Starts

-   Expected on free tier
-   Consider upgrading to paid plan for production
-   Or use a keep-alive service to ping periodically

## Alternative: Docker Hub + Render

If you want to use pre-built images:

```bash
# Build and tag
docker build -t yourusername/ml-event-tagger:0.0.6 .

# Push to Docker Hub
docker push yourusername/ml-event-tagger:0.0.6

# Update render.yaml to use image instead of Dockerfile
```

## Production Recommendations

For production use, consider:

-   ✅ Upgrade to paid plan (persistent service)
-   ✅ Add authentication (API keys)
-   ✅ Add rate limiting
-   ✅ Add CORS configuration
-   ✅ Add monitoring/logging service
-   ✅ Use CDN for model files

## Cost Estimate

-   **Free tier:** $0/month (750 hours, spin-down after 15 min)
-   **Starter:** $7/month (persistent, 512MB RAM)
-   **Standard:** $25/month (1GB RAM, better performance)

For a portfolio demo, free tier is sufficient!

## Documentation

-   Render Docs: https://render.com/docs
-   Docker deployment: https://render.com/docs/docker
-   Health checks: https://render.com/docs/health-checks
