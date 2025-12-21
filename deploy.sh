#!/bin/bash

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ============================================================================
# Deployment Script with Safety Checks and Rollback Capability
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="/tmp/deploy.lock"
LOG_FILE="${SCRIPT_DIR}/deploy.log"
APP_NAME="${APP_NAME:-chatbot}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

cleanup() {
    rm -f "$LOCK_FILE"
}

trap cleanup EXIT

# ============================================================================
# Pre-deployment Checks
# ============================================================================

log "Starting deployment pre-checks..."

# Check for deployment lock
if [ -f "$LOCK_FILE" ]; then
    error "Deployment already in progress (lock file exists: $LOCK_FILE)"
    exit 1
fi

touch "$LOCK_FILE"

# Ensure we're in the project directory
cd "$SCRIPT_DIR"

# Add common uv installation paths
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Check uv is available
if ! command -v uv &> /dev/null; then
    error "uv not found in PATH. Please install uv first."
    exit 1
fi

# Check PM2 is available
if ! command -v pm2 &> /dev/null; then
    error "pm2 not found in PATH. Please install PM2 first."
    exit 1
fi

# Load environment variables safely
if [ -f .env ]; then
    log "Loading environment variables from .env..."
    set -a
    source .env
    set +a
else
    warn ".env file not found, using existing environment variables"
fi

# Check disk space (warn if less than 1GB available)
AVAILABLE_SPACE=$(df -k . | awk 'NR==2 {print $4}')
if [ "$AVAILABLE_SPACE" -lt 1048576 ]; then
    warn "Low disk space: less than 1GB available"
fi

# ============================================================================
# Git Operations
# ============================================================================

log "Fetching latest changes from origin..."
git fetch origin develop

# Check if we can fast-forward
if ! git merge-base --is-ancestor HEAD origin/develop; then
    error "Cannot fast-forward. Local branch has diverged from origin/develop"
    error "Please resolve conflicts manually before deploying"
    exit 1
fi

# Store current commit for rollback
PREVIOUS_COMMIT=$(git rev-parse HEAD)
log "Current commit: $PREVIOUS_COMMIT"

log "Pulling latest changes..."
if ! git pull origin develop --ff-only; then
    error "Failed to pull changes from origin/develop"
    exit 1
fi

NEW_COMMIT=$(git rev-parse HEAD)
log "New commit: $NEW_COMMIT"

# ============================================================================
# Dependencies Installation
# ============================================================================

log "Installing/updating dependencies with uv sync..."
if ! uv sync; then
    error "Failed to install dependencies"
    log "Rolling back to commit: $PREVIOUS_COMMIT"
    git reset --hard "$PREVIOUS_COMMIT"
    exit 1
fi

# ============================================================================
# Database Migrations
# ============================================================================

log "Running database migrations..."
if ! uv run alembic upgrade head; then
    error "Migration failed!"
    warn "Database may be in an inconsistent state"
    log "Rolling back code to commit: $PREVIOUS_COMMIT"
    git reset --hard "$PREVIOUS_COMMIT"
    uv sync
    error "Manual database rollback may be required"
    exit 1
fi

# ============================================================================
# Application Restart
# ============================================================================

log "Restarting application: $APP_NAME"

# Use PM2 reload for zero-downtime restart, fallback to restart
if pm2 describe "$APP_NAME" &> /dev/null; then
    log "App exists in PM2, performing reload..."
    if ! pm2 reload "$APP_NAME" --update-env; then
        warn "Reload failed, attempting restart..."
        pm2 restart "$APP_NAME"
    fi
else
    log "App not found in PM2, starting from ecosystem config..."
    pm2 start ecosystem.config.js
fi

# ============================================================================
# Health Check
# ============================================================================

log "Waiting for application to start..."
sleep 3

# Check PM2 status
if ! pm2 describe "$APP_NAME" | grep -q "online"; then
    error "Application failed to start!"
    log "Rolling back to commit: $PREVIOUS_COMMIT"
    git reset --hard "$PREVIOUS_COMMIT"
    uv sync
    pm2 restart "$APP_NAME"
    exit 1
fi

# Optional: HTTP health check (uncomment and configure if you have a health endpoint)
# if command -v curl &> /dev/null; then
#     log "Performing health check..."
#     for i in {1..5}; do
#         if curl -f http://localhost:8000/health &> /dev/null; then
#             log "Health check passed"
#             break
#         fi
#         if [ $i -eq 5 ]; then
#             error "Health check failed after 5 attempts"
#             exit 1
#         fi
#         sleep 2
#     done
# fi

# ============================================================================
# Deployment Summary
# ============================================================================

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Deployment completed successfully!"
log "App: $APP_NAME"
log "Commit: $NEW_COMMIT"
log "Deployed by: $(whoami)"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Show PM2 status
pm2 list