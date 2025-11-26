#!/bin/bash

set -e

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

APP_NAME="${APP_NAME:-chatbot}"

echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting deployment..."

# Pull latest changes
echo "Pulling latest changes..."
git pull origin develop

# Install/update dependencies
echo "Installing dependencies..."
uv sync

# Run database migrations
echo "Running database migrations..."
uv run alembic upgrade head

# Restart PM2 app
echo "Restarting $APP_NAME..."
pm2 restart "$APP_NAME" || pm2 start ecosystem.config.js

echo "$(date '+%Y-%m-%d %H:%M:%S') - Deployment completed successfully!"