#!/bin/bash
set -e

# Configuration for Account 1 (where we deploy)
RESOURCE_GROUP="churn-prediction-rg"
LOCATION="eastus"
ENVIRONMENT="${ENVIRONMENT:-production}"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
    exit 1
}

# Validate required environment variables
validate_env() {
    local required_vars=("POSTGRES_HOST" "POSTGRES_PASSWORD" "AZURE_STORAGE_CONNECTION_STRING" "AUTH_SECRET")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            error "Missing required environment variable: $var"
        fi
    done
    log "All required environment variables are set"
}

# Switch to Account 1 context
switch_to_account1() {
    log "Switching to Account 1 (Main account)"
    az account set --subscription "$AZURE_SUBSCRIPTION_ID" || error "Failed to switch to Account 1"
}

# Deploy to Account 1
deploy_to_account1() {
    local api_image="$1"
    local data_pipeline_image="$2"
    local training_image="$3"
    
    log "Deploying to Account 1 (Main account)"
    
    # Create resource group if needed
    az group create \
        --name "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --tags "environment=$ENVIRONMENT" "project=churn-prediction" \
        --output none
    
    # Deploy API
    log "Deploying API container..."
    az container create \
        --resource-group "$RESOURCE_GROUP" \
        --name "churn-api-$ENVIRONMENT" \
        --image "$api_image" \
        --cpu 1 \
        --memory 2 \
        --ports 8000 \
        --ip-address Public \
        --environment-variables \
            POSTGRES_HOST="$POSTGRES_HOST" \
            POSTGRES_PORT="$POSTGRES_PORT" \
            POSTGRES_DB="$POSTGRES_DB_NAME" \
            POSTGRES_USER="$POSTGRES_DB_USER" \
            AZURE_STORAGE_CONNECTION_STRING="$AZURE_STORAGE_CONNECTION_STRING" \
            AZURE_ML_SUBSCRIPTION_ID="$AZURE2_SUBSCRIPTION_ID" \
            AZURE_ML_RESOURCE_GROUP="$AZURE2_RESOURCE_GROUP" \
            AZURE_ML_WORKSPACE_NAME="$AZURE2_ML_WORKSPACE_NAME" \
            AUTH_SECRET="$AUTH_SECRET" \
            ENVIRONMENT="$ENVIRONMENT" \
            LOG_LEVEL="$LOG_LEVEL" \
        --secrets \
            POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
        --dns-name-label "churn-api-${ENVIRONMENT}-$(date +%s)" \
        --restart-policy Always \
        --output table
    
    # Deploy Data Pipeline (on-demand)
    log "Deploying Data Pipeline container..."
    az container create \
        --resource-group "$RESOURCE_GROUP" \
        --name "churn-data-pipeline-$ENVIRONMENT" \
        --image "$data_pipeline_image" \
        --cpu 1 \
        --memory 1.5 \
        --environment-variables \
            POSTGRES_HOST="$POSTGRES_HOST" \
            POSTGRES_PORT="$POSTGRES_PORT" \
            POSTGRES_DB="$POSTGRES_DB_NAME" \
            POSTGRES_USER="$POSTGRES_DB_USER" \
            AZURE_STORAGE_CONNECTION_STRING="$AZURE_STORAGE_CONNECTION_STRING" \
            ENVIRONMENT="$ENVIRONMENT" \
            LOG_LEVEL="$LOG_LEVEL" \
        --secrets \
            POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
        --restart-policy Never \
        --output table
    
    # Deploy Training (on-demand)
    log "Deploying Training container..."
    az container create \
        --resource-group "$RESOURCE_GROUP" \
        --name "churn-training-$ENVIRONMENT" \
        --image "$training_image" \
        --cpu 2 \
        --memory 4 \
        --environment-variables \
            POSTGRES_HOST="$POSTGRES_HOST" \
            POSTGRES_PORT="$POSTGRES_PORT" \
            POSTGRES_DB="$POSTGRES_DB_NAME" \
            POSTGRES_USER="$POSTGRES_DB_USER" \
            AZURE_STORAGE_CONNECTION_STRING="$AZURE_STORAGE_CONNECTION_STRING" \
            AZURE_ML_SUBSCRIPTION_ID="$AZURE2_SUBSCRIPTION_ID" \
            AZURE_ML_RESOURCE_GROUP="$AZURE2_RESOURCE_GROUP" \
            AZURE_ML_WORKSPACE_NAME="$AZURE2_ML_WORKSPACE_NAME" \
            MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
            ENVIRONMENT="$ENVIRONMENT" \
            LOG_LEVEL="$LOG_LEVEL" \
        --secrets \
            POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
        --restart-policy Never \
        --output table
}

main() {
    # Validate environment first
    validate_env
    
    # Switch to Account 1 for deployment
    switch_to_account1
    
    # Deploy containers to Account 1
    deploy_to_account1 "$1" "$2" "$3"
    
    log "✅ Deployment completed to Account 1"
    log "📊 Azure ML tracking in Account 2: $AZURE2_ML_WORKSPACE_NAME"
    log "🌐 API URL: churn-api-${ENVIRONMENT}-*.${LOCATION}.azurecontainer.io:8000"
}

# Run with image arguments
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <api-image> <data-pipeline-image> <training-image>"
    echo "Example: $0 ghcr.io/org/api:latest ghcr.io/org/data:latest ghcr.io/org/training:latest"
    exit 1
fi

main "$1" "$2" "$3"