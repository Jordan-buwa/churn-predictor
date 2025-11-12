#!/bin/bash
set -e

# Configuration
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
    local required_vars=(
        "AZURE_SUBSCRIPTION_ID"
        "POSTGRES_HOST" 
        "POSTGRES_PASSWORD" 
        "AZURE_STORAGE_CONNECTION_STRING" 
        "AUTH_SECRET"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            error "Missing required environment variable: $var"
        fi
    done
    
    # Check if Account 2 credentials are provided (optional but recommended)
    if [[ -n "${AZURE2_SUBSCRIPTION_ID}" ]] && [[ -n "${AZURE2_ML_WORKSPACE_NAME}" ]]; then
        log "✅ Account 2 (ML) configuration detected"
        
        # Support both naming conventions for Account 2 credentials
        # GitHub secrets: AZURE_CLIENT_ID or AZURE2_CLIENT_ID
        CLIENT_ID="${AZURE2_CLIENT_ID:-${AZURE_CLIENT_ID}}"
        CLIENT_SECRET="${AZURE2_CLIENT_SECRET:-${AZURE_CLIENT_SECRET}}"
        TENANT_ID="${AZURE2_TENANT_ID:-${AZURE_TENANT_ID}}"
        
        if [[ -n "${CLIENT_ID}" ]] && [[ -n "${CLIENT_SECRET}" ]] && [[ -n "${TENANT_ID}" ]]; then
            log "✅ Account 2 service principal credentials provided"
        else
            log "⚠️  Account 2 configured but no service principal credentials. Containers will use DefaultAzureCredential."
        fi
    fi
    
    log "✅ All required environment variables are set"
}

# Verify Azure login and subscription access
verify_azure_access() {
    log "Verifying Azure access to Account 1 (Deployment)..."
    
    # Check if logged in
    if ! az account show &>/dev/null; then
        error "Not logged in to Azure. Run: az login"
    fi
    
    # Verify access to deployment subscription
    if ! az account show --subscription "$AZURE_SUBSCRIPTION_ID" &>/dev/null; then
        error "Cannot access subscription: $AZURE_SUBSCRIPTION_ID. Make sure you're logged in to Account 1."
    fi
    
    # Switch to deployment subscription
    log "Switching to deployment subscription: $AZURE_SUBSCRIPTION_ID"
    az account set --subscription "$AZURE_SUBSCRIPTION_ID"
    
    CURRENT_SUB=$(az account show --query name -o tsv)
    CURRENT_TENANT=$(az account show --query tenantId -o tsv)
    log "✅ Using subscription: $CURRENT_SUB (Tenant: ${CURRENT_TENANT:0:8}...)"
}

# Register required Azure resource providers
register_providers() {
    log "Checking Azure resource providers..."
    
    # Check if Microsoft.ContainerInstance is registered
    PROVIDER_STATE=$(az provider show --namespace Microsoft.ContainerInstance --query "registrationState" -o tsv 2>/dev/null || echo "NotRegistered")
    
    if [[ "$PROVIDER_STATE" != "Registered" ]]; then
        log "⚠️  Microsoft.ContainerInstance provider not registered"
        log "Registering Microsoft.ContainerInstance provider (this may take 1-2 minutes)..."
        
        az provider register --namespace Microsoft.ContainerInstance --output none
        
        # Wait for registration to complete
        log "Waiting for provider registration..."
        for i in {1..30}; do
            PROVIDER_STATE=$(az provider show --namespace Microsoft.ContainerInstance --query "registrationState" -o tsv 2>/dev/null)
            if [[ "$PROVIDER_STATE" == "Registered" ]]; then
                log "✅ Microsoft.ContainerInstance provider registered successfully"
                return 0
            fi
            echo -n "."
            sleep 5
        done
        
        error "Provider registration timed out. Please run manually: az provider register --namespace Microsoft.ContainerInstance"
    else
        log "✅ Microsoft.ContainerInstance provider already registered"
    fi
}

# Deploy containers
deploy_containers() {
    local api_image="$1"
    local data_pipeline_image="$2"
    local training_image="$3"
    
    log "Starting deployment to resource group: $RESOURCE_GROUP"
    
    # Create resource group if needed
    log "Ensuring resource group exists..."
    az group create \
        --name "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --tags "environment=$ENVIRONMENT" "project=churn-prediction" \
        --output none
    
    # Delete existing containers (optional - comment out to update instead)
    log "Checking for existing containers..."
    for container in "churn-api-$ENVIRONMENT" "churn-data-pipeline-$ENVIRONMENT" "churn-training-$ENVIRONMENT"; do
        if az container show --resource-group "$RESOURCE_GROUP" --name "$container" &>/dev/null; then
            log "Deleting existing container: $container"
            az container delete \
                --resource-group "$RESOURCE_GROUP" \
                --name "$container" \
                --yes \
                --output none || log "⚠️  Failed to delete $container (may not exist)"
        fi
    done
    
    # Prepare secure environment variables for Account 2 access
    SECURE_ENV_VARS="POSTGRES_PASSWORD=$POSTGRES_PASSWORD AUTH_SECRET=$AUTH_SECRET"
    
    # Add Account 2 service principal credentials if provided
    # Support both AZURE_CLIENT_ID and AZURE2_CLIENT_ID naming
    CLIENT_ID="${AZURE2_CLIENT_ID:-${AZURE_CLIENT_ID}}"
    CLIENT_SECRET="${AZURE2_CLIENT_SECRET:-${AZURE_CLIENT_SECRET}}"
    TENANT_ID="${AZURE2_TENANT_ID:-${AZURE_TENANT_ID}}"
    
    if [[ -n "${CLIENT_ID}" ]] && [[ -n "${CLIENT_SECRET}" ]] && [[ -n "${TENANT_ID}" ]]; then
        SECURE_ENV_VARS="$SECURE_ENV_VARS AZURE_CLIENT_ID=$CLIENT_ID AZURE_CLIENT_SECRET=$CLIENT_SECRET AZURE_TENANT_ID=$TENANT_ID"
        log "Adding Account 2 service principal credentials to containers"
    fi
    
    # Deploy API
    log "📦 Deploying API container..."
    az container create \
        --resource-group "$RESOURCE_GROUP" \
        --name "churn-api-$ENVIRONMENT" \
        --image "$api_image" \
        --cpu 1 \
        --memory 2 \
        --ports 8000 \
        --ip-address Public \
        --secure-environment-variables $SECURE_ENV_VARS \
        --environment-variables \
            POSTGRES_HOST="$POSTGRES_HOST" \
            POSTGRES_PORT="${POSTGRES_PORT:-5432}" \
            POSTGRES_DB="${POSTGRES_DB_NAME:-churn_db}" \
            POSTGRES_USER="${POSTGRES_DB_USER:-postgres}" \
            AZURE_ML_SUBSCRIPTION_ID="${AZURE2_SUBSCRIPTION_ID:-}" \
            AZURE_ML_RESOURCE_GROUP="${AZURE2_RESOURCE_GROUP:-}" \
            AZURE_ML_WORKSPACE_NAME="${AZURE2_ML_WORKSPACE_NAME:-}" \
            MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-}" \
            ENVIRONMENT="$ENVIRONMENT" \
            LOG_LEVEL="${LOG_LEVEL:-INFO}" \
        --dns-name-label "churn-api-${ENVIRONMENT}-$(date +%s)" \
        --restart-policy Always \
        --output table
    
    # Get API URL
    API_FQDN=$(az container show \
        --resource-group "$RESOURCE_GROUP" \
        --name "churn-api-$ENVIRONMENT" \
        --query "ipAddress.fqdn" -o tsv 2>/dev/null || echo "unavailable")
    
    log "✅ API deployed: http://${API_FQDN}:8000"
    
    # Deploy Data Pipeline (on-demand)
    log "📦 Deploying Data Pipeline container..."
    az container create \
        --resource-group "$RESOURCE_GROUP" \
        --name "churn-data-pipeline-$ENVIRONMENT" \
        --image "$data_pipeline_image" \
        --cpu 1 \
        --memory 1.5 \
        --secure-environment-variables \
            POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
        --environment-variables \
            POSTGRES_HOST="$POSTGRES_HOST" \
            POSTGRES_PORT="${POSTGRES_PORT:-5432}" \
            POSTGRES_DB="${POSTGRES_DB_NAME:-churn_db}" \
            POSTGRES_USER="${POSTGRES_DB_USER:-postgres}" \
            ENVIRONMENT="$ENVIRONMENT" \
            LOG_LEVEL="${LOG_LEVEL:-INFO}" \
        --restart-policy Never \
        --output table
    
    log "✅ Data Pipeline deployed (on-demand)"
    
    # Prepare secure env vars for training (includes Account 2 credentials)
    TRAINING_SECURE_ENV_VARS="POSTGRES_PASSWORD=$POSTGRES_PASSWORD"
    
    # Support both naming conventions
    CLIENT_ID="${AZURE2_CLIENT_ID:-${AZURE_CLIENT_ID}}"
    CLIENT_SECRET="${AZURE2_CLIENT_SECRET:-${AZURE_CLIENT_SECRET}}"
    TENANT_ID="${AZURE2_TENANT_ID:-${AZURE_TENANT_ID}}"
    
    if [[ -n "${CLIENT_ID}" ]] && [[ -n "${CLIENT_SECRET}" ]] && [[ -n "${TENANT_ID}" ]]; then
        TRAINING_SECURE_ENV_VARS="$TRAINING_SECURE_ENV_VARS AZURE_CLIENT_ID=$CLIENT_ID AZURE_CLIENT_SECRET=$CLIENT_SECRET AZURE_TENANT_ID=$TENANT_ID"
    fi
    
    # Deploy Training (on-demand)
    log "📦 Deploying Training container..."
    az container create \
        --resource-group "$RESOURCE_GROUP" \
        --name "churn-training-$ENVIRONMENT" \
        --image "$training_image" \
        --cpu 2 \
        --memory 4 \
        --secure-environment-variables $TRAINING_SECURE_ENV_VARS \
        --environment-variables \
            POSTGRES_HOST="$POSTGRES_HOST" \
            POSTGRES_PORT="${POSTGRES_PORT:-5432}" \
            POSTGRES_DB="${POSTGRES_DB_NAME:-churn_db}" \
            POSTGRES_USER="${POSTGRES_DB_USER:-postgres}" \
            AZURE_ML_SUBSCRIPTION_ID="${AZURE2_SUBSCRIPTION_ID:-}" \
            AZURE_ML_RESOURCE_GROUP="${AZURE2_RESOURCE_GROUP:-}" \
            AZURE_ML_WORKSPACE_NAME="${AZURE2_ML_WORKSPACE_NAME:-}" \
            MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-}" \
            ENVIRONMENT="$ENVIRONMENT" \
            LOG_LEVEL="${LOG_LEVEL:-INFO}" \
        --restart-policy Never \
        --output table
    
    log "✅ Training deployed (on-demand)"
}

main() {
    log "🚀 Starting Churn Prediction Deployment"
    log "Environment: $ENVIRONMENT"
    log "Note: This deploys to Account 1. Containers will access Account 2 using provided credentials."
    
    # Validate environment
    validate_env
    
    # Verify Azure access
    verify_azure_access
    
    # Register required providers
    register_providers
    
    # Deploy containers
    deploy_containers "$1" "$2" "$3"
    
    log ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "✅ Deployment completed successfully!"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "📊 Resource Group: $RESOURCE_GROUP"
    log "🌐 Location: $LOCATION"
    log "🔧 Environment: $ENVIRONMENT"
    
    if [[ -n "${AZURE2_ML_WORKSPACE_NAME:-}" ]]; then
        log "📈 ML Workspace (Account 2): $AZURE2_ML_WORKSPACE_NAME"
        if [[ -n "${AZURE2_CLIENT_ID:-}" ]]; then
            log "   Authentication: Service Principal"
        else
            log "   Authentication: DefaultAzureCredential"
        fi
    fi
    
    log ""
    log "Check deployment status:"
    log "  ./scripts/aci-management.sh status"
    log ""
    log "Test API health:"
    log "  ./scripts/aci-management.sh health"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Validate arguments
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <api-image> <data-pipeline-image> <training-image>"
    echo ""
    echo "Example:"
    echo "  $0 ghcr.io/org/api:latest ghcr.io/org/data:latest ghcr.io/org/training:latest"
    echo ""
    echo "Required environment variables (Account 1 - Deployment):"
    echo "  - AZURE_SUBSCRIPTION_ID"
    echo "  - POSTGRES_HOST"
    echo "  - POSTGRES_PASSWORD"
    echo "  - AZURE_STORAGE_CONNECTION_STRING"
    echo "  - AUTH_SECRET"
    echo ""
    echo "Optional (Account 2 - ML Tracking):"
    echo "  - AZURE2_SUBSCRIPTION_ID"
    echo "  - AZURE2_RESOURCE_GROUP"
    echo "  - AZURE2_ML_WORKSPACE_NAME"
    echo "  - MLFLOW_TRACKING_URI"
    echo ""
    echo "Optional (Account 2 - Service Principal Authentication):"
    echo "  - AZURE2_CLIENT_ID or AZURE_CLIENT_ID"
    echo "  - AZURE2_CLIENT_SECRET or AZURE_CLIENT_SECRET"
    echo "  - AZURE2_TENANT_ID or AZURE_TENANT_ID"
    echo ""
    echo "Note: Login to Account 1 first (az login)"
    exit 1
fi

main "$1" "$2" "$3"