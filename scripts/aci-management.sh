set -e

RESOURCE_GROUP="churn-prediction-rg"
ENVIRONMENT="${ENVIRONMENT:-production}"

case "$1" in
    "status")
        az container list \
            --resource-group "$RESOURCE_GROUP" \
            --query "sort_by([].{Name:name, Status:instanceView.state, IP:ipAddress.ip, FQDN:ipAddress.fqdn}, &Name)" \
            --output table
        ;;
    "logs")
        CONTAINER_NAME="${2:-churn-api-$ENVIRONMENT}"
        az container logs \
            --resource-group "$RESOURCE_GROUP" \
            --name "$CONTAINER_NAME"
        ;;
    "health")
        API_URL=$(az container show \
            --resource-group "$RESOURCE_GROUP" \
            --name "churn-api-$ENVIRONMENT" \
            --query "ipAddress.fqdn" -o tsv)
        echo "Testing API health: http://${API_URL}:8000/health"
        curl -f "http://${API_URL}:8000/health" || echo "Health check failed"
        ;;
    "delete")
        az container delete \
            --resource-group "$RESOURCE_GROUP" \
            --name "${2:-churn-api-$ENVIRONMENT}" \
            --yes
        ;;
        *)
        echo "Usage: $0 {status|logs|health|delete} [container-name]"
        echo "Examples:"
        echo "  $0 status"
        echo "  $0 logs churn-api-production"
        echo "  $0 health"
        echo "  $0 delete churn-api-production"
        ;;
esac