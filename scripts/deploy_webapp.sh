#!/bin/bash

# Print a message
echo "Deploying the web application..."

# Apply the deployment configuration from the deployment.yaml file
kubectl apply -f ./k8s/deployment.yaml

# Restart the deployment to use the latest Docker image, if updated
echo "Restarting the deployment to ensure the latest image is used..."
kubectl rollout restart deployment/webapp-photo2monet

# Provide feedback that the deployment has been triggered
echo "Deployment has been updated. Check the status with 'kubectl get pods' to ensure pods are running."
