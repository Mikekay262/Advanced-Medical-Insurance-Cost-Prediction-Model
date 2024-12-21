# Load the XGBoost model
load("xgboost_model.Rdata")  # Ensure you are in the correct directory

# Check feature names the model expects
feature_names <- xgboost_model$feature_names
print(feature_names)

