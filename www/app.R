# Function to ensure required libraries are installed
check_and_install <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE)) {
      install.packages(pkg, dependencies = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
}

# List of required libraries
required_packages <- c(
  "shiny", "shinydashboard", "shinyWidgets", "ggplot2",
  "randomForest", "xgboost", "caret", "DALEX", "DALEXtra", "plotly", "dplyr"
)

# Check and install missing libraries
check_and_install(required_packages)

# Load libraries
library(shiny)
library(shinydashboard)
library(shinyWidgets)
library(ggplot2)
library(randomForest)
library(xgboost)
library(caret)
library(DALEX)
library(DALEXtra)
library(plotly)
library(dplyr)

# Load trained models
model_path <- "./models/"  # Models should be in the 'models' directory
load(file.path(model_path, "linear_model.RData"))  # Loads `linear_model`
load(file.path(model_path, "random_forest_model.RData"))  # Loads `rf_optimized`
load(file.path(model_path, "xgboost_model.RData"))  # Loads `xgb_best`
dummy_model <- readRDS(file.path(model_path, "dummy_model.rds"))  # Loads dummy model

# Correct references to models and variables
linear_model <- lm_model
random_forest_model <- rf_optimized
xgboost_model <- xgb_best

# Prepare target variable
y_target <- train_data$charges

# Exclude the target variable for model input
train_data_features <- train_data
train_data_features$charges <- NULL

# Convert categorical variables to dummy/one-hot encoding
train_data_numeric <- model.matrix(~ . - 1, data = train_data_features)
train_data_numeric <- as.matrix(train_data_numeric) # Ensure it's a matrix

# Verify dimensions
if (nrow(train_data_numeric) != length(y_target)) {
  stop("Mismatch between feature matrix rows and target variable length.")
}

# Create XGBoost explainer
xgb_explainer <- explain_xgboost(
  model = xgboost_model,
  data = train_data_numeric,
  y = y_target,
  label = "XGBoost"
)

# Separate preprocessing pipelines
preprocess_data_xgb <- function(data, train_columns = xgb_best$feature_names) {
  # Encode 'smoker' as 1 for 'yes' and 0 for 'no'
  data <- data %>%
    mutate(
      smoker.yes = ifelse(smoker == "yes", 1, 0),       # Binary encode smoker
      smoker_bmi = ifelse(smoker == "yes", bmi, 0),    # Derived feature
      age_squared = age^2                              # Derived feature
    )
  
  # One-hot encode 'region' with explicit columns for consistency
  region_levels <- c("northwest", "southeast", "southwest")
  for (region in region_levels) {
    col_name <- paste0("region.", region)
    data[[col_name]] <- ifelse(data$region == region, 1, 0)
  }
  data <- data %>% select(-region)  # Drop original 'region'
  
  # Handle 'sex' column (binary encoding for male)
  if ("sex" %in% colnames(data)) {
    data <- data %>%
      mutate(sex.male = ifelse(sex == "male", 1, 0)) %>%
      select(-sex)  # Drop original 'sex'
  }
  
  # Ensure all required features are included
  if (!is.null(train_columns)) {
    # Add missing columns with default values of 0
    missing_cols <- setdiff(train_columns, colnames(data))
    if (length(missing_cols) > 0) {
      data[missing_cols] <- 0
    }
    
    # Remove extra columns not present in train_columns
    extra_cols <- setdiff(colnames(data), train_columns)
    if (length(extra_cols) > 0) {
      data <- data[, train_columns, drop = FALSE]
    }
  }
  
  # Ensure all columns are numeric
  data <- data %>% mutate(across(everything(), as.numeric))
  
  # Remove the target variable 'charges' for predictions if present
  data$charges <- NULL
  
  # Arrange columns in the order of `train_columns`
  data <- data[, train_columns[train_columns != "charges"], drop = FALSE]
  
  return(data)
}
preprocess_data_rf <- function(data) {
  # Add derived features
  data$smoker_bmi <- ifelse(data$smoker == "yes", data$bmi, 0)
  data$age_squared <- data$age^2
  
  # Ensure categorical variables are factors with consistent levels
  data$smoker <- factor(data$smoker, levels = c("yes", "no"))
  data$region <- factor(data$region, levels = c("northeast", "northwest", "southeast", "southwest"))
  
  # Random Forest works with raw data frames, no need for encoding
  return(data)
}

# Updated predict_cost function
predict_cost <- function(input_data, xgb_model, rf_model, lm_model, dummy_model, model_type = "xgboost") {
  # Preprocess input data based on model type
  if (model_type == "xgboost") {
    preprocessed_data <- preprocess_data_xgb(input_data)
  } else if (model_type == "random_forest") {
    preprocessed_data <- preprocess_data_rf(input_data)
  } else if (model_type == "dummy") {
    preprocessed_data <- preprocess_data_xgb(input_data)
  } else {
    preprocessed_data <- input_data
  }
  
  # Predict using the selected model
  if (model_type == "xgboost") {
    prediction <- predict(xgb_model, as.matrix(preprocessed_data))
  } else if (model_type == "random_forest") {
    prediction <- predict(rf_model, preprocessed_data)
  } else if (model_type == "linear_model") {
    prediction <- predict(lm_model, input_data)
  } else if (model_type == "dummy") {
    prediction <- predict(dummy_model, newdata = preprocessed_data)
  } else {
    stop("Invalid model type. Choose either 'xgboost', 'random_forest', 'linear_model', or 'dummy'.")
  }
  
  return(prediction)
}


# UI Section
ui <- dashboardPage(
  dashboardHeader(title = "Medical Insurance Cost Prediction"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Home", tabName = "home", icon = icon("home")),
      menuItem("Predict Costs", tabName = "predict", icon = icon("calculator")),
      menuItem("Model Comparison", tabName = "comparison", icon = icon("chart-bar")),
      menuItem("SHAP Visualizations", tabName = "shap", icon = icon("chart-area")),
      menuItem("About", tabName = "about", icon = icon("info-circle"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(
        tabName = "home",
        fluidRow(
          box(
            title = "Welcome to the Medical Insurance Cost Prediction App",
            width = 12,
            status = "primary",
            solidHeader = TRUE,
            p("This app predicts medical costs using advanced machine learning models."),
            p("Explore features like model performance comparison, batch predictions, and explainable AI insights using SHAP visualizations."),
            p("Author: Dr. Michael Adu"),
            p("Email: mikekay262@gmail.com"),
            p("LinkedIn: ", a("Dr. Michael Adu", href = "https://www.linkedin.com/in/drmichael-adu"))
          )
        )
      ),
      tabItem(
        tabName = "predict",
        fluidRow(
          box(
            title = "Input Patient Details",
            width = 6,
            status = "info",
            numericInput("age", "Age:", value = 30, min = 18, max = 64, step = 1),
            selectInput("sex", "Sex:", choices = c("male", "female")),
            numericInput("bmi", "BMI:", value = 25, min = 15, max = 55, step = 0.1),
            numericInput("children", "Number of Children:", value = 0, min = 0, max = 5, step = 1),
            selectInput("smoker", "Smoker:", choices = c("yes", "no")),
            selectInput("region", "Region:", choices = c("northeast", "northwest", "southeast", "southwest")),
            pickerInput("model_choice", "Select Model:", choices = c("Linear Regression", "Random Forest", "XGBoost"), selected = "XGBoost"),
            actionButton("predict", "Predict")
          ),
          box(
            title = "Prediction Results",
            width = 6,
            status = "success",
            solidHeader = TRUE,
            verbatimTextOutput("prediction")
          )
        )
      ),
      tabItem(
        tabName = "comparison",
        fluidRow(
          box(
            title = "Model Performance Comparison",
            width = 12,
            status = "warning",
            plotlyOutput("modelComparisonPlot", height = "500px")
          )
        )
      ),
      tabItem(
        tabName = "shap",
        fluidRow(
          box(
            title = "SHAP Visualizations",
            width = 12,
            status = "info",
            selectInput("shap_type", "Select Visualization Type:", choices = c("Feature Importance", "Individual Explanation")),
            plotOutput("shapPlot", height = "500px")
          )
        )
      ),
      tabItem(
        tabName = "about",
        fluidRow(
          box(
            title = "About the Project",
            width = 12,
            status = "info",
            solidHeader = TRUE,
            p("This project demonstrates the use of machine learning models to predict medical costs."),
            p("Features advanced regression techniques like Linear Regression, Random Forest, and XGBoost."),
            p("Explainability tools ensure the predictions are interpretable and actionable."),
            p("Contact: Dr. Michael Adu"),
            p("Email: mikekay262@gmail.com"),
            p("LinkedIn: ", a("Dr. Michael Adu", href = "https://www.linkedin.com/in/drmichael-adu"))
          )
        )
      )
    )
  )
)

# Server Section
server <- function(input, output, session) {
  # Predict function
  predict_cost_output <- reactive({
    req(input$predict)
    
    user_data <- data.frame(
      age = input$age,
      sex = input$sex,
      bmi = input$bmi,
      children = input$children,
      smoker = input$smoker,
      region = input$region
    )
    
    # Add engineered features
    user_data$smoker_bmi <- ifelse(user_data$smoker == "yes", user_data$bmi, 0)
    user_data$age_squared <- user_data$age^2
    
    model <- switch(input$model_choice,
                    "Linear Regression" = linear_model,
                    "Random Forest" = random_forest_model,
                    "XGBoost" = xgboost_model
    )
    
    # Map model choice to model_type for predict_cost function
    model_type <- switch(input$model_choice,
                         "Linear Regression" = "linear_model",
                         "Random Forest" = "random_forest",
                         "XGBoost" = "xgboost")
    
    # Corrected function call: Removed encoder argument
    prediction <- predict_cost(user_data,
                               xgb_model = xgboost_model, 
                               rf_model = random_forest_model, 
                               lm_model = linear_model, 
                               model_type = model_type)
    return(prediction)
  })
  
  # Render prediction
  output$prediction <- renderText({
    cost <- predict_cost_output()
    paste("Predicted Medical Cost: $", round(cost, 2))
  })
}

# Run the app
shinyApp(ui = ui, server = server)
