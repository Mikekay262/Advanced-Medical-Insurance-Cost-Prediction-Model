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
  "randomForest", "xgboost", "caret", "DALEX", "DALEXtra", "plotly"
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

# Load trained models
load("linear_model.RData")
load("random_forest_model.RData")
load("xgboost_model.RData")

# Convert categorical variables to factors or dummy variables
train_data$smoker <- factor(train_data$smoker, levels = c("no", "yes"))
train_data$region <- factor(train_data$region, levels = c("northeast", "northwest", "southeast", "southwest"))


# Convert data to numeric (excluding target variable)
train_data_numeric <- data.frame(model.matrix(~ ., data = train_data)[, -1])  # Remove intercept column

# Create explainer
xgb_explainer <- explain_xgboost(
  model = xgb_best,
  data = as.matrix(train_data_numeric[, -which(names(train_data_numeric) == "charges")]), # Exclude target variable
  y = train_data$charges,
  label = "XGBoost"
)


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
  predict_cost <- reactive({
    req(input$predict)
    
    user_data <- data.frame(
      age = input$age,
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
                    "XGBoost" = xgb_best
    )
    
    prediction <- if (input$model_choice == "XGBoost") {
      predict(model, as.matrix(user_data))
    } else {
      predict(model, user_data)
    }
    return(prediction)
  })
  
  # Render prediction
  output$prediction <- renderText({
    cost <- predict_cost()
    paste("Predicted Medical Cost: $", round(cost, 2))
  })
  
  # Model comparison plot
  output$modelComparisonPlot <- renderPlotly({
    model_results <- data.frame(
      Model = c("Linear Regression", "Random Forest", "XGBoost"),
      RMSE = c(5408.64, 5189.68, 76.42)
    )
    p <- ggplot(model_results, aes(x = reorder(Model, RMSE), y = RMSE, fill = Model)) +
      geom_bar(stat = "identity") +
      coord_flip() +
      labs(title = "Model RMSE Comparison", x = "Model", y = "RMSE")
    ggplotly(p)
  })
  
  # SHAP visualizations
  output$shapPlot <- renderPlot({
    req(input$shap_type)
    if (input$shap_type == "Feature Importance") {
      plot(model_parts(xgb_explainer)) # SHAP feature importance
    } else if (input$shap_type == "Individual Explanation") {
      user_data <- data.frame(
        age = input$age,
        bmi = input$bmi,
        children = input$children,
        smoker = input$smoker,
        region = input$region
      )
      
      # Add engineered features
      user_data$smoker_bmi <- ifelse(user_data$smoker == "yes", user_data$bmi, 0)
      user_data$age_squared <- user_data$age^2
      
      shap_values <- predict_parts(xgb_explainer, new_observation = user_data)
      plot(shap_values)
    }
  })
}

# Run the app
shinyApp(ui = ui, server = server)
