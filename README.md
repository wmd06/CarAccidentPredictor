# Predicting Traffic Accidents

## Overview
Traffic accidents pose significant risks to public safety and infrastructure. This project leverages machine learning to predict the probability of traffic accidents using the **US Accidents Dataset**. By identifying key temporal, weather-related, and road condition features, the model provides insights to enhance road safety and inform targeted interventions.

## Motivation
Traffic accidents result in severe consequences, including loss of life, injuries, and economic impacts. Existing tools, like Waze and Google Maps, lack emphasis on localized risk factors, making machine learning a promising alternative for accident prediction. This project focuses on:
- Enhancing safety through proactive prediction.
- Addressing the limitations of existing accident prediction systems.

## Dataset
- **Source**: [US Accidents Dataset](https://www.kaggle.com/sobhanmoosavi/us-accidents)
- **Features**:
  - Temporal: Hour, month, day of the week.
  - Weather: Wind speed, precipitation, visibility, etc.
  - Road Conditions: Traffic signals, bumps, and junctions.
  - Location: Latitude and longitude.
- **Preprocessing**:
  - Missing values imputed.
  - Numerical features standardized.
  - Categorical weather conditions numerically encoded (e.g., Clear = 0, Rain = 3).
 
## Methodology
### Models and Techniques
1. **Neural Network**: Explored initially but rejected due to overfitting and interpretability challenges.
2. **Random Forest Classifier**:
   - Selected for robustness and interpretability.
   - Hyperparameters tuned using GridSearchCV:
     - Number of trees (`n_estimators`).
     - Maximum tree depth (`max_depth`).
     - Minimum samples for splits and leaf nodes.

### Addressing Class Imbalance
- **SMOTE**: Initially used to oversample minority classes but removed due to introducing synthetic noise.
- Final model excludes SMOTE, resulting in better generalization.

## Results
- **Random Forest Performance**:
  - Test Accuracy: **79.13%**
  - Train Accuracy: **83.27%**
  - ROC-AUC Score: **0.906**
  - Confusion Matrix: High true positive and negative rates.
- **Feature Importance**:
  - Temporal features (e.g., time of day) were the most influential.
  - Weather features (e.g., rain, fog) significantly contributed.
  - Road conditions (e.g., traffic signals) also played a role.
 
- **User Interface**:
    - ***Live Data Integration***
        A real-time user interface has been developed to fetch live traffic and weather data. This data is processed through the trained 
        Random Forest model to predict accident probabilities for specific locations and times. The interface integrates APIs for real- 
        time data, ensuring up-to-date predictions.

      ![1](https://github.com/user-attachments/assets/3312748a-0f10-4e52-841e-294afbf12c7e)
      
    - ***User Input***
        The interface allows users to input custom data points, including:
          1. Location: Latitude and longitude.
          2. Temporal Information: Time, day, and month.
          3. Weather Conditions: Temperature, wind speed, precipitation, and visibility.
          4. Road Features: Presence of traffic signals, bumps, and other road indicators.

      ![3](https://github.com/user-attachments/assets/3397eb6a-83d6-43c1-a505-4fdbb6974160)

    - ***Output Display***
          Predicted probabilities of traffic accidents are displayed on the interface, categorized into risk levels:
      
          - Low: Probability < 25%
      
          - Medium-Low: 25% ≤ Probability < 50%
      
          - Medium-High: 50% ≤ Probability < 75%
      
          - High: Probability ≥ 75%

      ![2](https://github.com/user-attachments/assets/306ffe4b-c5ad-4d7d-acf1-354e8d512045)
  
## Ethical Considerations
1. **Bias in Data**:
   - Geographic bias with urban overrepresentation.
   - Limited generalizability to rural settings.
2. **Unintended Consequences**:
   - Potential inequities in resource allocation.
   - Need for fairness audits and explainable AI techniques.

## Future Work
1. **Expand Dataset**:
   - Include additional sources for geographic and demographic diversity.
2. **Real-Time Prediction**:
   - Develop pipelines for live data integration.
3. **Advanced Models**:
   - Explore ensemble methods and interpretable ML techniques.
4. **Interactive Tools**:
   - Build user-friendly interfaces for policymakers and drivers.

## Contributions
- Demonstrated the utility of Random Forest for accident prediction.
- Insights into influential features can inform targeted interventions.
- Removed SMOTE for improved generalization and accuracy.

## Resources
- [US Accidents Dataset](https://www.kaggle.com/sobhanmoosavi/us-accidents)

## Authors
- Antoine Abou Faycal
- Mahdi Alhakim
- Wael Dgheim

