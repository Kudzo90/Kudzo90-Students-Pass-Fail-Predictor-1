# Kudzo90-Students-Pass-Fail-Predictor-1
# 🚨 Trajectory-Based Early Warning System

## Student Pass/Fail Prediction System

A machine learning-powered early warning system that predicts student pass/fail outcomes based on historical performance trajectories in Mathematical Literacy. This system enables educators to identify at-risk students early in the academic year, providing crucial time for intervention and support.
**For ethical reasons, no real students' data is included in this repo; users should supply their own de-identified/synthetic data**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Prediction Checkpoints](#prediction-checkpoints)
- [Understanding the Results](#understanding-the-results)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This system addresses a critical challenge in education: **identifying struggling students early enough to make a difference**. Unlike traditional approaches that wait until it's too late, this trajectory-based system learns from historical patterns to predict outcomes at multiple stages throughout the academic year.

### The Problem
- Traditional grading systems only identify failing students at the end of the year
- By then, it's often too late for effective intervention
- Educators need early warning signals to provide timely support

### The Solution
- **Trajectory-based predictions**: Learn from historical patterns of student performance
- **Multiple checkpoints**: Make predictions after T1, T2, T3, June Exam, T5, T6, and T7
- **Realistic predictions**: Based on actual student trajectories, not unrealistic cumulative thresholds
- **Actionable insights**: Identify at-risk students with time to intervene

---

## ✨ Key Features

### 🔮 Progressive Predictions
- Make predictions at **7 different checkpoints** throughout the academic year
- Earlier predictions = more time for intervention
- Each checkpoint uses only the data available at that stage

### 📊 Trajectory Learning
- Models learn from historical patterns: *"Students who scored X, Y, Z at this stage typically ended up passing/failing"*
- No unrealistic expectations (e.g., expecting 30% cumulative after just 3 tasks)
- Based on real student performance trajectories

### 🎯 Advanced Feature Engineering
- **Statistical features**: Average, min, max, range, standard deviation
- **Trend analysis**: Is performance improving or declining?
- **Consistency metrics**: How stable is the student's performance?
- **On-track indicators**: Is cumulative performance proportional to expectations?

### 🤖 Multiple ML Models
- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression
- Automatic selection of best-performing model for each checkpoint

### ⚖️ Class Imbalance Handling
- SMOTE (Synthetic Minority Over-sampling Technique)
- Balanced class weights
- Optimized for detecting at-risk students (high fail recall)

### 📈 Interactive Dashboard
- Single student predictions with detailed analysis
- Bulk upload for analyzing entire classes
- Performance comparison across checkpoints
- Visual analytics and downloadable reports

---

## 🔬 How It Works

### 1. **Training Phase**

The system learns from historical student data:

```
Historical Data → Feature Engineering → Model Training → Validation → Save Models
```

**Input**: Excel file with columns:
- `T1 %`, `T2 %`, `T3 %`, `June Exam %`, `T5 %`, `T6 %`, `T7 %`
- `EXAM MARK` (final exam mark used to determine pass/fail)

**Process**:
1. Calculate Year Mark using task weights (10%, 10%, 10%, 20%, 10%, 10%, 30%)
2. Create Pass/Fail target (Fail if EXAM MARK < 30%, Pass otherwise)
3. For each checkpoint, create features from available tasks
4. Train separate models for each checkpoint
5. Handle class imbalance using SMOTE
6. Select best-performing model based on fail recall
7. Save all models and metadata

**Output**: 
- `trajectory_models.pkl` (trained models for all checkpoints)
- `model_metadata.pkl` (model configuration and performance metrics)

### 2. **Prediction Phase**

The system predicts outcomes for new students:

```
Student Marks → Feature Creation → Model Prediction → Risk Assessment → Recommendations
```

**Input**: Student marks for completed tasks

**Process**:
1. Select appropriate checkpoint based on available data
2. Create features (statistical, trend, consistency)
3. Load corresponding trained model
4. Predict pass/fail outcome with probability
5. Assess risk level (High/Medium/Low)
6. Generate recommendations

**Output**:
- Prediction (Pass/Fail)
- Pass probability (0-100%)
- Risk level
- Key contributing factors
- Intervention recommendations

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download

```bash
# If using git
git clone <repository-url>
cd early-warning-system

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn streamlit plotly openpyxl
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Step 3: Prepare Your Data

Ensure your training data Excel file has the following columns:
- `T1 %` (Task 1 percentage)
- `T2 %` (Task 2 percentage)
- `T3 %` (Task 3 percentage)
- `June Exam %` (June Exam/T4 percentage)
- `T5 %` (Task 5 percentage)
- `T6 %` (Task 6 percentage)
- `T7 %` (Task 7 percentage)
- `EXAM MARK` (Final exam mark - used to determine pass/fail)

**Note**: Missing values will be filled with 0 (consistent with the assumption that incomplete work = 0%)

---

## 📖 Usage

### Training the Models

1. **Prepare your training data**:
   - Ensure your Excel file is named `Marks for Prediction.xlsx`
   - Place it in the same directory as the training script
   - Verify all required columns are present

2. **Run the training script**:

```bash
python train_trajectory_model.py
```

3. **Review the output**:
   - The script will display performance metrics for each checkpoint
   - You'll see a comparison table showing accuracy, fail recall, and AUC-ROC
   - The system will recommend the best early warning checkpoint
   - Models will be saved automatically

4. **Expected output files**:
   - `trajectory_models.pkl` (all trained models)
   - `model_metadata.pkl` (configuration and performance data)

### Running the Streamlit App

1. **Ensure models are trained**:
   - Verify `trajectory_models.pkl` and `model_metadata.pkl` exist

2. **Launch the app**:

```bash
streamlit run app.py
```

3. **Access the dashboard**:
   - The app will open in your default browser
   - Default URL: `http://localhost:8501`

### Using the Dashboard

#### Tab 1: Single Student Prediction

1. **Select assessment stage**: Choose when you want to make the prediction (e.g., "After T3")
2. **Enter student marks**: Fill in the marks for completed tasks
3. **Click "Predict Trajectory"**: Get instant prediction with detailed analysis
4. **Review results**:
   - Prediction (Pass/Fail)
   - Pass likelihood percentage
   - Performance analysis (average, trend, consistency)
   - Key contributing factors
   - Intervention recommendations

#### Tab 2: Bulk Upload & Analysis

1. **Select assessment stage**: Choose the checkpoint for bulk analysis
2. **Upload Excel file**: Must contain required columns for selected checkpoint
3. **Review dashboard**:
   - Total students analyzed
   - Number of at-risk students
   - Risk distribution charts
   - Pass likelihood distribution
4. **Download results**:
   - All predictions (CSV)
   - At-risk students only (CSV)
   - High-risk students only (CSV)

#### Tab 3: System Performance

1. **View performance metrics**: Compare accuracy across all checkpoints
2. **Analyze trends**: See how prediction quality improves over time
3. **Review recommendations**: Identify the optimal early warning checkpoint
4. **Understand reliability**: Check AUC-ROC scores and detection rates

---

## 📁 Project Structure

```
early-warning-system/
│
├── train_trajectory_model.py      # Training script
├── app.py                          # Streamlit dashboard
├── Marks for Prediction.xlsx      # Training data (your file)
├── trajectory_models.pkl           # Trained models (generated)
├── model_metadata.pkl              # Model metadata (generated)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
└── (optional folders)
    ├── data/                       # Additional datasets
    ├── exports/                    # Downloaded predictions
    └── docs/                       # Additional documentation
```

---

## 🧠 Model Architecture

### Feature Engineering

For each checkpoint, the system creates the following features:

#### 1. **Raw Task Scores**
- Individual task percentages (T1, T2, T3, etc.)

#### 2. **Statistical Features**
- **Average**: Mean of all completed tasks
- **Min**: Lowest task score
- **Max**: Highest task score
- **Std**: Standard deviation (consistency measure)
- **Range**: Difference between max and min

#### 3. **Trend Features** (if ≥2 tasks available)
- **Trend**: Difference between last and first task
- **Improving**: Binary indicator (1 if improving, 0 if declining)

#### 4. **Consistency Features** (if ≥3 tasks available)
- **Consistency**: 100 - Standard Deviation

#### 5. **Progress Features**
- **Cumulative_YearMark**: Weighted sum of completed tasks
- **On_Track**: Binary indicator (1 if on track, 0 if behind)

### Model Selection

For each checkpoint, the system trains three models:

1. **Random Forest Classifier**
   - Ensemble of decision trees
   - Good for non-linear relationships
   - Provides feature importance

2. **Gradient Boosting Classifier**
   - Sequential ensemble method
   - Often highest accuracy
   - Robust to overfitting

3. **Logistic Regression**
   - Linear model
   - Fast and interpretable
   - Good baseline

**Selection Criteria**: The model with the highest **fail recall** (ability to detect at-risk students) is selected for each checkpoint.

### Class Imbalance Handling

Since most students pass (class imbalance), the system uses:

- **SMOTE**: Creates synthetic examples of failing students
- **Balanced class weights**: Penalizes misclassification of minority class more heavily
- **Optimized threshold**: Adjusts decision boundary to favor detecting at-risk students

---

## 📍 Prediction Checkpoints

| Checkpoint | Available Tasks | Typical Fail Recall | Best Use Case |
|------------|----------------|---------------------|---------------|
| **After T1** | T1 | 10-20% | Very early signal, limited reliability |
| **After T2** | T1, T2 | 15-30% | Early signal, improving reliability |
| **After T3** | T1, T2, T3 | 25-45% | **First meaningful early warning** ⚠️ |
| **After T4 (June)** | T1, T2, T3, June | 40-65% | **Recommended checkpoint** 🎯 |
| **After T5** | T1-T3, June, T5 | 50-75% | Late warning, high reliability |
| **After T6** | T1-T3, June, T5, T6 | 55-80% | Very late, very reliable |
| **After T7** | All tasks | 60-85% | Final confirmation, limited intervention time |

### Recommended Strategy

1. **After T3**: First screening to identify high-risk students
2. **After T4 (June Exam)**: Primary checkpoint for intervention planning
3. **After T5**: Follow-up assessment for students receiving support
4. **After T7**: Final check before external exam

---

## 📊 Understanding the Results

### Prediction Output

When you make a prediction, you'll receive:

#### 1. **Primary Prediction**
- **Pass** or **Fail** classification
- Based on learned trajectories from historical data

#### 2. **Pass Likelihood**
- Probability (0-100%) that the student will pass
- Higher = more confident in pass prediction
- Lower = higher risk of failure

#### 3. **Risk Level**
- **High Risk**: Pass likelihood < 30%
- **Medium Risk**: Pass likelihood 30-70%
- **Low Risk**: Pass likelihood > 70%

#### 4. **Performance Analysis**
- **Current Average**: Mean of completed tasks
- **Cumulative Year Mark**: Weighted sum of completed tasks
- **Performance Range**: Difference between best and worst task
- **Trend**: Whether performance is improving or declining

#### 5. **On Track Indicator**
- ✅ **On Track**: Cumulative performance is proportional to expectations
- ⚠️ **Below Track**: Cumulative performance is below expectations

#### 6. **Feature Importance**
- Shows which factors most influenced the prediction
- Helps understand what's driving the risk assessment

### Model Performance Metrics

#### Accuracy
- Percentage of correct predictions (both pass and fail)
- **Good**: > 70%
- **Excellent**: > 80%

#### Fail Recall (At-Risk Detection Rate)
- Percentage of failing students correctly identified
- **Most important metric** for early warning systems
- **Good**: > 40%
- **Excellent**: > 60%

#### Pass Recall
- Percentage of passing students correctly identified
- Should be high (> 80%) to avoid unnecessary interventions

#### AUC-ROC (Area Under Curve)
- Measures model's ability to discriminate between pass and fail
- **Random guess**: 0.5
- **Good**: > 0.7
- **Excellent**: > 0.8

#### False Alarms
- Passing students incorrectly flagged as at-risk
- Should be minimized to avoid wasting resources

---

## 💡 Best Practices

### For Educators

1. **Use Multiple Checkpoints**
   - Don't rely on a single prediction
   - Track students across multiple checkpoints
   - Look for consistent patterns

2. **Focus on High-Risk Students**
   - Prioritize students with pass likelihood < 30%
   - These students need immediate intervention

3. **Monitor Trends**
   - Pay attention to improving vs. declining trends
   - Declining trends are red flags even if current average is acceptable

4. **Combine with Qualitative Data**
   - Use predictions as one input, not the only input
   - Consider attendance, engagement, personal circumstances

5. **Intervene Early**
   - The earlier you intervene, the better the outcomes
   - Use "After T3" or "After T4" as primary checkpoints

### For Administrators

1. **Regular Model Retraining**
   - Retrain models annually with updated data
   - Performance improves as more historical data accumulates

2. **Monitor System Performance**
   - Track actual outcomes vs. predictions
   - Adjust intervention strategies based on results

3. **Resource Allocation**
   - Use bulk analysis to plan support resources
   - Identify classes or cohorts with high at-risk percentages

4. **Ethical Considerations**
   - Predictions are probabilities, not certainties
   - Avoid labeling or stigmatizing students
   - Use predictions to provide support, not to punish

### For Data Scientists

1. **Feature Engineering**
   - Experiment with additional features (e.g., task difficulty, time to completion)
   - Consider domain-specific knowledge from educators

2. **Model Tuning**
   - Adjust hyperparameters for your specific dataset
   - Balance fail recall vs. false alarm rate based on institutional priorities

3. **Threshold Optimization**
   - Experiment with different classification thresholds
   - Use precision-recall curves to find optimal balance

4. **Validation**
   - Use cross-validation for more robust performance estimates
   - Test on multiple cohorts to ensure generalizability

---

## 🔧 Troubleshooting

### Common Issues

#### Issue: "KeyError: 0" during training

**Cause**: Index mismatch when using SMOTE with pandas Series

**Solution**: The updated code uses numpy arrays to avoid this issue. Ensure you're using the latest version of the training script.

#### Issue: "Missing required columns" when uploading file

**Cause**: Excel file doesn't have the required column names

**Solution**: 
- Ensure column names exactly match: `T1 %`, `T2 %`, etc.
- Check for extra spaces or special characters
- Verify column names are in the first row

#### Issue: Low fail recall (< 20%)

**Cause**: Severe class imbalance or insufficient training data

**Solutions**:
- Collect more historical data, especially failing students
- Adjust SMOTE parameters (increase sampling ratio)
- Try different classification thresholds
- Use ensemble methods with stronger class weights

#### Issue: High false alarm rate

**Cause**: Model is too conservative (over-predicting failures)

**Solutions**:
- Adjust classification threshold (increase from 0.5)
- Reduce SMOTE over-sampling
- Use precision-recall curve to find optimal balance

#### Issue: Streamlit app won't start

**Cause**: Missing dependencies or model files

**Solutions**:
```bash
# Reinstall dependencies
pip install --upgrade streamlit pandas numpy scikit-learn plotly

# Verify model files exist
ls -la trajectory_models.pkl model_metadata.pkl

# If missing, retrain models
python train_trajectory_model.py
```

#### Issue: Predictions seem unrealistic

**Cause**: Model trained on different data distribution

**Solutions**:
- Retrain with current cohort data
- Verify training data quality (no data entry errors)
- Check that task weights match your grading scheme
- Ensure EXAM MARK is correctly calculated

---

## 🔄 Updating the System

### Adding New Training Data

1. **Append new student records** to your Excel file
2. **Retrain the models**:
```bash
python train_trajectory_model.py
```
3. **Restart the Streamlit app** to load new models

### Modifying Task Weights

If your grading scheme changes, update the weights in both files:

**In `train_trajectory_model.py`**:
```python
df['YEAR MARK'] = (
    df['T1 %'] * 0.10 +      # Change these values
    df['T2 %'] * 0.10 +
    df['T3 %'] * 0.10 +
    df['June Exam %'] * 0.20 +
    df['T5 %'] * 0.10 +
    df['T6 %'] * 0.10 +
    df['T7 %'] * 0.30
)
```

**In `app.py`**:
```python
TASK_WEIGHTS = {
    'T1 %': 0.10,           # Change these values
    'T2 %': 0.10,
    'T3 %': 0.10,
    'June Exam %': 0.20,
    'T5 %': 0.10,
    'T6 %': 0.10,
    'T7 %': 0.30
}
```

Then retrain the models.

### Changing Pass Threshold

If your institution uses a different pass threshold (e.g., 40% instead of 30%):

**In `train_trajectory_model.py`**:
```python
df['Pass/Fail'] = df['EXAM MARK'].apply(lambda x: 'Fail' if x < 40 else 'Pass')  # Change 30 to 40
```

Then retrain the models.

---

## 📈 Performance Optimization

### For Large Datasets (> 10,000 students)

1. **Use sampling for SMOTE**:
```python
smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.5)  # Reduce synthetic samples
```

2. **Reduce model complexity**:
```python
RandomForestClassifier(n_estimators=50, max_depth=8)  # Fewer trees, shallower depth
```

3. **Use parallel processing**:
```python
RandomForestClassifier(n_jobs=-1)  # Use all CPU cores
```

### For Real-Time Predictions

1. **Cache models in memory** (already implemented in Streamlit app)
2. **Pre-compute features** for bulk predictions
3. **Use simpler models** (Logistic Regression) for faster inference

---

## 🤝 Contributing

Contributions are welcome! Here are some ways you can help:

### Feature Requests
- Additional prediction checkpoints
- Integration with student information systems
- Mobile app version
- Email alerts for at-risk students

### Bug Reports
- Use GitHub Issues to report bugs
- Include error messages and steps to reproduce

### Code Contributions
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/) for machine learning
- [imbalanced-learn](https://imbalanced-learn.org/) for handling class imbalance
- [Streamlit](https://streamlit.io/) for the interactive dashboard
- [Plotly](https://plotly.com/) for visualizations

---

## 📞 Support

For questions, issues, or suggestions:

- **Email**: [wondere@mthashana.edu.za]/[jewk90@yahoo.co.za]
- **GitHub Issues**: [repository-url/issues]


---

## 🎓 Citation

If you use this system in your research or institution, please cite:

```
Early Warning System for Student Performance Prediction
Trajectory-Based Machine Learning Approach
[WK Ekpe/Mthashana TVET College], [2025]
```

---

## 🔮 Future Enhancements

Planned features for future versions:

- [ ] Integration with Learning Management Systems (LMS)
- [ ] Automated email alerts for at-risk students
- [ ] Mobile app for on-the-go predictions
- [ ] Advanced visualizations (student trajectories over time)
- [ ] Recommendation engine for personalized interventions
- [ ] Multi-subject support
- [ ] API for integration with other systems
- [ ] A/B testing framework for intervention strategies
- [ ] Explainable AI features (SHAP values, LIME)
- [ ] Time-series forecasting for future task performance

---

## 📚 Additional Resources

### Understanding the Metrics
- [Precision, Recall, and F1 Score Explained](https://en.wikipedia.org/wiki/Precision_and_recall)
- [ROC Curves and AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Handling Imbalanced Datasets](https://imbalanced-learn.org/stable/introduction.html)

### Machine Learning Concepts
- [Random Forests](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

### Educational Data Mining
- [Educational Data Mining Community](https://educationaldatamining.org/)
- [Learning Analytics](https://en.wikipedia.org/wiki/Learning_analytics)

---

## ⚠️ Important Notes

### Data Privacy
- Ensure compliance with data protection regulations (GDPR, FERPA, etc.)
- Anonymize student data when sharing or publishing
- Secure storage of sensitive information
- Obtain necessary permissions before collecting/using student data

### Ethical Considerations
- Predictions are **probabilities**, not certainties
- Use predictions to **support** students, not to label or limit them
- Combine algorithmic predictions with human judgment
- Be transparent with students about how the system works
- Regularly audit for bias and fairness
- **Example data not included; users should supply their own de-identified/synthetic data**

### Limitations
- System performance depends on quality and quantity of training data
- Predictions are based on historical patterns (may not account for unprecedented situations)
- Cannot capture all factors affecting student performance (personal circumstances, motivation, etc.)
- Should be used as one tool among many, not the sole decision-making criterion

---

## 🎯 Quick Start Checklist

- [ ] Install Python 3.8+
- [ ] Install required packages (`pip install -r requirements.txt`)
- [ ] Prepare training data Excel file with required columns
- [ ] Run training script (`python train_trajectory_model.py`)
- [ ] Verify model files are created (`trajectory_models.pkl`, `model_metadata.pkl`)
- [ ] Launch Streamlit app (`streamlit run app.py`)
- [ ] Test with sample student data
- [ ] Review performance metrics
- [ ] Identify recommended checkpoint
- [ ] Begin using for early warning predictions!

---

**Version**: 1.0.0  
**Last Updated**: 2025  
**Status**: Production Ready ✅

---

*Built with ❤️ for educators who want to make a difference in students' lives*
