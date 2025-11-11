import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load the trained models
@st.cache_resource
def load_models():
    with open("trajectory_models.pkl", "rb") as file:
        trajectory_models = pickle.load(file)
    with open("model_metadata.pkl", "rb") as file:
        metadata = pickle.load(file)
    return trajectory_models, metadata

trajectory_models, metadata = load_models()

# Task weightings
TASK_WEIGHTS = metadata['task_weights']
TASK_COLUMNS = metadata['task_columns']
CHECKPOINTS = metadata['checkpoints']
CHECKPOINT_RESULTS = metadata['checkpoint_results']

# Function to create features for a checkpoint
def create_features_for_checkpoint(input_data, available_tasks):
    """
    Create meaningful features from available tasks
    """
    features = {}
    
    # Original task values
    for task in available_tasks:
        features[task] = input_data[task].values[0]
    
    # Statistical features
    task_values = [input_data[task].values[0] for task in available_tasks]
    features['Average'] = np.mean(task_values)
    features['Min'] = np.min(task_values)
    features['Max'] = np.max(task_values)
    features['Std'] = np.std(task_values) if len(task_values) > 1 else 0
    features['Range'] = features['Max'] - features['Min']
    
    # Trend features (if we have at least 2 tasks)
    if len(available_tasks) >= 2:
        first_task = input_data[available_tasks[0]].values[0]
        last_task = input_data[available_tasks[-1]].values[0]
        features['Trend'] = last_task - first_task
        features['Improving'] = 1 if features['Trend'] > 0 else 0
    
    # Consistency features
    if len(available_tasks) >= 3:
        features['Consistency'] = 100 - features['Std']
    
    # Cumulative partial year mark
    cumulative_mark = 0
    for task in available_tasks:
        cumulative_mark += input_data[task].values[0] * TASK_WEIGHTS[task]
    features['Cumulative_YearMark'] = cumulative_mark
    
    # On track indicator
    total_weight_so_far = sum(TASK_WEIGHTS[task] for task in available_tasks)
    expected_contribution = 30 * total_weight_so_far
    features['On_Track'] = 1 if cumulative_mark >= expected_contribution else 0
    
    return pd.DataFrame([features])

# Function to predict at a specific checkpoint
def predict_at_checkpoint(input_data, checkpoint_name):
    """
    Predict student outcome based on trajectory at a specific checkpoint
    """
    checkpoint_info = trajectory_models[checkpoint_name]
    model = checkpoint_info['model']
    available_tasks = checkpoint_info['available_tasks']
    feature_names = checkpoint_info['feature_names']
    
    # Create features
    X_features = create_features_for_checkpoint(input_data, available_tasks)
    
    # Ensure features are in the correct order
    X_features = X_features[feature_names]
    
    # Predict
    prediction = model.predict(X_features)[0]
    prediction_proba = model.predict_proba(X_features)[0]
    
    pass_probability = prediction_proba[1]  # Probability of passing
    fail_probability = prediction_proba[0]  # Probability of failing
    
    # Get feature importance if available
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(feature_names, model.feature_importances_))
    
    return {
        'prediction': 'Pass' if prediction == 1 else 'Fail',
        'pass_probability': pass_probability,
        'fail_probability': fail_probability,
        'available_tasks': available_tasks,
        'features': X_features.to_dict('records')[0],
        'feature_importance': feature_importance,
        'model_name': checkpoint_info['model_name']
    }

# Streamlit App
st.set_page_config(page_title="Early Warning System", page_icon="🚨", layout="wide")

st.title("🚨 Trajectory-Based Early Warning System")
st.markdown("### Predict student outcomes based on historical performance patterns")

# Sidebar - System Information
with st.sidebar:
    st.header("📊 About This System")
    st.markdown("""
    ### How It Works:
    
    This system learns from **historical trajectories**:
    
    *"Students who scored similar marks at this stage typically ended up passing/failing"*
    
    ### Prediction Checkpoints:
    
    - **After T1** (Very early signal)
    - **After T2** (Early signal)
    - **After T3** (Early warning) ⚠️
    - **After T4/June** (Strong signal) 🎯
    - **After T5** (Late warning)
    - **After T6** (Very late)
    - **After T7** (Final confirmation)
    
    ✅ **Key Advantage:** Predictions are based on realistic patterns, not impossible cumulative thresholds!
    """)
    
    st.markdown("---")
    st.markdown("### 📈 System Performance")
    
    # Show performance summary
    perf_data = []
    for cp_name, results in CHECKPOINT_RESULTS.items():
        perf_data.append({
            'Checkpoint': cp_name.replace('After ', ''),
            'At-Risk Detection': results['fail_recall'],
            'Accuracy': results['accuracy']
        })
    
    perf_df = pd.DataFrame(perf_data)
    
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=perf_df['Checkpoint'],
        y=perf_df['At-Risk Detection'] * 100,
        mode='lines+markers',
        name='At-Risk Detection',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))
    fig_perf.add_trace(go.Scatter(
        x=perf_df['Checkpoint'],
        y=perf_df['Accuracy'] * 100,
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='green', width=2),
        marker=dict(size=8)
    ))
    fig_perf.update_layout(
        title='Detection Performance',
        xaxis_title='',
        yaxis_title='Percentage (%)',
        height=300,
        showlegend=True,
        xaxis={'tickangle': -45},
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # Recommended checkpoint
    st.markdown("---")
    st.success(f"**Recommended:** {metadata['recommended_checkpoint']}")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📝 Single Student Prediction", "📊 Bulk Upload & Analysis", "📈 System Performance"])

# Tab 1: Single Student Prediction
with tab1:
    st.header("Student Trajectory Assessment")
    st.markdown("*Enter marks as they become available to predict final outcome*")
    
    # Select checkpoint
    st.subheader("🎯 Select Assessment Stage")
    checkpoint_options = list(CHECKPOINTS.keys())
    selected_checkpoint = st.selectbox(
        "When do you want to make the prediction?",
        checkpoint_options,
        index=3  # Default to "After T4 (June Exam)"
    )
    
    required_tasks = CHECKPOINTS[selected_checkpoint]
    
    # Show what this checkpoint means
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(f"📋 **Required marks:** {', '.join(required_tasks)}")
    with col2:
        checkpoint_perf = CHECKPOINT_RESULTS[selected_checkpoint]
        st.metric("Detection Rate", f"{checkpoint_perf['fail_recall']:.1%}")
    
    # Input fields
    st.subheader("Enter Student Marks")
    
    col1, col2 = st.columns(2)
    
    task_values = {}
    
    with col1:
        if 'T1 %' in required_tasks:
            task_values['T1 %'] = st.number_input("T1 % (Weight: 10%)", min_value=0.0, max_value=100.0, value=None, step=0.1, key='t1')
        if 'T2 %' in required_tasks:
            task_values['T2 %'] = st.number_input("T2 % (Weight: 10%)", min_value=0.0, max_value=100.0, value=None, step=0.1, key='t2')
        if 'T3 %' in required_tasks:
            task_values['T3 %'] = st.number_input("T3 % (Weight: 10%)", min_value=0.0, max_value=100.0, value=None, step=0.1, key='t3')
        if 'June Exam %' in required_tasks:
            task_values['June Exam %'] = st.number_input("June Exam % / T4 (Weight: 20%)", min_value=0.0, max_value=100.0, value=None, step=0.1, key='t4')
    
    with col2:
        if 'T5 %' in required_tasks:
            task_values['T5 %'] = st.number_input("T5 % (Weight: 10%)", min_value=0.0, max_value=100.0, value=None, step=0.1, key='t5')
        if 'T6 %' in required_tasks:
            task_values['T6 %'] = st.number_input("T6 % (Weight: 10%)", min_value=0.0, max_value=100.0, value=None, step=0.1, key='t6')
        if 'T7 %' in required_tasks:
            task_values['T7 %'] = st.number_input("T7 % (Weight: 30%)", min_value=0.0, max_value=100.0, value=None, step=0.1, key='t7')
    
    if st.button("🔮 Predict Trajectory", type="primary", use_container_width=True):
        # Check if all required fields are filled
        missing_fields = [task for task in required_tasks if task_values.get(task) is None]
        
        if missing_fields:
            st.error(f"❌ Please fill in all required fields: {', '.join(missing_fields)}")
        else:
            # Prepare input data
            input_data = pd.DataFrame({
                task: [task_values.get(task, 0)]
                for task in TASK_COLUMNS
            })
            
            # Make prediction
            result = predict_at_checkpoint(input_data, selected_checkpoint)
            
            # Display result
            st.markdown("---")
            st.subheader(f"📊 Trajectory Prediction: {selected_checkpoint}")
            
            # Main prediction
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if result['prediction'] == "Pass":
                    st.success(f"### ✅ Predicted Trajectory: ON TRACK TO PASS")
                    st.markdown(f"**Pass Likelihood:** {result['pass_probability']:.1%}")
                else:
                    st.error(f"### ❌ Predicted Trajectory: AT RISK OF FAILING")
                    st.markdown(f"**Fail Likelihood:** {result['fail_probability']:.1%}")
                    st.warning("⚠️ **Intervention Recommended:** This student's trajectory suggests they may fail without support")
            
            with col2:
                st.metric("Model Used", result['model_name'])
                st.metric("Checkpoint Accuracy", f"{checkpoint_perf['accuracy']:.1%}")
            
            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['pass_probability'] * 100,
                title={'text': "Pass Likelihood", 'font': {'size': 24}},
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen" if result['prediction'] == "Pass" else "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "#ffcccc"},
                        {'range': [30, 70], 'color': "#ffffcc"},
                        {'range': [70, 100], 'color': "#ccffcc"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Show key features
            st.markdown("---")
            st.subheader("📈 Performance Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            features = result['features']
            col1.metric("Current Average", f"{features['Average']:.1f}%")
            col2.metric("Cumulative Year Mark", f"{features['Cumulative_YearMark']:.1f}%")
            col3.metric("Performance Range", f"{features['Range']:.1f}%")
            
            if 'Trend' in features:
                trend_val = features['Trend']
                col4.metric("Trend", f"{trend_val:+.1f}%", 
                           delta="Improving" if trend_val > 0 else "Declining",
                           delta_color="normal" if trend_val > 0 else "inverse")
            
            # On track indicator
            if features.get('On_Track', 0) == 1:
                st.success("✅ **On Track:** Student's cumulative performance is proportional to expectations at this stage")
            else:
                st.warning("⚠️ **Below Track:** Student's cumulative performance is below expectations for this stage")
            
            # Feature importance
            if result['feature_importance']:
                st.markdown("---")
                st.subheader("🔍 What Matters Most for This Prediction?")
                
                # Sort by importance
                importance_sorted = sorted(result['feature_importance'].items(), key=lambda x: x[1], reverse=True)
                top_features = importance_sorted[:5]
                
                feature_names = [f[0] for f in top_features]
                feature_values = [f[1] * 100 for f in top_features]
                
                fig_importance = go.Figure(go.Bar(
                    x=feature_values,
                    y=feature_names,
                    orientation='h',
                    marker=dict(color=feature_values, colorscale='RdYlGn', showscale=False)
                ))
                fig_importance.update_layout(
                    title='Top 5 Most Important Factors',
                    xaxis_title='Importance (%)',
                    yaxis_title='',
                    height=300
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Model reliability
            st.markdown("---")
            st.subheader("📊 Prediction Reliability")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Model Accuracy", f"{checkpoint_perf['accuracy']:.1%}")
            col2.metric("At-Risk Detection", f"{checkpoint_perf['fail_recall']:.1%}")
            col3.metric("Historical Success", f"{checkpoint_perf['caught_fails']}/{checkpoint_perf['total_fails']}")
            col4.metric("AUC-ROC", f"{checkpoint_perf['auc']:.3f}")
            
            if checkpoint_perf['fail_recall'] < 0.3:
                st.info("ℹ️ **Note:** This is an early checkpoint. Predictions become more reliable as more tasks are completed.")
            elif checkpoint_perf['fail_recall'] >= 0.5:
                st.success("✅ **High Confidence:** This checkpoint has strong predictive power based on historical trajectories.")

# Tab 2: Bulk Upload
with tab2:
    st.header("Bulk Student Trajectory Analysis")
    st.markdown("Upload an Excel file with student data for batch predictions")
    
    # Select checkpoint for bulk analysis
    bulk_checkpoint = st.selectbox(
        "Select assessment stage for bulk analysis:",
        list(CHECKPOINTS.keys()),
        index=3,
        key='bulk_checkpoint'
    )
    
    required_tasks_bulk = CHECKPOINTS[bulk_checkpoint]
    st.info(f"📋 **Required columns:** {', '.join(required_tasks_bulk)}")
    
    # File upload
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ File uploaded successfully! Found {len(df)} students.")
        
        # Validate required columns
        missing_columns = [col for col in required_tasks_bulk if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
        else:
            # Fill missing values with 0
            df[required_tasks_bulk] = df[required_tasks_bulk].fillna(0)
            
            # Make predictions for all students
            predictions = []
            for idx, row in df.iterrows():
                # Create full input with zeros for unavailable tasks
                input_data = pd.DataFrame({
                    task: [row[task] if task in required_tasks_bulk else 0]
                    for task in TASK_COLUMNS
                })
                result = predict_at_checkpoint(input_data, bulk_checkpoint)
                predictions.append(result)
            
            # Add predictions to dataframe
            df['Prediction'] = [p['prediction'] for p in predictions]
            df['Pass Likelihood'] = [p['pass_probability'] for p in predictions]
            df['Risk Level'] = df['Pass Likelihood'].apply(
                lambda x: 'High Risk' if x < 0.3 else ('Medium Risk' if x < 0.7 else 'Low Risk')
            )
            
            # Statistics Dashboard
            st.markdown("---")
            st.subheader(f"📈 Analysis Dashboard - {bulk_checkpoint}")
            
            total_students = len(df)
            at_risk_count = (df['Prediction'] == 'Fail').sum()
            safe_count = (df['Prediction'] == 'Pass').sum()
            high_risk_count = (df['Risk Level'] == 'High Risk').sum()
            medium_risk_count = (df['Risk Level'] == 'Medium Risk').sum()
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Students", total_students)
            col2.metric("At Risk", at_risk_count, f"{at_risk_count/total_students*100:.1f}%", delta_color="inverse")
            col3.metric("On Track", safe_count, f"{safe_count/total_students*100:.1f}%")
            col4.metric("High Risk", high_risk_count, delta_color="inverse")
            col5.metric("Medium Risk", medium_risk_count, delta_color="inverse")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk distribution pie chart
                fig_pie = px.pie(
                    values=[safe_count, at_risk_count],
                    names=['On Track', 'At Risk'],
                    title='Student Risk Distribution',
                    color_discrete_sequence=['#00CC96', '#EF553B']
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Risk level breakdown
                risk_counts = df['Risk Level'].value_counts()
                fig_risk = px.bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    title='Risk Level Breakdown',
                    labels={'x': 'Risk Level', 'y': 'Number of Students'},
                    color=risk_counts.index,
                    color_discrete_map={'Low Risk': '#00CC96', 'Medium Risk': '#FFA500', 'High Risk': '#EF553B'}
                )
                st.plotly_chart(fig_risk, use_container_width=True)
            
            # Pass likelihood distribution
            fig_hist = px.histogram(
                df,
                x='Pass Likelihood',
                nbins=20,
                title='Distribution of Pass Likelihood',
                labels={'Pass Likelihood': 'Pass Likelihood'},
                color='Prediction',
                color_discrete_map={'Pass': '#00CC96', 'Fail': '#EF553B'}
            )
            fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red", 
                               annotation_text="50% Threshold")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Display results
            st.markdown("---")
            st.subheader("📋 Detailed Prediction Results")
            
            # Add styling
            def highlight_prediction(row):
                if row['Prediction'] == 'Fail':
                    return ['background-color: #ffcccc'] * len(row)
                else:
                    return ['background-color: #ccffcc'] * len(row)
            
            display_cols = [col for col in df.columns if col in required_tasks_bulk] + ['Prediction', 'Pass Likelihood', 'Risk Level']
            
            st.dataframe(
                df[display_cols].style.apply(highlight_prediction, axis=1).format({
                    'Pass Likelihood': '{:.2%}',
                    **{col: '{:.1f}' for col in required_tasks_bulk}
                }),
                use_container_width=True
            )
            
            # Download options
            st.markdown("---")
            st.subheader("💾 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_all = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download All Results (CSV)",
                    data=csv_all,
                    file_name=f"trajectory_predictions_{bulk_checkpoint.replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                at_risk = df[df['Prediction'] == 'Fail']
                if len(at_risk) > 0:
                    csv_at_risk = at_risk.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⚠️ Download At-Risk Students (CSV)",
                        data=csv_at_risk,
                        file_name=f"at_risk_students_{bulk_checkpoint.replace(' ', '_')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("🎉 No at-risk students!")
            
            with col3:
                high_risk = df[df['Risk Level'] == 'High Risk']
                if len(high_risk) > 0:
                    csv_high_risk = high_risk.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="🚨 Download High Risk Students (CSV)",
                        data=csv_high_risk,
                        file_name=f"high_risk_students_{bulk_checkpoint.replace(' ', '_')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("✅ No high-risk students!")

# Tab 3: System Performance
with tab3:
    st.header("📈 System Performance Analysis")
    st.markdown("*Compare trajectory prediction accuracy across different stages*")
    
    # Create comparison dataframe
    comparison_data = []
    for cp_name, results in CHECKPOINT_RESULTS.items():
        comparison_data.append({
            'Checkpoint': cp_name,
            'Accuracy': results['accuracy'] * 100,
            'At-Risk Detection': results['fail_recall'] * 100,
            'Pass Detection': results['pass_recall'] * 100,
            'AUC-ROC': results['auc'],
            'Caught At-Risk': f"{results['caught_fails']}/{results['total_fails']}",
            'False Alarms': f"{results['false_alarms']}/{results['total_passes']}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.dataframe(
        comparison_df.style.format({
            'Accuracy': '{:.2f}%',
            'At-Risk Detection': '{:.2f}%',
            'Pass Detection': '{:.2f}%',
            'AUC-ROC': '{:.4f}'
        }).background_gradient(subset=['At-Risk Detection'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    # Visualizations
    st.markdown("---")
    st.subheader("Performance Trends Across Checkpoints")
    
    # Multi-metric line chart
    fig_trends = go.Figure()
    
    fig_trends.add_trace(go.Scatter(
        x=comparison_df['Checkpoint'],
        y=comparison_df['Accuracy'],
        mode='lines+markers',
        name='Overall Accuracy',
        line=dict(color='blue', width=3),
        marker=dict(size=10)
    ))
    
    fig_trends.add_trace(go.Scatter(
        x=comparison_df['Checkpoint'],
        y=comparison_df['At-Risk Detection'],
        mode='lines+markers',
        name='At-Risk Detection',
        line=dict(color='red', width=3),
        marker=dict(size=10)
    ))
    
    fig_trends.add_trace(go.Scatter(
        x=comparison_df['Checkpoint'],
        y=comparison_df['Pass Detection'],
        mode='lines+markers',
        name='Pass Detection',
        line=dict(color='green', width=3),
        marker=dict(size=10)
    ))
    
    fig_trends.update_layout(
        xaxis_title='Checkpoint',
        yaxis_title='Percentage (%)',
        height=500,
        xaxis={'tickangle': -45},
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # AUC-ROC comparison
    fig_auc = px.bar(
        comparison_df,
        x='Checkpoint',
        y='AUC-ROC',
        title='Model Discrimination Power (AUC-ROC)',
        labels={'AUC-ROC': 'AUC-ROC Score'},
        color='AUC-ROC',
        color_continuous_scale='RdYlGn',
        range_color=[0.5, 1.0]
    )
    fig_auc.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Random Guess (0.5)")
    fig_auc.add_hline(y=0.7, line_dash="dash", line_color="orange", annotation_text="Good (0.7)")
    fig_auc.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="Excellent (0.8)")
    fig_auc.update_layout(xaxis={'tickangle': -45}, height=400)
    st.plotly_chart(fig_auc, use_container_width=True)
    
    # Key insights
    st.markdown("---")
    st.subheader("🔍 Key Insights")
    
    # Find best early checkpoint
    best_early_checkpoint = None
    for cp_name, results in CHECKPOINT_RESULTS.items():
        if 'T7' not in cp_name and results['fail_recall'] >= 0.30:
            best_early_checkpoint = cp_name
            break
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Recommended Checkpoint")
        if best_early_checkpoint:
            st.success(f"**{best_early_checkpoint}**")
            rec_perf = CHECKPOINT_RESULTS[best_early_checkpoint]
            st.markdown(f"""
            - **At-Risk Detection:** {rec_perf['fail_recall']:.1%}
            - **Accuracy:** {rec_perf['accuracy']:.1%}
            - **AUC-ROC:** {rec_perf['auc']:.3f}
            
            This is the earliest checkpoint with meaningful predictive power.
            """)
        else:
            st.info("Early checkpoints have limited predictive power. Consider using 'After T4 (June Exam)' as the primary checkpoint.")
    
    with col2:
        st.markdown("### 📊 Performance Summary")
        first_checkpoint_recall = list(CHECKPOINT_RESULTS.values())[0]['fail_recall']
        last_checkpoint_recall = list(CHECKPOINT_RESULTS.values())[-1]['fail_recall']
        improvement = (last_checkpoint_recall - first_checkpoint_recall) * 100
        
        st.metric(
            "Detection Improvement",
            f"{improvement:+.1f} pp",
            delta=f"From {first_checkpoint_recall:.1%} to {last_checkpoint_recall:.1%}"
        )
        
        avg_accuracy = np.mean([r['accuracy'] for r in CHECKPOINT_RESULTS.values()])
        st.metric("Average Accuracy", f"{avg_accuracy:.1%}")
        
        avg_auc = np.mean([r['auc'] for r in CHECKPOINT_RESULTS.values()])
        st.metric("Average AUC-ROC", f"{avg_auc:.3f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Trajectory-Based Early Warning System</strong></p>
    <p><em>Predict student outcomes based on historical performance patterns</em></p>
    <p>✅ Realistic predictions | 🎯 Early intervention | 📊 Data-driven decisions</p>
</div>
""", unsafe_allow_html=True)