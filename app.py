import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import datetime
from src.preprocessing.data_processor import DataProcessor
from src.models.model_trainer import ModelTrainer
from src.visualization.data_viz import DataVisualizer
from src.utils.model_utils import ModelUtils

# Configure the page
st.set_page_config(
    page_title="Data Preprocessing & ML Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# Create dataset-specific directories
def create_dataset_directories(dataset_name):
    """Create directories for a specific dataset"""
    # Convert dataset name to a valid directory name
    dataset_dir_name = dataset_name.lower().replace(' ', '_')
    dataset_dir = f'data/datasets/{dataset_dir_name}'
    
    # Create dataset-specific directories
    directories = [
        dataset_dir,
        f"{dataset_dir}/raw",
        f"{dataset_dir}/processed",
        f"{dataset_dir}/models",
        f"{dataset_dir}/visualizations",
        f"{dataset_dir}/logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return dataset_dir

# Initialize session state
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'dataset_dir' not in st.session_state:
    st.session_state.dataset_dir = None

# Initialize components
data_processor = DataProcessor()
model_trainer = ModelTrainer()
data_visualizer = DataVisualizer()
model_utils = ModelUtils()

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    This application provides an end-to-end machine learning pipeline:
    
    1. **Data Preprocessing**
       - Upload and clean your data
       - Handle missing values
       - Encode categorical variables
       - Remove correlated features
       - Balance the dataset
    
    2. **Analysis & Visualization**
       - View dataset statistics
       - Explore feature distributions
       - Analyze correlations
    
    3. **Prediction**
       - Train multiple models
       - Make predictions
       - Ensemble voting
       - Download results
    """)
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Upload your CSV file
    2. Select target column
    3. Process the data
    4. View visualizations
    5. Make predictions
    """)

# Main content
st.title('🤖 Data Preprocessing & ML Pipeline')

tab1, tab2, tab3 = st.tabs(['⚙️ Processing', '📊 Analysis & Visualizations', '🤖 Prediction'])

with tab1:
    st.header("Data Processing")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Get dataset name from file
            dataset_name = os.path.splitext(uploaded_file.name)[0]
            
            # Create dataset-specific directories
            dataset_dir = create_dataset_directories(dataset_name)
            st.session_state.dataset_dir = dataset_dir
            
            # Save uploaded file
            raw_file_path = os.path.join(dataset_dir, "raw", uploaded_file.name)
            with open(raw_file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Read the file
            df = pd.read_csv(raw_file_path)
            st.session_state.df_original = df
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Show basic dataset info
            st.write("Dataset Information:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.stop()
    
    if "df_original" in st.session_state:
        df = st.session_state.df_original
        target_column = st.selectbox("Select the target column", df.columns)
        
        if st.button('Process File'):
            try:
                with st.spinner('Processing data...'):
                    # Data preprocessing
                    progress_bar = st.progress(0)
                    
                    # Clean data
                    progress_bar.progress(20)
                    df = data_processor.clean_data(df, target_column)
                    
                    # Encode categorical
                    progress_bar.progress(40)
                    df = data_processor.encode_categorical(df)
                    
                    # Remove correlated features
                    progress_bar.progress(60)
                    df = data_processor.remove_correlated_features(df, target_column)
                    
                    # Split and prepare data
                    progress_bar.progress(80)
                    x_train, x_test, y_train, y_test = data_processor.split_and_balance(df, target_column)
                    x_train_sc, x_test_sc = data_processor.scale_features(x_train, x_test)
                    
                    # Train models
                    progress_bar.progress(90)
                    model_trainer.train_models(x_train_sc, y_train)
                    
                    # Save state
                    st.session_state.df = df
                    st.session_state.df_copy = df.copy()
                    st.session_state.target_column = target_column
                    st.session_state.data_processor = data_processor
                    st.session_state.model_trainer = model_trainer
                    
                    # Save processed data
                    progress_bar.progress(95)
                    
                    # Save to dataset-specific directory
                    processed_file = os.path.join(st.session_state.dataset_dir, "processed", "processed_data.csv")
                    train_file = os.path.join(st.session_state.dataset_dir, "processed", "train_data.csv")
                    test_file = os.path.join(st.session_state.dataset_dir, "processed", "test_data.csv")
                    
                    # Save files
                    df.to_csv(processed_file, index=False)
                    pd.DataFrame(x_train_sc).to_csv(train_file, index=False)
                    pd.DataFrame(x_test_sc).to_csv(test_file, index=False)
                    
                    # Save for download
                    st.session_state.processed_csv = df.to_csv(index=False).encode('utf-8')
                    st.session_state.train_csv = pd.DataFrame(x_train_sc).to_csv(index=False).encode('utf-8')
                    st.session_state.test_csv = pd.DataFrame(x_test_sc).to_csv(index=False).encode('utf-8')
                    
                    # Save models
                    models_dir = os.path.join(st.session_state.dataset_dir, "models")
                    model_trainer.save_models(models_dir)
                    
                    progress_bar.progress(100)
                    st.session_state.processing_complete = True
                    
                    st.success('Processing and training completed successfully!')
                    
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                st.stop()
        
        if st.session_state.processing_complete:
            st.markdown("### Download Processed Data")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📥 Preprocessed Data",
                    data=st.session_state.processed_csv,
                    file_name='preprocessed_data.csv',
                    mime='text/csv'
                )

            with col2:
                st.download_button(
                    label="📥 Train Data",
                    data=st.session_state.train_csv,
                    file_name='train_data.csv',
                    mime='text/csv'
                )

            with col3:
                st.download_button(
                    label="📥 Test Data",
                    data=st.session_state.test_csv,
                    file_name='test_data.csv',
                    mime='text/csv'
                )

with tab2:
    if 'df' in st.session_state: 
        df = st.session_state.df

        st.header("Dataset Analysis")
        
        # Dataset Summary
        st.subheader("Dataset Summary")
        summary = data_visualizer.dataset_summary(df)
        st.json(summary)

        # Visualizations
        st.subheader("Feature Visualizations")
        visualizations = data_visualizer.create_visualizations(df, st.session_state.target_column)
        
        for viz_type, column, fig in visualizations:
            with st.expander(f"{viz_type} - {column}"):
                st.plotly_chart(fig, use_container_width=True)
                # Save visualization
                if st.session_state.dataset_dir:
                    viz_dir = os.path.join(st.session_state.dataset_dir, "visualizations")
                    fig.write_html(os.path.join(viz_dir, f"{viz_type}_{column}.html"))
    else:
        st.warning("⚠️ No data available. Please upload and process the data in the 'Processing' tab first.")

with tab3:
    if 'df_copy' in st.session_state:
        df = st.session_state.df_copy
        target_column = st.session_state.target_column
        
        st.header("Make Predictions")
        
        # Feature inputs
        st.subheader("Input Features")
        input_features = df.drop(target_column, axis=1).columns
        cat_features = df.drop(target_column, axis=1).select_dtypes(include='object').columns
        
        # Create two columns for feature inputs
        col1, col2 = st.columns(2)
        
        inputs = {}
        for i, feature in enumerate(input_features):
            with col1 if i % 2 == 0 else col2:
                if feature in cat_features:
                    unique_values = df[feature].unique().tolist()
                    inputs[feature] = st.selectbox(feature, options=unique_values)
                else:
                    if pd.api.types.is_float_dtype(df[feature]):
                        inputs[feature] = st.number_input(feature, step=0.1, format='%.2f')
                    else:
                        inputs[feature] = st.number_input(feature, step=1)
        
        # Prepare features for prediction
        features_list = []
        for col in input_features:
            value = inputs[col]
            if col in cat_features:
                transformed_value = st.session_state.data_processor.encoder[col].transform(np.array([[value]]))
                features_list.append(transformed_value.item())
            else:
                features_list.append(value)

        features_array = np.array(features_list).reshape(1, -1)
        features_scaled = st.session_state.data_processor.transform_new_data(features_array)

        if 'predictions' not in st.session_state:
            st.session_state.predictions = []

        # Prediction buttons
        st.subheader("Model Predictions")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button('🤖 Logistic Regression'):
                pred = st.session_state.model_trainer.predict_logistic(features_scaled)
                st.session_state.predictions.append(pred)
                if pred == 1:
                    st.success(f'✅ {target_column}')
                else:
                    st.error(f'❌ Not {target_column}')

        with col2:
            if st.button('🌲 Random Forest'):
                pred = st.session_state.model_trainer.predict_random_forest(features_scaled)
                st.session_state.predictions.append(pred)
                if pred == 1:
                    st.success(f'✅ {target_column}')
                else:
                    st.error(f'❌ Not {target_column}')

        with col3:
            if st.button('🚀 XGBoost'):
                pred = st.session_state.model_trainer.predict_xgboost(features_scaled)
                st.session_state.predictions.append(pred)
                if pred == 1:
                    st.success(f'✅ {target_column}')
                else:
                    st.error(f'❌ Not {target_column}')

        with col4:
            if st.button('🎯 Ensemble Vote'):
                if len(st.session_state.predictions) == 3:
                    final_prediction = model_trainer.ensemble_voting(st.session_state.predictions)
                    if final_prediction == 1:
                        st.success(f'✅ Final Vote: {target_column}')
                    else:
                        st.error(f'❌ Final Vote: Not {target_column}')
                    
                    # Save prediction
                    input_with_prediction = inputs.copy()
                    input_with_prediction['Prediction'] = final_prediction
                    st.session_state.user_inputs.append(input_with_prediction)
                    
                    # Show prediction history
                    if st.session_state.user_inputs:
                        st.subheader("Prediction History")
                        history_df = pd.DataFrame(st.session_state.user_inputs)
                        st.dataframe(history_df)
                        
                        # Save prediction history
                        if st.session_state.dataset_dir:
                            history_file = os.path.join(st.session_state.dataset_dir, "processed", "prediction_history.csv")
                            history_df.to_csv(history_file, index=False)
                        
                        # Offer download of predictions
                        csv_data = model_utils.save_user_inputs(st.session_state.user_inputs)
                        if csv_data:
                            st.download_button(
                                label="📥 Download Prediction History",
                                data=csv_data,
                                file_name='prediction_history.csv',
                                mime='text/csv'
                            )
                    
                    st.session_state.predictions = []
                else:
                    st.error('⚠️ Please run all individual model predictions first')
                    st.session_state.predictions = []
    else:
        st.warning("⚠️ No data available. Please upload and process the data in the 'Processing' tab first.")
