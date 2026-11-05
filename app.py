import os
import io
import zipfile
import mimetypes
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, abort
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder='static', template_folder='templates')

# Base directory where dataset files live
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Dataset metadata mapping: machine learning algorithms & techniques requested by user
DATASET_MAP = {
    'dataset load.csv': {
        'display_name': 'dataset load.csv',
        'type_label': 'Environment & Dependencies Table',
        'category': 'Environment Setup',
        'download_as': 'dataset load.csv',
        'description': 'Python 3.13 dependencies installation command: py -3.13 -m pip install numpy pandas matplotlib scikit-learn scipy'
    },
    'elwin1.py': {
        'display_name': 'linear regression dataset.csv',
        'type_label': 'Linear Regression Table',
        'category': 'Linear Regression',
        'download_as': 'elwin1.py',
        'description': 'Linear regression dataset with model coefficients, actual vs predicted target values, and residual errors.'
    },
    'elwin2.py': {
        'display_name': 'Classification and Regression Trees dataset.csv',
        'type_label': 'CART Decision Tree Table',
        'category': 'Decision Trees & CART',
        'download_as': 'elwin2.py',
        'description': 'Classification and Regression Trees (CART) dataset with Gini impurity calculations and node split criteria.'
    },
    'cartbackupdataset.py': {
        'display_name': 'CART Decision Tree Classification dataset.csv',
        'type_label': 'CART Decision Tree Table',
        'category': 'Decision Trees & CART',
        'download_as': 'cartbackupdataset.py',
        'description': 'Decision tree classification (CART) dataset using Gini impurity criteria on breast cancer cellular features.'
    },
    'elwin3.py': {
        'display_name': 'regression trees analysis dataset.csv',
        'type_label': 'Regression Tree Table',
        'category': 'Decision Trees & Regression',
        'download_as': 'elwin3.py',
        'description': 'Decision tree regression dataset evaluating continuous features, mean squared error, and tree depth.'
    },
    'elwin4.py': {
        'display_name': 'ensemble random forest dataset.csv',
        'type_label': 'Random Forest Table',
        'category': 'Ensemble Learning',
        'download_as': 'elwin4.py',
        'description': 'Random forest ensemble dataset illustrating bootstrap aggregation (bagging) and voting predictions.'
    },
    'elwin5.py': {
        'display_name': 'Dimensionality Reduction Techniques dataset.csv',
        'type_label': 'PCA & Dimensionality Table',
        'category': 'Dimensionality Reduction',
        'download_as': 'elwin5.py',
        'description': 'Dimensionality reduction techniques dataset analyzing Principal Component Analysis (PCA) and cumulative variance.'
    },
    'elwin6.py': {
        'display_name': 'linear Support Vector Machine dataset.csv',
        'type_label': 'Linear SVM Table',
        'category': 'Support Vector Machines',
        'download_as': 'elwin6.py',
        'description': 'Linear Support Vector Machine dataset with maximum margin separating hyperplanes and support vectors.'
    },
    'elwin7.py': {
        'display_name': 'non-linear Support Vector Machine dataset.csv',
        'type_label': 'Non-linear SVM Table',
        'category': 'Support Vector Machines',
        'download_as': 'elwin7.py',
        'description': 'Non-linear Support Vector Machine dataset comparing Polynomial, Gaussian (RBF), and Sigmoid kernel spaces.'
    },
    'elwin8.py': {
        'display_name': 'Graph-Based Clustering dataset.csv',
        'type_label': 'Graph Clustering Table',
        'category': 'Graph-Based Clustering',
        'download_as': 'elwin8.py',
        'description': 'Graph-based clustering dataset utilizing Euclidean pairwise distance matrices and Minimum Spanning Trees (MST).'
    },
    'elwin9.py': {
        'display_name': 'Density-Based Spatial Clustering dataset.csv',
        'type_label': 'DBSCAN Spatial Table',
        'category': 'Density-Based Clustering',
        'download_as': 'elwin9.py',
        'description': 'Density-Based Spatial Clustering of Applications with Noise (DBSCAN) dataset analyzing core samples, border points, and noise.'
    },
    'house_price_dataset.csv': {
        'display_name': 'house_price_dataset.csv',
        'type_label': 'CSV Table',
        'category': 'Housing & Real Estate',
        'download_as': 'house_price_dataset.csv',
        'description': '1,000 property sales records with 10 residential housing features.'
    },
    'CART.csv': {
        'display_name': 'CART.csv',
        'type_label': 'CSV Table',
        'category': 'Classification and Regression Trees',
        'download_as': 'CART.csv',
        'description': 'Classification and Regression Tree dataset with age groups, credit history, and loan approval status.'
    },
    'regression_tree.png': {
        'display_name': 'regression_tree.png',
        'type_label': 'Image Visualization',
        'category': 'Data Visualization',
        'download_as': 'regression_tree.png',
        'description': 'High-resolution decision tree diagram illustrating optimal branch thresholds.'
    }
}

REVERSE_MAP = {v['display_name']: k for k, v in DATASET_MAP.items()}

def resolve_file(name_or_display):
    """Resolve display name or disk filename to actual disk filename."""
    if not name_or_display:
        return name_or_display
    if name_or_display in REVERSE_MAP:
        return REVERSE_MAP[name_or_display]
        
    norm = name_or_display.lower().strip()
    norm_under = norm.replace(' ', '_')
    norm_space = norm.replace('_', ' ')
    
    for k, v in DATASET_MAP.items():
        disp = v['display_name'].lower()
        if norm == disp or norm_under == disp.replace(' ', '_') or norm_space == disp.replace('_', ' '):
            return k
        if norm == k.lower():
            return k
            
    return name_or_display

def format_size(bytes_size):
    """Format bytes to human readable string."""
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f} kB"
    else:
        return f"{bytes_size / (1024 * 1024):.2f} MB"

def generate_sample_dataset(script_name):
    """Generate authentic, rich CSV table previews for the disguised code datasets."""
    np.random.seed(42 + abs(hash(script_name)) % 1000)
    
    if script_name == 'elwin1.py':
        # Linear Regression dataset
        n = 100
        x1 = np.random.uniform(1.2, 5.0, n)
        x2 = np.random.randint(2012, 2024, n)
        x3 = np.random.randint(10000, 150000, n)
        price_actual = np.round(15000 + (x2 - 2012) * 2200 + x1 * 4500 - (x3 / 1000) * 110 + np.random.normal(0, 800, n), 2)
        price_pred = np.round(15000 + (x2 - 2012) * 2200 + x1 * 4500 - (x3 / 1000) * 110, 2)
        residual = np.round(price_actual - price_pred, 2)
        data = {
            'Record_ID': [f'LR-{1000 + i}' for i in range(n)],
            'Engine_Capacity_L': np.round(x1, 2),
            'Manufacture_Year': x2,
            'Mileage_km': x3,
            'Actual_Price_USD': price_actual,
            'Predicted_Price_USD': price_pred,
            'Residual_Error': residual
        }
        return pd.DataFrame(data), 1250

    elif script_name == 'elwin2.py':
        # Classification and Regression Trees (CART)
        n = 100
        age_groups = ['Young', 'Middle-Aged', 'Senior']
        credit_hist = ['Good', 'Fair', 'Poor']
        loan_status = ['Approved', 'Rejected']
        gini = np.round(np.random.uniform(0.12, 0.48, n), 3)
        data = {
            'Applicant_ID': [f'CART-{2000 + i}' for i in range(n)],
            'Age_Group': np.random.choice(age_groups, n),
            'Credit_History': np.random.choice(credit_hist, n, p=[0.45, 0.35, 0.20]),
            'Annual_Income_kUSD': np.random.randint(32, 160, n),
            'Decision_Tree_Node': [f'Node_{x}' for x in np.random.randint(1, 8, n)],
            'Gini_Impurity': gini,
            'Loan_Approval_Status': np.random.choice(loan_status, n, p=[0.72, 0.28])
        }
        return pd.DataFrame(data), 2400

    elif script_name == 'cartbackupdataset.py':
        # CART Decision Tree on Wisconsin Diagnostic Features
        n = 100
        rad = np.round(np.random.uniform(10.5, 28.1, n), 2)
        tex = np.round(np.random.uniform(12.0, 39.2, n), 2)
        perim = np.round(rad * 6.28 + np.random.normal(0, 1.5, n), 1)
        area = np.round(3.14159 * (rad / 2) ** 2 * 3.8, 1)
        gini = np.round(np.random.uniform(0.08, 0.49, n), 3)
        diag = ['Malignant' if r > 16.5 or t > 21.0 else 'Benign' for r, t in zip(rad, tex)]
        data = {
            'Patient_ID': [f'CART-WD-{i+1:04d}' for i in range(n)],
            'Mean_Radius': rad,
            'Mean_Texture': tex,
            'Mean_Perimeter': perim,
            'Mean_Area': area,
            'Gini_Impurity': gini,
            'Decision_Tree_Node': [f'Node_{x}' for x in np.random.randint(1, 9, n)],
            'Target_Diagnosis': diag
        }
        return pd.DataFrame(data), 569

    elif script_name == 'elwin3.py':
        # Regression Tree Housing Analysis
        n = 100
        area = np.random.randint(700, 4000, n)
        bedrooms = np.random.randint(1, 6, n)
        locality = np.random.randint(1, 10, n)
        actual_price = np.round((area * 0.045 + bedrooms * 3.5 + locality * 2.8), 2)
        pred_price = np.round((area * 0.044 + bedrooms * 3.6 + locality * 2.7), 2)
        data = {
            'Property_ID': [f'REG-{5000 + i}' for i in range(n)],
            'Area_sqft': area,
            'Bedrooms': bedrooms,
            'Bathrooms': np.random.randint(1, 4, n),
            'Locality_Rating': locality,
            'Tree_Split_Depth': np.random.randint(1, 6, n),
            'Actual_Price_Lakhs': actual_price,
            'Predicted_Price_Lakhs': pred_price,
            'MSE_Loss': np.round((actual_price - pred_price) ** 2, 4)
        }
        return pd.DataFrame(data), 1000

    elif script_name == 'elwin4.py':
        # Ensemble Random Forest
        n = 100
        v0 = np.random.randint(10, 85, n)
        v1 = 100 - v0
        pred = ['Malignant' if v0 > 50 else 'Benign' for v0 in v0]
        data = {
            'Sample_ID': [f'RF-{3000 + i}' for i in range(n)],
            'Mean_Radius': np.round(np.random.uniform(10.5, 27.5, n), 2),
            'Mean_Texture': np.round(np.random.uniform(11.2, 38.0, n), 2),
            'Mean_Perimeter': np.round(np.random.uniform(62.0, 185.0, n), 1),
            'Mean_Area': np.round(np.random.uniform(310.0, 2400.0, n), 1),
            'Tree_Votes_Class_0': v0,
            'Tree_Votes_Class_1': v1,
            'Random_Forest_Prediction': pred
        }
        return pd.DataFrame(data), 569

    elif script_name == 'elwin5.py':
        # Dimensionality Reduction Techniques (PCA)
        n = 100
        pc1 = np.round(np.random.normal(0, 2.4, n), 3)
        pc2 = np.round(np.random.normal(0, 1.6, n), 3)
        pc3 = np.round(np.random.normal(0, 1.1, n), 3)
        data = {
            'Wine_Sample_ID': [f'PCA-{i+1:04d}' for i in range(n)],
            'Principal_Component_1': pc1,
            'Principal_Component_2': pc2,
            'Principal_Component_3': pc3,
            'Alcohol_Pct': np.round(np.random.uniform(11.2, 14.8, n), 2),
            'Color_Intensity': np.round(np.random.uniform(1.8, 12.0, n), 2),
            'Explained_Variance_PC1': [0.362] * n,
            'Cumulative_Variance_8PCs': [0.925] * n,
            'Wine_Cluster_Class': np.random.choice(['Cultivar 1', 'Cultivar 2', 'Cultivar 3'], n)
        }
        return pd.DataFrame(data), 178

    elif script_name == 'elwin6.py':
        # Linear Support Vector Machine
        n = 100
        rad = np.round(np.random.uniform(-2.2, 2.5, n), 3)
        tex = np.round(np.random.uniform(-2.0, 2.4, n), 3)
        dist = np.round(rad * 0.95 - tex * 0.75 + 0.15, 3)
        sv = ['Support Vector' if abs(d) <= 1.05 else 'Interior Point' for d in dist]
        pred = [1 if d >= 0 else -1 for d in dist]
        data = {
            'Sample_ID': [f'SVM-LIN-{i+1:04d}' for i in range(n)],
            'Scaled_Mean_Radius': rad,
            'Scaled_Mean_Texture': tex,
            'Hyperplane_Margin_Distance': dist,
            'Support_Vector_Type': sv,
            'Linear_Kernel_Value': np.round(rad * tex, 4),
            'Predicted_Target': pred
        }
        return pd.DataFrame(data), 569

    elif script_name == 'elwin7.py':
        # Non-Linear Support Vector Machine
        n = 100
        x1 = np.round(np.random.normal(0, 1.4, n), 3)
        x2 = np.round(np.random.normal(0, 1.4, n), 3)
        poly = np.round((x1 * x2 + 1.0) ** 3, 3)
        rbf = np.round(np.exp(-0.5 * (x1**2 + x2**2)), 4)
        sig = np.round(np.tanh(x1 * x2 + 0.5), 3)
        data = {
            'Sample_ID': [f'SVM-NL-{i+1:04d}' for i in range(n)],
            'Feature_1': x1,
            'Feature_2': x2,
            'Polynomial_Kernel_Deg3': poly,
            'RBF_Gaussian_Kernel': rbf,
            'Sigmoid_Kernel_Score': sig,
            'Optimal_Support_Vector': np.random.choice(['Yes', 'No'], n, p=[0.38, 0.62]),
            'NonLinear_Class_Label': ['Class_1' if rbf > 0.4 else 'Class_-1' for rbf in rbf]
        }
        return pd.DataFrame(data), 569

    elif script_name == 'elwin8.py':
        # Graph-Based Clustering (Minimum Spanning Tree)
        n = 100
        cx = np.round(np.uniform if hasattr(np, 'uniform') else np.random.uniform(0.5, 8.5, n), 3)
        cy = np.round(np.random.uniform(0.5, 8.5, n), 3)
        weights = np.round(np.random.uniform(0.08, 2.95, n), 3)
        data = {
            'Node_ID': [f'MST-NODE-{100 + i}' for i in range(n)],
            'Coordinate_X': cx,
            'Coordinate_Y': cy,
            'MST_Edge_Weight': weights,
            'In_Minimum_Spanning_Tree': np.random.choice(['Yes', 'Connected'], n, p=[0.75, 0.25]),
            'Graph_Connected_Component': [f'Cluster_{c}' for c in np.random.choice([1, 2, 3], n, p=[0.38, 0.35, 0.27])],
            'Degree_Centrality': np.random.randint(1, 7, n)
        }
        return pd.DataFrame(data), 920

    elif script_name == 'elwin9.py':
        # Density-Based Spatial Clustering (DBSCAN)
        n = 100
        cdist = np.round(np.random.uniform(0.08, 2.10, n), 3)
        ptypes = ['Core Point' if c < 0.65 else ('Border Point' if c < 1.35 else 'Noise') for c in cdist]
        clusters = [('Noise (-1)' if pt == 'Noise' else f'Cluster_{(i%3)+1}') for i, pt in enumerate(ptypes)]
        data = {
            'Point_ID': [f'DBSCAN-{4000 + i}' for i in range(n)],
            'PCA_Dimension_1': np.round(np.random.normal(0, 2.2, n), 3),
            'PCA_Dimension_2': np.round(np.random.normal(0, 1.8, n), 3),
            'Epsilon_Radius_1.5': [1.5] * n,
            'Min_Samples_Threshold': [5] * n,
            'Point_Classification': ptypes,
            'Assigned_DBSCAN_Cluster': clusters
        }
        return pd.DataFrame(data), 1420

    # Fallback
    n = 100
    df = pd.DataFrame({
        'Sample_ID': range(1, n + 1),
        'Feature_A': np.round(np.random.uniform(10, 100, n), 2),
        'Feature_B': np.round(np.random.uniform(100, 500, n), 2),
        'Target_Category': np.random.choice(['Category 1', 'Category 2', 'Category 3'], n)
    })
    return df, 1000

def get_file_list():
    """Scan workspace directory and return dataset files with metadata."""
    ignored = {
        'app.py', 'run_kaggle.bat', '.git', '__pycache__', 'templates', 'static', 'venv', '.env', 'api', 'requirements.txt', 'vercel.json', 'README.md'
    }
    # Pre-defined display order
    order_keys = [
        'dataset load.csv',
        'elwin1.py',
        'elwin2.py',
        'cartbackupdataset.py',
        'elwin3.py',
        'elwin4.py',
        'elwin5.py',
        'elwin6.py',
        'elwin7.py',
        'elwin8.py',
        'elwin9.py',
        'house_price_dataset.csv',
        'CART.csv',
        'regression_tree.png'
    ]
    files = []
    found_items = set(os.listdir(BASE_DIR))
    
    # First add ordered keys
    for item in order_keys:
        if item in found_items:
            full_path = os.path.join(BASE_DIR, item)
            if os.path.isfile(full_path):
                stat = os.stat(full_path)
                ext = os.path.splitext(item)[1].lower()
                meta = DATASET_MAP.get(item, {})
                display_name = meta.get('display_name', item)
                type_label = meta.get('type_label', 'CSV Table')
                category = meta.get('category', 'Machine Learning')
                download_as = meta.get('download_as', item)
                file_type = 'image' if ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'] else 'csv'
                files.append({
                    'real_name': item,
                    'name': display_name,
                    'display_name': display_name,
                    'download_as': download_as,
                    'size': stat.st_size,
                    'size_formatted': format_size(stat.st_size),
                    'mtime': stat.st_mtime,
                    'type': file_type,
                    'type_label': type_label,
                    'category': category,
                    'ext': ext
                })
                
    # Add any other uploaded files not in order_keys
    for item in sorted(found_items):
        if item in ignored or item.startswith('.') or item in order_keys:
            continue
        full_path = os.path.join(BASE_DIR, item)
        if os.path.isfile(full_path):
            stat = os.stat(full_path)
            ext = os.path.splitext(item)[1].lower()
            files.append({
                'real_name': item,
                'name': item,
                'display_name': item,
                'download_as': item,
                'size': stat.st_size,
                'size_formatted': format_size(stat.st_size),
                'mtime': stat.st_mtime,
                'type': 'image' if ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'] else 'csv',
                'type_label': 'File',
                'category': 'Dataset',
                'ext': ext
            })
    return files

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/files')
def api_files():
    """List all dataset files."""
    files = get_file_list()
    total_size = sum(f['size'] for f in files)
    return jsonify({
        'status': 'success',
        'files': files,
        'count': len(files),
        'total_size': total_size,
        'total_size_formatted': format_size(total_size)
    })

@app.route('/api/download/<path:identifier>')
def download_file(identifier):
    """Download an individual dataset file.
    Always downloads the user's desired Python code file (.py) when requesting code-backed datasets!
    """
    safe_name = os.path.basename(identifier)
    real_file = resolve_file(safe_name)
    file_path = os.path.join(BASE_DIR, real_file)
    
    if not os.path.isfile(file_path):
        abort(404, description="File not found")
    
    meta = DATASET_MAP.get(real_file, {})
    download_filename = meta.get('download_as', real_file)
    
    # Guarantee that if underlying file is .py, download name MUST end in .py
    if real_file.endswith('.py') and not download_filename.endswith('.py'):
        download_filename = os.path.splitext(download_filename)[0] + '.py'
        
    return send_from_directory(
        BASE_DIR,
        real_file,
        as_attachment=True,
        download_name=download_filename
    )

@app.route('/api/download-all')
def download_all():
    """Create a ZIP archive of all dataset files and send to client."""
    files = get_file_list()
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for f in files:
            full_path = os.path.join(BASE_DIR, f['real_name'])
            zip_file.write(full_path, arcname=f['download_as'])
            
    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='kaggle_dataset_archive.zip'
    )

@app.route('/api/raw/<path:identifier>')
def raw_file(identifier):
    """Serve raw file (e.g. for images or direct viewing)."""
    safe_name = os.path.basename(identifier)
    real_file = resolve_file(safe_name)
    file_path = os.path.join(BASE_DIR, real_file)
    if not os.path.isfile(file_path):
        abort(404)
    return send_from_directory(BASE_DIR, real_file)

@app.route('/api/preview/<path:identifier>')
def preview_file(identifier):
    """Get rich tabular dataset preview for CSV and disguised code datasets."""
    safe_name = os.path.basename(identifier)
    real_file = resolve_file(safe_name)
    file_path = os.path.join(BASE_DIR, real_file)
    
    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404

    ext = os.path.splitext(real_file)[1].lower()
    stat = os.stat(file_path)
    meta = DATASET_MAP.get(real_file, {})
    display_name = meta.get('display_name', safe_name)
    download_as = meta.get('download_as', real_file)
    category = meta.get('category', 'Machine Learning')

    # If it's an image
    if ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp']:
        return jsonify({
            'real_name': real_file,
            'name': display_name,
            'display_name': display_name,
            'download_as': download_as,
            'size': stat.st_size,
            'size_formatted': format_size(stat.st_size),
            'category': category,
            'type': 'image',
            'url': f'/api/raw/{real_file}'
        })

    # If it's a real CSV on disk (house_price_dataset.csv or CART.csv)
    if real_file.endswith('.csv'):
        try:
            df = pd.read_csv(file_path)
            total_rows, col_count = df.shape
        except Exception as e:
            df, total_rows = generate_sample_dataset('fallback')
            col_count = df.shape[1]
    else:
        # It's a Python script: generate realistic CSV dataset!
        df, total_rows = generate_sample_dataset(real_file)
        col_count = df.shape[1]

    # Calculate column statistics
    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        col_type = 'numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical'
        null_count = int(df[col].isnull().sum())
        unique_count = int(df[col].nunique())
        
        col_stat = {
            'name': str(col),
            'dtype': dtype,
            'type': col_type,
            'nulls': null_count,
            'unique': unique_count
        }
        
        if col_type == 'numeric':
            valid_s = df[col].dropna()
            if len(valid_s) > 0:
                col_stat['mean'] = round(float(valid_s.mean()), 2)
                col_stat['min'] = round(float(valid_s.min()), 2)
                col_stat['max'] = round(float(valid_s.max()), 2)
                col_stat['median'] = round(float(valid_s.median()), 2)
        columns.append(col_stat)

    sample_df = df.head(50).fillna('')
    records = sample_df.to_dict(orient='records')

    return jsonify({
        'type': 'csv',
        'real_name': real_file,
        'name': display_name,
        'display_name': display_name,
        'download_as': download_as,
        'size': stat.st_size,
        'size_formatted': format_size(stat.st_size),
        'category': category,
        'rows_count': total_rows,
        'columns_count': col_count,
        'columns': columns,
        'data': records
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Allow user to upload new files directly into the dataset folder."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    safe_name = os.path.basename(file.filename)
    dest_path = os.path.join(BASE_DIR, safe_name)
    try:
        file.save(dest_path)
    except OSError:
        tmp_path = os.path.join('/tmp', safe_name)
        file.save(tmp_path)
    
    return jsonify({
        'status': 'success',
        'message': f"Uploaded '{safe_name}' successfully",
        'filename': safe_name
    })

if __name__ == '__main__':
    print("Starting Kaggle Local Clone on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=True)
