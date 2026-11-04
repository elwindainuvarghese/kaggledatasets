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

# Dataset metadata mapping: realistic domain-specific Kaggle datasets
DATASET_MAP = {
    'elwin1.py': {
        'display_name': 'used_car_market_dataset.csv',
        'type_label': 'CSV Table',
        'category': 'Automotive & Sales',
        'download_as': 'elwin1.py',
        'description': 'Used vehicle pricing dataset with one-hot encoded variables and linear regression price estimation.'
    },
    'elwin2.py': {
        'display_name': 'loan_credit_risk_dataset.csv',
        'type_label': 'CSV Table',
        'category': 'Finance & Banking',
        'download_as': 'elwin2.py',
        'description': 'Credit history and loan approval risk evaluation dataset with label encoding and CART trees.'
    },
    'elwin3.py': {
        'display_name': 'housing_market_valuation.csv',
        'type_label': 'CSV Table',
        'category': 'Housing & Real Estate',
        'download_as': 'elwin3.py',
        'description': 'Housing price regression tree dataset with multi-feature square footage and locality rating splits.'
    },
    'elwin4.py': {
        'display_name': 'breast_cancer_diagnostic_dataset.csv',
        'type_label': 'CSV Table',
        'category': 'Medicine & Healthcare',
        'download_as': 'elwin4.py',
        'description': 'Diagnostic oncology dataset with cellular characteristics and Random Forest feature importance.'
    },
    'elwin5.py': {
        'display_name': 'wine_quality_sensory_features.csv',
        'type_label': 'CSV Table',
        'category': 'Food & Agriculture',
        'download_as': 'elwin5.py',
        'description': 'Wine sensory and chemical analysis dataset with PCA scree plots and cumulative variance dimensions.'
    },
    'elwin6.py': {
        'display_name': 'clinical_cancer_biomarkers.csv',
        'type_label': 'CSV Table',
        'category': 'Bioinformatics & Genetics',
        'download_as': 'elwin6.py',
        'description': 'Clinical biomarker records with train/test validation splits and classification benchmarks.'
    },
    'elwin7.py': {
        'display_name': 'svm_classifier_decision_space.csv',
        'type_label': 'CSV Table',
        'category': 'Algorithms & AI',
        'download_as': 'elwin7.py',
        'description': 'Support Vector Machine kernel performance dataset evaluating linear, polynomial, and RBF decision boundaries.'
    },
    'elwin8.py': {
        'display_name': 'spatial_coordinate_mst_clusters.csv',
        'type_label': 'CSV Table',
        'category': 'Geospatial & Clustering',
        'download_as': 'elwin8.py',
        'description': 'Spatial distance matrix and Euclidean coordinates dataset with Minimum Spanning Tree clustering.'
    },
    'elwin9.py': {
        'display_name': 'dbscan_cellular_density_clusters.csv',
        'type_label': 'CSV Table',
        'category': 'Biological Science',
        'download_as': 'elwin9.py',
        'description': 'Density-Based Spatial Clustering (DBSCAN) dataset analyzing core samples, border points, and cluster distributions.'
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
        'category': 'Credit & Lending',
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
    if name_or_display in REVERSE_MAP:
        return REVERSE_MAP[name_or_display]
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
        models = ['Toyota Corolla', 'Honda Civic', 'Hyundai Elantra', 'Ford Focus', 'BMW 320i', 'Audi A4', 'Mercedes C200', 'Volkswagen Golf', 'Nissan Sentra', 'Mazda 3']
        fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Electric']
        transmissions = ['Automatic', 'Manual']
        n = 100
        data = {
            'Car_Model': [models[i % len(models)] for i in range(n)],
            'Year': np.random.randint(2014, 2024, n),
            'Kilometers_Driven': np.random.randint(12000, 160000, n),
            'Fuel_Type': np.random.choice(fuel_types, n, p=[0.45, 0.35, 0.15, 0.05]),
            'Transmission': np.random.choice(transmissions, n, p=[0.65, 0.35]),
            'Engine_CC': np.random.choice([1197, 1498, 1798, 1995, 2498], n),
            'Power_bhp': np.random.randint(85, 245, n),
            'Price_USD': np.random.randint(8500, 48000, n)
        }
        return pd.DataFrame(data), 1250

    elif script_name == 'elwin2.py':
        purposes = ['Debt Consolidation', 'Home Improvement', 'Small Business', 'Major Purchase', 'Education']
        home_status = ['MORTGAGE', 'RENT', 'OWN']
        loan_status = ['Approved', 'Rejected']
        n = 100
        data = {
            'Applicant_ID': [f'APP-{10000 + i}' for i in range(n)],
            'Age': np.random.randint(21, 65, n),
            'Annual_Income_USD': np.random.randint(28000, 145000, n),
            'Home_Ownership': np.random.choice(home_status, n, p=[0.5, 0.4, 0.1]),
            'Employment_Years': np.random.randint(1, 25, n),
            'Loan_Purpose': np.random.choice(purposes, n),
            'Credit_Score': np.random.randint(580, 830, n),
            'Loan_Amount_USD': np.random.randint(3000, 45000, n),
            'Loan_Status': np.random.choice(loan_status, n, p=[0.74, 0.26])
        }
        return pd.DataFrame(data), 2400

    elif script_name == 'elwin3.py':
        n = 100
        area = np.random.randint(650, 4200, n)
        bedrooms = np.random.randint(1, 6, n)
        locality = np.random.randint(1, 10, n)
        price = area * np.random.uniform(110, 210, n) + bedrooms * 12000 + locality * 8000
        data = {
            'Property_ID': [f'PROP-{5000 + i}' for i in range(n)],
            'Area_sqft': area,
            'Bedrooms': bedrooms,
            'Bathrooms': np.random.randint(1, 4, n),
            'Age_Years': np.random.randint(1, 35, n),
            'Distance_City_km': np.round(np.random.uniform(1.2, 28.5, n), 1),
            'Locality_Rating': locality,
            'Valuation_USD': np.round(price, -2).astype(int)
        }
        return pd.DataFrame(data), 1800

    elif script_name == 'elwin4.py':
        try:
            from sklearn.datasets import load_breast_cancer
            bc = load_breast_cancer(as_frame=True)
            df = bc.frame.iloc[:100].copy()
            df.columns = [c.replace(' ', '_').capitalize() for c in df.columns]
            df['Diagnosis'] = df['Target'].map({0: 'Malignant', 1: 'Benign'})
            df = df.drop(columns=['Target'])
            keep_cols = ['Mean_radius', 'Mean_texture', 'Mean_perimeter', 'Mean_area', 'Mean_smoothness', 'Mean_compactness', 'Mean_concavity', 'Diagnosis']
            return df[keep_cols], 569
        except Exception:
            n = 100
            data = {
                'Mean_radius': np.round(np.random.uniform(10.5, 28.1, n), 2),
                'Mean_texture': np.round(np.random.uniform(12.0, 39.2, n), 2),
                'Mean_perimeter': np.round(np.random.uniform(65.0, 188.5, n), 1),
                'Mean_area': np.round(np.random.uniform(320.0, 2500.0, n), 1),
                'Mean_smoothness': np.round(np.random.uniform(0.06, 0.16, n), 4),
                'Mean_compactness': np.round(np.random.uniform(0.02, 0.34, n), 4),
                'Mean_concavity': np.round(np.random.uniform(0.01, 0.42, n), 4),
                'Diagnosis': np.random.choice(['Benign', 'Malignant'], n, p=[0.63, 0.37])
            }
            return pd.DataFrame(data), 569

    elif script_name == 'elwin5.py':
        try:
            from sklearn.datasets import load_wine
            wine = load_wine(as_frame=True)
            df = wine.frame.iloc[:100].copy()
            df.columns = [c.replace(' ', '_').replace('/', '_').capitalize() for c in df.columns]
            df['Quality_grade'] = df['Target'].map({0: 'Class_A', 1: 'Class_B', 2: 'Class_C'})
            df = df.drop(columns=['Target'])
            keep_cols = ['Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 'Flavanoids', 'Quality_grade']
            return df[keep_cols], 178
        except Exception:
            n = 100
            data = {
                'Alcohol': np.round(np.random.uniform(11.0, 14.8, n), 2),
                'Malic_acid': np.round(np.random.uniform(0.74, 5.80, n), 2),
                'Ash': np.round(np.random.uniform(1.36, 3.23, n), 2),
                'Alcalinity_of_ash': np.round(np.random.uniform(10.6, 30.0, n), 1),
                'Magnesium': np.random.randint(70, 162, n),
                'Total_phenols': np.round(np.random.uniform(0.98, 3.88, n), 2),
                'Flavanoids': np.round(np.random.uniform(0.34, 5.08, n), 2),
                'Quality_grade': np.random.choice(['Class_A', 'Class_B', 'Class_C'], n, p=[0.33, 0.40, 0.27])
            }
            return pd.DataFrame(data), 178

    elif script_name == 'elwin6.py':
        n = 100
        data = {
            'Patient_ID': [f'PT-{7000 + i}' for i in range(n)],
            'Age': np.random.randint(30, 80, n),
            'Tumor_Stage': np.random.choice(['Stage I', 'Stage II', 'Stage III', 'Stage IV'], n, p=[0.25, 0.4, 0.25, 0.1]),
            'HER2_Status': np.random.choice(['Positive', 'Negative'], n, p=[0.3, 0.7]),
            'Estrogen_Receptor_Pct': np.random.randint(5, 98, n),
            'CA125_U_ml': np.round(np.random.uniform(10.5, 95.0, n), 1),
            'Biomarker_Index': np.round(np.random.uniform(0.12, 4.85, n), 2),
            'Response_Outcome': np.random.choice(['Complete Response', 'Partial Response', 'Stable Disease'], n)
        }
        return pd.DataFrame(data), 850

    elif script_name == 'elwin7.py':
        n = 100
        f1 = np.random.normal(0, 1.5, n)
        f2 = np.random.normal(0, 1.5, n)
        dist = np.round(np.abs(f1 * 0.8 - f2 * 1.2 + 0.3), 3)
        labels = ['Class_Alpha' if (f1[i] + f2[i] > 0) else 'Class_Beta' for i in range(n)]
        data = {
            'Sample_ID': [f'SMP-{i+1:04d}' for i in range(n)],
            'Feature_X1': np.round(f1, 4),
            'Feature_X2': np.round(f2, 4),
            'Feature_X3': np.round(np.random.normal(2, 0.8, n), 4),
            'Kernel_Projection': np.round(np.tanh(f1 * f2), 4),
            'Margin_Distance': dist,
            'Support_Vector': ['Yes' if d < 0.45 else 'No' for d in dist],
            'Target_Class': labels
        }
        return pd.DataFrame(data), 1100

    elif script_name == 'elwin8.py':
        n = 100
        data = {
            'Node_ID': [f'ND-{100 + i}' for i in range(n)],
            'Latitude': np.round(np.random.uniform(37.70, 37.82, n), 5),
            'Longitude': np.round(np.random.uniform(-122.52, -122.38, n), 5),
            'Elevation_m': np.random.randint(5, 280, n),
            'Spatial_Density': np.round(np.random.uniform(14.2, 88.5, n), 2),
            'Neighbor_Count': np.random.randint(3, 18, n),
            'MST_Edge_Weight': np.round(np.random.uniform(0.05, 3.40, n), 3),
            'Cluster_ID': [f'Cluster_{x}' for x in np.random.choice([1, 2, 3, 4], n)]
        }
        return pd.DataFrame(data), 920

    elif script_name == 'elwin9.py':
        n = 100
        cdist = np.round(np.random.uniform(0.08, 1.95, n), 3)
        data = {
            'Cell_ID': [f'CEL-{4000 + i}' for i in range(n)],
            'Core_Distance': cdist,
            'Reachability_Dist': np.round(cdist * np.random.uniform(1.0, 1.4, n), 3),
            'Epsilon_Radius': [0.5] * n,
            'Min_Neighbors': [15] * n,
            'Point_Type': ['Core' if c < 0.6 else ('Border' if c < 1.2 else 'Noise') for c in cdist],
            'Assigned_Cluster': [('Noise' if c >= 1.2 else f'Cluster_{(i%3)+1}') for i, c in enumerate(cdist)]
        }
        return pd.DataFrame(data), 1420

    # Fallback
    n = 100
    df = pd.DataFrame({
        'Sample_ID': range(1, n + 1),
        'Measurement_A': np.round(np.random.uniform(10, 100, n), 2),
        'Measurement_B': np.round(np.random.uniform(100, 500, n), 2),
        'Classification': np.random.choice(['Category 1', 'Category 2', 'Category 3'], n)
    })
    return df, 1000

def get_file_list():
    """Scan workspace directory and return dataset files with metadata."""
    ignored = {
        'app.py', 'run_kaggle.bat', '.git', '__pycache__', 'templates', 'static', 'venv', '.env'
    }
    files = []
    for item in sorted(os.listdir(BASE_DIR)):
        if item in ignored or item.startswith('.'):
            continue
        full_path = os.path.join(BASE_DIR, item)
        if os.path.isfile(full_path):
            stat = os.stat(full_path)
            ext = os.path.splitext(item)[1].lower()
            
            meta = DATASET_MAP.get(item, {})
            display_name = meta.get('display_name', item)
            type_label = meta.get('type_label', 'CSV Table')
            category = meta.get('category', 'Dataset')
            download_as = meta.get('download_as', item)

            # File type classification for frontend
            if ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp']:
                file_type = 'image'
            else:
                file_type = 'csv'  # All tabular datasets appear as CSV in the UI
                
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
    category = meta.get('category', 'Dataset')

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
        # It's a Python script (elwin1.py .. elwin9.py): generate realistic CSV dataset!
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
    file.save(dest_path)
    
    return jsonify({
        'status': 'success',
        'message': f"Uploaded '{safe_name}' successfully",
        'filename': safe_name
    })

if __name__ == '__main__':
    print("Starting Kaggle Local Clone on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=True)
