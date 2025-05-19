import os
import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import (ColumnDataSource, CustomJS, CheckboxGroup, ColorBar, 
                         LinearColorMapper, HoverTool, Legend, LegendItem,
                         Select, RangeSlider, Button, Div, Panel, Tabs, Slider)
from bokeh.layouts import column, row, layout
from bokeh.palettes import Viridis256, Turbo256
from bokeh.transform import linear_cmap
from bokeh.io import curdoc
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import glob
'''
add button to update kde after adjusting area, deformability filtering with webui
'''
# Try to import GPU libraries
try:
    import torch
    from torch.distributions import Normal
    HAS_GPU = torch.cuda.is_available()
    print(f"PyTorch {'with CUDA GPU acceleration' if HAS_GPU else 'CPU only'} will be used for KDE")
except ImportError:
    HAS_GPU = False
    print("PyTorch not available, using CPU for KDE calculations")

class GPUAcceleratedKDE:
    """KDE implementation that leverages PyTorch GPU acceleration if available"""
    
    def __init__(self, data_points, bandwidth=None):
        """
        Initialize KDE with data points
        
        Args:
            data_points: numpy array of shape (n_dimensions, n_points)
            bandwidth: bandwidth for kernel, or None to use Scott's rule
        """
        self.data_np = data_points
        self.n_dims, self.n_points = data_points.shape
        
        # Determine if we should use GPU (only for larger datasets)
        self.use_gpu = HAS_GPU and data_points.shape[1] > 1000
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        try:
            # Convert to torch tensor
            self.data = torch.tensor(data_points, dtype=torch.float32, device=self.device)
            
            # Calculate bandwidth using Scott's rule if not provided
            if bandwidth is None:
                # Scott's rule: n**(-1/(d+4)) where n is number of data points and d is dimensions
                self.bandwidth = data_points.std(axis=1) * (self.n_points ** (-1.0 / (self.n_dims + 4)))
            else:
                self.bandwidth = bandwidth if isinstance(bandwidth, np.ndarray) else np.array([bandwidth] * self.n_dims)
                
            # Convert bandwidth to tensor
            self.bandwidth_tensor = torch.tensor(self.bandwidth, dtype=torch.float32, device=self.device)
            
            if self.use_gpu:
                print(f"Using PyTorch GPU for KDE calculation on {self.n_points} points")
            else:
                print(f"Using PyTorch CPU for KDE calculation on {self.n_points} points")
                
        except Exception as e:
            print(f"PyTorch KDE initialization failed: {e}, falling back to CPU")
            self.use_gpu = False
            # Initialize scipy's KDE as fallback
            self.kde = gaussian_kde(data_points)
    
    def evaluate(self, points):
        """
        Evaluate the KDE at the given points
        
        Args:
            points: points at which to evaluate the KDE, shape (n_dims, n_eval_points)
        
        Returns:
            Density values at the given points
        """
        if not hasattr(self, 'kde'):  # If PyTorch initialization succeeded
            try:
                # Convert evaluation points to tensor
                points_tensor = torch.tensor(points, dtype=torch.float32, device=self.device)
                n_eval_points = points_tensor.shape[1]
                
                # Reshape tensors for broadcasting
                # data: (n_dims, n_points, 1)
                # points: (n_dims, 1, n_eval_points)
                # bandwidth: (n_dims, 1, 1)
                data_expanded = self.data.unsqueeze(2)
                points_expanded = points_tensor.unsqueeze(1)
                bandwidth_expanded = self.bandwidth_tensor.unsqueeze(1).unsqueeze(2)
                
                # Calculate squared distances: [(x_i - x'_j)/h_i]^2 for all dimensions
                # Shape: (n_dims, n_points, n_eval_points)
                squared_diffs = ((data_expanded - points_expanded) / bandwidth_expanded) ** 2
                
                # Sum across dimensions
                # Shape: (n_points, n_eval_points)
                mahalanobis_dist = squared_diffs.sum(dim=0)
                
                # Apply Gaussian kernel: exp(-0.5 * dist^2)
                # Shape: (n_points, n_eval_points)
                kernel_values = torch.exp(-0.5 * mahalanobis_dist)
                
                # Normalize by bandwidth and constant factors
                # (2π)^(-d/2) * h1^-1 * h2^-1 * ... * hd^-1
                norm_factor = (2 * torch.tensor(np.pi, device=self.device)) ** (-0.5 * self.n_dims)
                norm_factor = norm_factor * torch.prod(self.bandwidth_tensor.reciprocal())
                
                # Apply normalization to kernel values
                kernel_values = kernel_values * norm_factor
                
                # Sum contributions from all data points and normalize
                # Shape: (n_eval_points,)
                density = kernel_values.sum(dim=0) / self.n_points
                
                # Return as numpy array
                return density.cpu().numpy()
                
            except Exception as e:
                print(f"PyTorch KDE evaluation failed: {e}, falling back to CPU")
                # If PyTorch evaluation fails, initialize scipy KDE and use it
                if not hasattr(self, 'kde'):
                    self.kde = gaussian_kde(self.data_np)
                return self.kde(points)
        else:
            # Use scipy KDE if PyTorch initialization failed
            return self.kde(points)

class SQLPlotter:
    """
    A class for efficient cell data plotting using SQLite and Bokeh.
    This class optimizes the slow matplotlib-based plotting from DI_size_PAA_all.py
    and adds interactive features.
    """
    
    def __init__(self, data_path=None, db_path=None, use_gpu=None, data_dir=None):
        """
        Initialize the plotter with data or database path.
        
        Args:
            data_path (str): Path to CSV/Excel file with cell data
            db_path (str): Path to SQLite database (if data already imported)
            use_gpu (bool): Force GPU usage on or off (None = auto-detect)
            data_dir (str): Directory containing multiple CSV files to combine
        """
        if data_dir and not db_path:
            self.db_path = os.path.join(data_dir, 'combined_cell_data.db')
        elif data_path and not db_path:
            data_dir = str(Path(data_path).parent)
            self.db_path = os.path.join(data_dir, 'cell_data.db')
        else:
            self.db_path = db_path or 'cell_data.db'
        self.conn = None
        self.conditions = []
        self.markers = ['circle', 'square', 'diamond', 'triangle', 'inverted_triangle', 
                       'plus', 'asterisk', 'cross']
        
        # GPU settings
        self.use_gpu = use_gpu if use_gpu is not None else HAS_GPU
        
        # Initialize database
        if data_path:
            self._initialize_db(data_path)
        elif data_dir:
            self._initialize_from_directory(data_dir)
        else:
            self._connect_db()
            self._load_conditions()
    
    def _initialize_from_directory(self, data_dir):
        """Create and populate SQLite database from multiple CSV/Excel files in a directory"""
        print(f"Initializing database from directory: {data_dir}")
        start_time = time.time()
        
        # Find all CSV and Excel files in the directory
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        excel_files = glob.glob(os.path.join(data_dir, "*.xlsx")) + glob.glob(os.path.join(data_dir, "*.xls"))
        all_files = csv_files + excel_files
        
        if not all_files:
            raise ValueError(f"No CSV or Excel files found in directory: {data_dir}")
        
        print(f"Found {len(all_files)} data files: {', '.join(os.path.basename(f) for f in all_files)}")
        
        # Connect to database
        self._connect_db()
        
        # Create tables
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS cell_data (
            id INTEGER PRIMARY KEY,
            condition TEXT,
            area REAL,
            deformability REAL,
            area_ratio REAL,
            density REAL DEFAULT 0,
            ringratio REAL,
            timestamp_us INTEGER,
            source_file TEXT
        )
        ''')
        
        # Create index for faster querying
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_condition ON cell_data(condition)')
        
        # Process each file
        combined_data = pd.DataFrame()
        
        for file_path in tqdm(all_files, desc="Processing files"):
            try:
                print(f"Processing file: {os.path.basename(file_path)}")
                
                # Read the data file
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    data = pd.read_excel(file_path)
                
                # Convert column names to lowercase
                data.columns = data.columns.str.lower()
                
                # Check for required columns
                required_columns = ['area', 'deformability']
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    print(f"Warning: File {os.path.basename(file_path)} is missing required columns: {', '.join(missing_columns)}")
                    print("Skipping this file.")
                    continue
                
                # Check for condition column, create if missing using filename
                if 'condition' not in data.columns:
                    file_name = os.path.splitext(os.path.basename(file_path))[0]
                    print(f"Warning: 'condition' column not found in {os.path.basename(file_path)}. Using filename '{file_name}' as condition.")
                    data['condition'] = file_name
                
                # Add source file column
                data['source_file'] = os.path.basename(file_path)
                
                # Clean data: replace Excel error values with NaN
                excel_errors = ['#NAME?', '#DIV/0!', '#N/A', '#NULL!', '#NUM!', '#REF!', '#VALUE!', '#ERROR!']
                
                # Function to clean values
                def clean_value(val):
                    if isinstance(val, str) and any(err in val for err in excel_errors):
                        return np.nan
                    return val
                
                # Apply to numeric columns
                numeric_columns = ['area', 'deformability', 'area_ratio', 'ringratio']
                for col in numeric_columns:
                    if col in data.columns:
                        data[col] = data[col].apply(clean_value)
                
                # Now drop rows with NaN in required columns
                original_rows = len(data)
                data = data.dropna(subset=['area', 'deformability'])
                
                # Remove infinite values
                data = data[(data['area'] != float('inf')) & (data['area'] != float('-inf'))]
                data = data[(data['deformability'] != float('inf')) & (data['deformability'] != float('-inf'))]
                
                # Report data cleaning results
                cleaned_rows = len(data)
                if original_rows != cleaned_rows:
                    print(f"Removed {original_rows - cleaned_rows} rows with invalid data ({(original_rows - cleaned_rows)/original_rows*100:.1f}%)")
                
                # Add to combined data
                combined_data = pd.concat([combined_data, data], ignore_index=True)
                
            except Exception as e:
                print(f"Error processing file {os.path.basename(file_path)}: {e}")
                print("Skipping this file and continuing...")
        
        if combined_data.empty:
            raise ValueError("No valid data found in any of the files")
        
        print(f"Combined data has {len(combined_data)} rows from {len(all_files)} files")
        
        # Save combined CSV for reference
        combined_csv_path = os.path.join(data_dir, 'combined_data.csv')
        combined_data.to_csv(combined_csv_path, index=False)
        print(f"Saved combined data to: {combined_csv_path}")
        
        # Insert data in batches for better performance
        print("Inserting data into database...")
        batch_size = 5000
        for i in tqdm(range(0, len(combined_data), batch_size), desc="Inserting batches"):
            batch = combined_data.iloc[i:i+batch_size]
            
            # Prepare data for insertion
            values = []
            for _, row in batch.iterrows():
                try:
                    condition = row.get('condition', 'Unknown')
                    
                    # Extra validation to prevent issues with float conversion
                    try:
                        area = float(row['area'])
                        if np.isnan(area) or np.isinf(area):
                            continue
                    except (ValueError, TypeError):
                        continue
                        
                    try:
                        deformability = float(row['deformability'])
                        if np.isnan(deformability) or np.isinf(deformability):
                            continue
                    except (ValueError, TypeError):
                        continue
                    
                    # Handle optional columns gracefully
                    try:
                        area_ratio = float(row.get('area_ratio', 1.0)) if 'area_ratio' in row else 1.0
                        if np.isnan(area_ratio) or np.isinf(area_ratio):
                            area_ratio = 1.0
                    except (ValueError, TypeError):
                        area_ratio = 1.0
                        
                    try:
                        ringratio = float(row.get('ringratio', 0.0)) if 'ringratio' in row else 0.0
                        if np.isnan(ringratio) or np.isinf(ringratio):
                            ringratio = 0.0
                    except (ValueError, TypeError):
                        ringratio = 0.0
                    
                    # Handle timestamp if present
                    timestamp_us = None
                    timestamp_col = next((col for col in row.index if col.lower() == 'timestamp_us'), None)
                    if timestamp_col and timestamp_col in row:
                        try:
                            timestamp_us = int(row[timestamp_col])
                            if np.isnan(timestamp_us) or np.isinf(timestamp_us):
                                timestamp_us = None
                        except (ValueError, TypeError):
                            timestamp_us = None
                    
                    # Get source file
                    source_file = row.get('source_file', 'Unknown')
                    
                    values.append((condition, area, deformability, area_ratio, ringratio, timestamp_us, source_file))
                except (ValueError, KeyError) as e:
                    pass
            
            # Insert batch
            self.conn.executemany(
                'INSERT INTO cell_data (condition, area, deformability, area_ratio, ringratio, timestamp_us, source_file) VALUES (?, ?, ?, ?, ?, ?, ?)',
                values
            )
            self.conn.commit()
        
        print(f"Database initialization completed in {time.time() - start_time:.2f} seconds")
        
        # Load conditions
        self._load_conditions()
        
        # Calculate and store density values
        self._calculate_densities()
    
    def _initialize_db(self, data_path):
        """Create and populate SQLite database from CSV/Excel"""
        print(f"Initializing database from {data_path}...")
        start_time = time.time()
        
        # Read the data file
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(data_path)
        else:
            raise ValueError("Data file must be CSV or Excel")
        
        # Convert column names to lowercase
        data.columns = data.columns.str.lower()
        
        # Check for required columns
        required_columns = ['area', 'deformability']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Required columns missing: {', '.join(missing_columns)}")
        
        # Check for condition column, create if missing
        if 'condition' not in data.columns:
            print("Warning: 'condition' column not found. Creating default condition 'Sample'.")
            data['condition'] = 'Sample'
        
        # Clean data: replace Excel error values with NaN first
        excel_errors = ['#NAME?', '#DIV/0!', '#N/A', '#NULL!', '#NUM!', '#REF!', '#VALUE!', '#ERROR!']
        
        # Function to clean values
        def clean_value(val):
            if isinstance(val, str) and any(err in val for err in excel_errors):
                print(f"Converting Excel error '{val}' to NaN")
                return np.nan
            return val
        
        # Apply to numeric columns
        numeric_columns = ['area', 'deformability', 'area_ratio', 'ringratio']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = data[col].apply(clean_value)
        
        # Now drop rows with NaN in required columns
        data = data.dropna(subset=['area', 'deformability'])
        
        # Remove infinite values
        data = data[(data['area'] != float('inf')) & (data['area'] != float('-inf'))]
        data = data[(data['deformability'] != float('inf')) & (data['deformability'] != float('-inf'))]
        
        # Report data cleaning results
        original_rows = len(data)
        cleaned_rows = len(data.dropna())
        if original_rows != cleaned_rows:
            print(f"Removed {original_rows - cleaned_rows} rows with invalid data ({(original_rows - cleaned_rows)/original_rows*100:.1f}%)")
        
        # Connect to database
        self._connect_db()
        
        # Create tables
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS cell_data (
            id INTEGER PRIMARY KEY,
            condition TEXT,
            area REAL,
            deformability REAL,
            area_ratio REAL,
            density REAL DEFAULT 0,
            ringratio REAL,
            timestamp_us INTEGER
        )
        ''')
        
        # Create index for faster querying
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_condition ON cell_data(condition)')
        
        # Insert data in batches for better performance
        print("Inserting data into database...")
        batch_size = 5000  # Increased batch size for better performance
        for i in tqdm(range(0, len(data), batch_size), desc="Inserting batches"):
            batch = data.iloc[i:i+batch_size]
            
            # Prepare data for insertion
            values = []
            for _, row in batch.iterrows():
                try:
                    condition = row.get('condition', 'Unknown')
                    
                    # Extra validation to prevent issues with float conversion
                    try:
                        area = float(row['area'])
                        if np.isnan(area) or np.isinf(area):
                            raise ValueError(f"Invalid area value: {row['area']}")
                    except (ValueError, TypeError):
                        continue  # Skip this row
                        
                    try:
                        deformability = float(row['deformability'])
                        if np.isnan(deformability) or np.isinf(deformability):
                            raise ValueError(f"Invalid deformability value: {row['deformability']}")
                    except (ValueError, TypeError):
                        continue  # Skip this row
                    
                    # Handle optional columns gracefully
                    try:
                        area_ratio = float(row.get('area_ratio', 1.0)) if 'area_ratio' in row else 1.0
                        if np.isnan(area_ratio) or np.isinf(area_ratio):
                            area_ratio = 1.0  # Default value for invalid data
                    except (ValueError, TypeError):
                        area_ratio = 1.0
                        
                    try:
                        ringratio = float(row.get('ringratio', 0.0)) if 'ringratio' in row else 0.0
                        if np.isnan(ringratio) or np.isinf(ringratio):
                            ringratio = 0.0  # Default value for invalid data
                    except (ValueError, TypeError):
                        ringratio = 0.0
                    
                    # Handle timestamp if present
                    timestamp_us = None
                    timestamp_col = next((col for col in row.index if col.lower() == 'timestamp_us'), None)
                    if timestamp_col and timestamp_col in row:
                        try:
                            timestamp_us = int(row[timestamp_col])
                            if np.isnan(timestamp_us) or np.isinf(timestamp_us):
                                timestamp_us = None
                        except (ValueError, TypeError):
                            timestamp_us = None
                    
                    values.append((condition, area, deformability, area_ratio, ringratio, timestamp_us))
                except (ValueError, KeyError) as e:
                    print(f"Skipping row due to error: {e}")
            
            # Insert batch
            self.conn.executemany(
                'INSERT INTO cell_data (condition, area, deformability, area_ratio, ringratio, timestamp_us) VALUES (?, ?, ?, ?, ?, ?)',
                values
            )
            self.conn.commit()  # Commit after each batch
        
        print(f"Database initialization completed in {time.time() - start_time:.2f} seconds")
        
        # Load conditions
        self._load_conditions()
        
        # Calculate and store density values (for performance optimization)
        self._calculate_densities()
    
    def _connect_db(self):
        """Connect to the SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            # Use dictionary cursor instead of row factory for better DataFrame compatibility
            self.conn.row_factory = sqlite3.Row
            
            # Enable WAL mode for better concurrent performance
            self.conn.execute('PRAGMA journal_mode = WAL')
            self.conn.execute('PRAGMA synchronous = NORMAL')
            self.conn.execute('PRAGMA cache_size = 10000')
            self.conn.execute('PRAGMA temp_store = MEMORY')
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            raise
    
    def _load_conditions(self):
        """Load unique conditions from the database"""
        try:
            cursor = self.conn.execute('SELECT DISTINCT condition FROM cell_data')
            self.conditions = [row[0] for row in cursor.fetchall()]
            print(f"Loaded {len(self.conditions)} conditions: {', '.join(self.conditions)}")
        except sqlite3.Error as e:
            print(f"Error loading conditions: {e}")
            self.conditions = []
    
    def _calculate_densities(self):
        """Calculate and store density values for each condition using GPU if available"""
        print("Calculating density values for all conditions...")
        start_time = time.time()
        
        # Create index first to speed up later operations
        try:
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_id ON cell_data(id)')
        except Exception as e:
            print(f"Warning: Could not create index: {e}")
        
        # Process each condition with optimized approach
        for condition in tqdm(self.conditions, desc="Processing conditions"):
            condition_start = time.time()
            # Get data for this condition
            cursor = self.conn.execute(
                'SELECT id, area, deformability FROM cell_data WHERE condition = ?',
                (condition,)
            )
            data = cursor.fetchall()
            
            if len(data) > 5:  # Need enough points for KDE
                print(f"Calculating KDE for '{condition}' with {len(data)} points...")
                
                # Extract data for KDE calculation
                ids = [row[0] for row in data]
                points = np.array([[row[1], row[2]] for row in data])
                
                try:
                    # Create and initialize KDE (GPU if available)
                    kde = GPUAcceleratedKDE(points.T)
                    
                    # Calculate densities
                    densities = kde.evaluate(points.T)
                    
                    # Scale densities to range 0-1 for better visualization
                    min_density = densities.min()
                    max_density = densities.max()
                    if max_density > min_density:
                        scaled_densities = (densities - min_density) / (max_density - min_density)
                    else:
                        scaled_densities = densities
                    
                    # Create a direct mapping of ID to density for fast updates
                    density_map = {ids[i]: float(scaled_densities[i]) for i in range(len(ids))}
                    
                    # Use optimized bulk update with temporary table approach
                    print(f"Updating densities for {len(density_map)} points...")
                    
                    # Method 1: Direct update with transaction
                    # Much faster than using a temporary table for smaller datasets
                    if len(density_map) < 50000:
                        # Break into manageable chunks for better performance
                        chunk_size = 5000
                        chunks = [list(density_map.items())[i:i+chunk_size] 
                                 for i in range(0, len(density_map), chunk_size)]
                        
                        # Begin transaction for faster updates
                        self.conn.execute('BEGIN TRANSACTION')
                        
                        updated = 0
                        for chunk in tqdm(chunks, desc=f"Updating density chunks for {condition}"):
                            for id_val, density in chunk:
                                self.conn.execute(
                                    'UPDATE cell_data SET density = ? WHERE id = ?',
                                    (density, id_val)
                                )
                                updated += 1
                        
                        self.conn.commit()
                        print(f"Updated {updated} rows in {time.time() - condition_start:.2f} seconds")
                    
                    # Method 2: Use temporary table for very large datasets
                    else:
                        print("Large dataset detected, using temporary table approach...")
                        # Create temporary table
                        self.conn.execute('DROP TABLE IF EXISTS temp_density')
                        self.conn.execute('''
                        CREATE TABLE temp_density (
                            id INTEGER,
                            density REAL
                        )
                        ''')
                        
                        # Prepare values for bulk insert in chunks
                        batch_size = 10000
                        values = list(density_map.items())
                        
                        for i in range(0, len(values), batch_size):
                            self.conn.execute('BEGIN TRANSACTION')
                            batch = values[i:i+batch_size]
                            self.conn.executemany(
                                'INSERT INTO temp_density VALUES (?, ?)',
                                batch
                            )
                            self.conn.commit()
                        
                        # Create index on temporary table for joining
                        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_temp_id ON temp_density(id)')
                        
                        # Use a single UPDATE with JOIN for better performance
                        print("Performing bulk update...")
                        self.conn.execute('BEGIN TRANSACTION')
                        updated = self.conn.execute('''
                        UPDATE cell_data 
                        SET density = (
                            SELECT density 
                            FROM temp_density 
                            WHERE temp_density.id = cell_data.id
                        )
                        WHERE EXISTS (
                            SELECT 1 
                            FROM temp_density 
                            WHERE temp_density.id = cell_data.id
                        )
                        ''').rowcount
                        self.conn.commit()
                        
                        print(f"Updated {updated} rows in {time.time() - condition_start:.2f} seconds")
                        
                        # Drop temporary table
                        self.conn.execute('DROP TABLE IF EXISTS temp_density')
                    
                except Exception as e:
                    print(f"Error calculating KDE for condition '{condition}': {e}")
                    print("Trying fallback to CPU method...")
                    try:
                        # Fallback to CPU method
                        kde = gaussian_kde(points.T)
                        densities = kde(points.T)
                        
                        # Scale densities to range 0-1
                        min_density = densities.min()
                        max_density = densities.max()
                        if max_density > min_density:
                            scaled_densities = (densities - min_density) / (max_density - min_density)
                        else:
                            scaled_densities = densities
                        
                        # Update directly
                        self.conn.execute('BEGIN TRANSACTION')
                        for i in range(len(ids)):
                            self.conn.execute(
                                'UPDATE cell_data SET density = ? WHERE id = ?',
                                (float(scaled_densities[i]), ids[i])
                            )
                        self.conn.commit()
                        
                    except Exception as e2:
                        print(f"Fallback also failed: {e2}")
        
        # Create index on density for faster filtering/sorting
        print("Creating final index on density column...")
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_density ON cell_data(density)')
        
        # Analyze database for query optimization
        print("Analyzing database for query optimization...")
        self.conn.execute('ANALYZE')
        
        # Vacuum to optimize storage
        print("Optimizing database storage...")
        self.conn.execute('VACUUM')
        
        print(f"Density calculation completed in {time.time() - start_time:.2f} seconds")
    
    def _get_data_for_plot(self, conditions=None, filters=None):
        """
        Get data for plotting with optional filtering
        
        Args:
            conditions (list): List of conditions to include
            filters (dict): Dictionary of filters to apply
        
        Returns:
            dict: Dictionary of data frames by condition
        """
        conditions = conditions or self.conditions
        
        if not conditions:
            print("Warning: No conditions available for plotting")
            return {}
        
        # Build SQL query
        query = 'SELECT * FROM cell_data WHERE condition IN (%s)' % ','.join('?' for _ in conditions)
        params = conditions.copy()
        
        # Apply filters
        if filters:
            for col, (min_val, max_val) in filters.items():
                query += f' AND {col} BETWEEN ? AND ?'
                params.extend([min_val, max_val])
        
        # Execute query
        cursor = self.conn.execute(query, params)
        
        try:
            # Get column names from cursor description
            column_names = [description[0] for description in cursor.description]
            print(f"Database column names: {column_names}")
            
            # Fetch all rows
            rows = cursor.fetchall()
            
            if not rows:
                print(f"Warning: No data found for conditions: {', '.join(conditions)}")
                return {}
            
            # Convert to DataFrame with explicit column names
            # Method 1: Convert each row to a dict using the row's keys
            try:
                # This uses the sqlite3.Row factory to get named columns
                row_dicts = [dict(row) for row in rows]
                all_data = pd.DataFrame(row_dicts)
                print("Successfully converted rows to DataFrame using row dictionaries")
            except Exception as e:
                print(f"Error converting using row dictionaries: {e}")
                
                # Method 2: Convert rows to lists and use column names
                try:
                    # Convert rows to list of lists
                    row_values = [list(row) for row in rows]
                    all_data = pd.DataFrame(row_values, columns=column_names)
                    print("Successfully converted rows to DataFrame using explicit column names")
                except Exception as e2:
                    print(f"Error converting with explicit columns: {e2}")
                    
                    # Method 3: Direct SQL to pandas
                    try:
                        # Create a new connection without row factory for pandas
                        temp_conn = sqlite3.connect(self.db_path)
                        all_data = pd.read_sql_query(query, temp_conn, params=params)
                        temp_conn.close()
                        print("Successfully loaded data using pd.read_sql_query")
                    except Exception as e3:
                        print(f"All DataFrame conversion methods failed: {e3}")
                        return {}
            
            # Verify dataframe contents
            print(f"DataFrame shape: {all_data.shape}")
            print(f"DataFrame columns: {all_data.columns.tolist()}")
            print(f"First row: {all_data.iloc[0].to_dict() if len(all_data) > 0 else 'No data'}")
            
            # Process timestamp if present
            has_timestamp = 'timestamp_us' in all_data.columns.str.lower()
            timestamp_column = None
            
            if has_timestamp:
                # Find the actual column name (case-insensitive)
                timestamp_candidates = [col for col in all_data.columns if col.lower() == 'timestamp_us']
                if timestamp_candidates:
                    timestamp_column = timestamp_candidates[0]
                    print(f"Found timestamp column: {timestamp_column}")
                    
                    # Convert to relative timestamps per condition
                    all_data['rel_timestamp'] = 0.0  # Initialize with zeros
                    
                    # Process each condition separately to create relative timestamps
                    for condition in conditions:
                        condition_mask = all_data['condition'] == condition
                        if condition_mask.any():
                            condition_data = all_data[condition_mask]
                            
                            # Find minimum timestamp for this condition
                            try:
                                min_timestamp = condition_data[timestamp_column].min()
                                
                                # Calculate relative timestamps in seconds
                                all_data.loc[condition_mask, 'rel_timestamp'] = \
                                    (all_data.loc[condition_mask, timestamp_column] - min_timestamp) / 1e6  # Convert microseconds to seconds
                                    
                                print(f"Processed relative timestamps for condition '{condition}': " +
                                      f"min={min_timestamp}, " +
                                      f"max_relative={all_data.loc[condition_mask, 'rel_timestamp'].max():.2f} seconds")
                            except Exception as e:
                                print(f"Error processing timestamps for condition '{condition}': {e}")
                                # If timestamps can't be processed, leave as zeros
                                pass
            
            # Ensure required columns exist
            required_columns = ['condition', 'area', 'deformability', 'density']
            missing_columns = [col for col in required_columns if col not in all_data.columns]
            
            if missing_columns:
                print(f"Warning: Missing required columns: {missing_columns}")
                
                # Try to map by position if column names don't match but we have the right number of columns
                if len(all_data.columns) >= len(required_columns):
                    print("Attempting to map columns by position...")
                    column_map = {}
                    
                    # Try to identify numeric columns for area and deformability
                    numeric_cols = all_data.select_dtypes(include=np.number).columns.tolist()
                    
                    if len(numeric_cols) >= 2:
                        # Assume first two numeric columns are area and deformability
                        area_col = numeric_cols[0]
                        deform_col = numeric_cols[1]
                        
                        # Check if we need to add area
                        if 'area' in missing_columns:
                            print(f"Mapping column '{area_col}' to 'area'")
                            all_data['area'] = all_data[area_col]
                        
                        # Check if we need to add deformability
                        if 'deformability' in missing_columns:
                            print(f"Mapping column '{deform_col}' to 'deformability'")
                            all_data['deformability'] = all_data[deform_col]
                    
                    # Check for condition column
                    if 'condition' in missing_columns:
                        # Try to find a text column for condition
                        text_cols = all_data.select_dtypes(include=['object']).columns.tolist()
                        if text_cols:
                            cond_col = text_cols[0]
                            print(f"Mapping column '{cond_col}' to 'condition'")
                            all_data['condition'] = all_data[cond_col]
                        else:
                            print("Added default 'condition' column with value 'Unknown'")
                            all_data['condition'] = 'Unknown'
                    
                    # Check for density column
                    if 'density' in missing_columns:
                        if len(numeric_cols) >= 3:
                            density_col = numeric_cols[2]
                            print(f"Mapping column '{density_col}' to 'density'")
                            all_data['density'] = all_data[density_col]
                        else:
                            print("Added default 'density' column with value 0.5")
                            all_data['density'] = 0.5
                
                # Create mapping for common column name variations as a fallback
                column_mapping = {
                    'condition': ['condition', 'group', 'category', 'sample', 'cell_type'],
                    'area': ['area', 'size', 'cell_area', 'cell_size'],
                    'deformability': ['deformability', 'deform', 'def', 'deformation'],
                    'density': ['density', 'point_density', 'kde']
                }
                
                # Try to remap column names
                for required_col, variants in column_mapping.items():
                    if required_col not in all_data.columns:
                        for variant in variants:
                            if variant in all_data.columns:
                                print(f"Remapping column '{variant}' to '{required_col}'")
                                all_data[required_col] = all_data[variant]
                                break
            
            # Final check for required columns
            still_missing = [col for col in required_columns if col not in all_data.columns]
            if still_missing:
                print(f"Error: Still missing required columns after remapping: {still_missing}")
                
                # Add placeholder columns if needed
                for col in still_missing:
                    if col == 'condition':
                        all_data['condition'] = 'Unknown'
                        print("Added default 'condition' column with value 'Unknown'")
                    elif col == 'density':
                        all_data['density'] = 0.5  # Default density value
                        print("Added default 'density' column with value 0.5")
                    elif col == 'area' and len(all_data.columns) >= 3:
                        # Desperate attempt: use first numeric column as area
                        numeric_cols = all_data.select_dtypes(include=np.number).columns.tolist()
                        if numeric_cols:
                            print(f"LAST RESORT: Using column '{numeric_cols[0]}' as 'area'")
                            all_data['area'] = all_data[numeric_cols[0]]
                        else:
                            print("Cannot create reasonable default for 'area' column")
                            return {}
                    elif col == 'deformability' and len(all_data.columns) >= 4:
                        # Desperate attempt: use second numeric column as deformability
                        numeric_cols = all_data.select_dtypes(include=np.number).columns.tolist()
                        if len(numeric_cols) >= 2:
                            print(f"LAST RESORT: Using column '{numeric_cols[1]}' as 'deformability'")
                            all_data['deformability'] = all_data[numeric_cols[1]]
                        else:
                            print("Cannot create reasonable default for 'deformability' column")
                            return {}
                    else:
                        print(f"Cannot create reasonable default for '{col}' column")
                        return {}  # Can't proceed without essential columns
            
            # Split by condition
            data_by_condition = {}
            for condition in conditions:
                condition_data = all_data[all_data['condition'] == condition]
                if not condition_data.empty:
                    data_by_condition[condition] = condition_data
            
            return data_by_condition
        
        except Exception as e:
            print(f"Error processing data: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def create_interactive_plot(self, output_path=None, show_plot=True):
        """
        Create an interactive client-side Bokeh plot with HTML output
        
        Args:
            output_path (str): Path to save the HTML output
            show_plot (bool): Whether to show the plot in browser
        """
        # Get data
        data_by_condition = self._get_data_for_plot()
        
        # Check if we have data
        if not data_by_condition:
            print("No data available for plotting. Creating empty plot.")
            p = figure(width=800, height=600, title="No data available")
            p.text(x=0, y=0, text=['No data available for plotting'],
                   text_font_size='20pt', text_align='center')
            
            if output_path:
                output_file(output_path)
                save(p)
                
            if show_plot:
                show(p)
                
            return p
            
        # Combine all data into a single DataFrame with condition column
        all_data = pd.concat(data_by_condition.values(), axis=0)
        
        # Find global min/max values for scaling
        min_density = all_data['density'].min()
        max_density = all_data['density'].max()
        min_area = all_data['area'].min()
        max_area = all_data['area'].max()
        min_deformability = all_data['deformability'].min()
        max_deformability = all_data['deformability'].max()
        
        # Check for timestamp column
        has_timestamp = 'rel_timestamp' in all_data.columns
        min_timestamp = 0
        max_timestamp = 0
        
        if has_timestamp:
            min_timestamp = 0  # Relative timestamp always starts at 0
            max_timestamp = all_data['rel_timestamp'].max()
            print(f"Found relative timestamp data: range 0 to {max_timestamp:.2f} seconds")
        
        # Check for RingRatio column
        has_ringratio = 'ringratio' in all_data.columns
        
        if has_ringratio:
            # Clean RingRatio data: Filter out NaN, negative, and zero values
            # Create a filtered copy for RingRatio analysis only
            ringratio_data = all_data.copy()
            ringratio_data = ringratio_data[~ringratio_data['ringratio'].isna()]  # Remove NaN
            ringratio_data = ringratio_data[ringratio_data['ringratio'] > 0]  # Remove <= 0
            
            if len(ringratio_data) == 0:
                print("Warning: No valid RingRatio values found after filtering. Disabling RingRatio histogram.")
                has_ringratio = False
            else:
                print(f"Found {len(ringratio_data)} valid RingRatio values for plotting.")
                min_ringratio = ringratio_data['ringratio'].min()
                max_ringratio = ringratio_data['ringratio'].max()
                
                # Update the data_by_condition with filtered RingRatio values
                for condition in data_by_condition.keys():
                    cond_data = data_by_condition[condition]
                    cond_data = cond_data[~cond_data['ringratio'].isna()]
                    cond_data = cond_data[cond_data['ringratio'] > 0]
                    if len(cond_data) > 0:
                        data_by_condition[condition] = cond_data
                    else:
                        print(f"Warning: No valid RingRatio values for condition '{condition}'")
        
        # Create hover tool
        hover = HoverTool(tooltips=[
            ("Condition", "@condition"),
            ("Cell Size", "@area{0,0.00} pixels"),
            ("Deformation", "@deformability{0.0000}"),
            ("Density", "@density{0.00}")
        ])
        
        if has_ringratio:
            hover.tooltips.append(("Ring Ratio", "@ringratio{0.0000}"))
            
        if has_timestamp:
            hover.tooltips.append(("Time", "@rel_timestamp{0.00} seconds"))
        
        # Create figure
        scatter_plot = figure(
            width=800, height=600,
            tools=[hover, "pan", "wheel_zoom", "box_zoom", "reset", "save"],
            x_axis_label="Cell Size (pixels)",
            y_axis_label="Deformation",
            title="Cell Morphology Data"
        )
        
        # Style the plot
        scatter_plot.title.text_font_size = '16pt'
        scatter_plot.xaxis.axis_label_text_font_size = "14pt"
        scatter_plot.yaxis.axis_label_text_font_size = "14pt"
        scatter_plot.xaxis.major_label_text_font_size = "12pt"
        scatter_plot.yaxis.major_label_text_font_size = "12pt"
        scatter_plot.grid.grid_line_alpha = 0.3
        scatter_plot.outline_line_color = None
        scatter_plot.xaxis.axis_line_width = 2
        scatter_plot.yaxis.axis_line_width = 2
        
        # Create separate renderers for each condition with unique colors
        scatter_renderers = {}
        palette = Turbo256[::max(1, 256 // len(data_by_condition))][:len(data_by_condition)]
        
        N_CONTOUR_GRID = 30 # Resolution for the density grid for contours
        
        for i, (condition, df) in enumerate(data_by_condition.items()):
            source = ColumnDataSource(data=df)
            color = palette[i]
            renderer = scatter_plot.scatter(
                x='area', 
                y='deformability', 
                source=source,
                size=8, 
                fill_color=color,
                fill_alpha=0.7,
                line_color=None,
                legend_label=condition,
                name=f"scatter_{condition}"
            )
            
            # Prepare density grid data for contours for this condition
            density_map_data = None
            if not df.empty and 'area' in df.columns and 'deformability' in df.columns and len(df) > 5:
                try:
                    x_min, x_max = df['area'].min(), df['area'].max()
                    y_min, y_max = df['deformability'].min(), df['deformability'].max()
                    
                    if x_max > x_min and y_max > y_min: # Ensure valid range
                        xx_coords = np.linspace(x_min, x_max, N_CONTOUR_GRID)
                        yy_coords = np.linspace(y_min, y_max, N_CONTOUR_GRID)
                        xx, yy = np.meshgrid(xx_coords, yy_coords)
                        
                        eval_points = np.vstack([xx.ravel(), yy.ravel()])
                        
                        # Use the condition's specific data points for its KDE
                        condition_points = df[['area', 'deformability']].values.T
                        
                        if condition_points.shape[1] > 1: # Need at least 2 points for KDE std dev calc
                            kde = GPUAcceleratedKDE(condition_points) # Use existing GPU KDE
                            grid_densities_flat = kde.evaluate(eval_points)
                            
                            # Normalize these grid densities (0-1 for this specific grid)
                            min_d = grid_densities_flat.min()
                            max_d = grid_densities_flat.max()
                            if max_d > min_d:
                                scaled_grid_densities_flat = (grid_densities_flat - min_d) / (max_d - min_d)
                            else:
                                scaled_grid_densities_flat = np.zeros_like(grid_densities_flat) if max_d == min_d else grid_densities_flat
                            
                            grid_densities_2d = scaled_grid_densities_flat.reshape((N_CONTOUR_GRID, N_CONTOUR_GRID))
                            
                            density_map_data = {
                                'X_coords': xx_coords.tolist(), # Store 1D arrays for X and Y axes
                                'Y_coords': yy_coords.tolist(),
                                'Z_grid': grid_densities_2d.tolist() # Store 2D Z values
                            }
                        else:
                            print(f"Condition '{condition}' has insufficient points ({condition_points.shape[1]}) for KDE grid.")
                    else:
                        print(f"Condition '{condition}' has zero range for area or deformability, skipping contour grid.")
                except Exception as e:
                    print(f"Error generating density grid for condition '{condition}': {e}")

            contour_source = ColumnDataSource(data=dict(xs=[], ys=[]))
            contour_renderer = scatter_plot.multi_line(
                xs='xs', ys='ys', source=contour_source, 
                line_width=2, color=color, alpha=0.6 # Use same color as points
            )
            contour_renderer.visible = True # Initially visible, JS will manage based on data

            scatter_renderers[condition] = {
                'renderer': renderer,
                'source': source,
                'original_data': ColumnDataSource(data=df),
                'color': color,
                'density_map_data': density_map_data,
                'contour_source': contour_source,
                'contour_renderer': contour_renderer
            }
        
        # Configure legend
        scatter_plot.legend.click_policy = "hide"
        scatter_plot.legend.location = "top_right"
        
        # Create RingRatio histogram plot if data available
        ringratio_plot = None
        hist_renderers = {}
        
        if has_ringratio:
            # Create a new figure for the histogram
            ringratio_plot = figure(
                width=800, height=600,
                tools=["pan", "wheel_zoom", "box_zoom", "reset", "save"],
                x_axis_label="Ring Ratio",
                y_axis_label="Count",
                title="Ring Ratio Distribution"
            )
            
            # Style the histogram plot
            ringratio_plot.title.text_font_size = '16pt'
            ringratio_plot.xaxis.axis_label_text_font_size = "14pt"
            ringratio_plot.yaxis.axis_label_text_font_size = "14pt"
            ringratio_plot.xaxis.major_label_text_font_size = "12pt"
            ringratio_plot.yaxis.major_label_text_font_size = "12pt"
            ringratio_plot.grid.grid_line_alpha = 0.3
            ringratio_plot.outline_line_color = None
            ringratio_plot.xaxis.axis_line_width = 2
            ringratio_plot.yaxis.axis_line_width = 2
            
            # Create histograms for each condition
            for i, (condition, df) in enumerate(data_by_condition.items()):
                if 'ringratio' in df.columns and not df['ringratio'].isnull().all():
                    # Filter out invalid values for histogram
                    valid_data = df[~df['ringratio'].isna()]  # Remove NaN
                    valid_data = valid_data[valid_data['ringratio'] > 0]  # Remove <= 0
                    
                    if len(valid_data) > 0:
                        # Create histogram data - using a standardized approach for all conditions
                        hist, edges = np.histogram(valid_data['ringratio'], 
                                                bins=30, 
                                                range=(min_ringratio, max_ringratio))
                        
                        # Create histogram data source
                        hist_source = ColumnDataSource({
                            'top': hist,
                            'left': edges[:-1],
                            'right': edges[1:],
                            'condition': [condition] * len(hist)
                        })
                        
                        # Plot histogram as rectangles
                        color = palette[i]
                        
                        # Create quadrant renderer for the histogram
                        renderer = ringratio_plot.quad(
                            top='top',
                            bottom=0,
                            left='left',
                            right='right',
                            source=hist_source,
                            line_color="white",
                            fill_color=color,
                            fill_alpha=0.7,
                            legend_label=condition,
                            name=f"hist_{condition}"
                        )
                        
                        # Store histogram renderer info
                        hist_renderers[condition] = {
                            'renderer': renderer,
                            'source': hist_source,
                            'original_data': ColumnDataSource(data=valid_data),
                            'color': color
                        }
            
            # Configure legend for histogram plot
            ringratio_plot.legend.click_policy = "hide"
            ringratio_plot.legend.location = "top_right"
        
        # Create widgets
        
        # Condition selection
        conditions = list(data_by_condition.keys())
        condition_select = CheckboxGroup(
            labels=conditions,
            active=list(range(len(conditions))),
            width=200
        )
        
        # Global filters
        area_slider = RangeSlider(
            title="Cell Size Range (pixels)",
            start=min_area, 
            end=max_area,
            value=(min_area, max_area),
            step=(max_area - min_area) / 100,
            width=400
        )
        
        deform_slider = RangeSlider(
            title="Deformability Range",
            start=min_deformability, 
            end=max_deformability,
            value=(min_deformability, max_deformability),
            step=(max_deformability - min_deformability) / 100,
            width=400
        )
        
        density_slider = RangeSlider(
            title="Density Range",
            start=min_density, 
            end=max_density,
            value=(min_density, max_density),
            step=(max_density - min_density) / 100,
            width=400
        )
        
        # Timestamp slider if data available
        timestamp_slider = None
        if has_timestamp:
            timestamp_slider = RangeSlider(
                title="Time Range (seconds)",
                start=min_timestamp, 
                end=max_timestamp,
                value=(min_timestamp, max_timestamp),
                step=(max_timestamp - min_timestamp) / 100 if max_timestamp > min_timestamp else 0.1,
                width=400
            )
        
        # Create RingRatio slider if data available
        ringratio_slider = None
        if has_ringratio:
            ringratio_slider = RangeSlider(
                title="Ring Ratio Range",
                start=min_ringratio, 
                end=max_ringratio,
                value=(min_ringratio, max_ringratio),
                step=(max_ringratio - min_ringratio) / 100,
                width=400
            )
        
        # Histogram settings
        bins_slider = None
        if has_ringratio:
            bins_slider = Slider(
                title="Histogram Bins",
                start=5, 
                end=50,
                value=30,
                step=1,
                width=400
            )
        
        opacity_slider = Slider(
            title="Point Opacity",
            start=0.1, 
            end=1.0,
            value=0.3,
            step=0.05,
            width=400
        )
        
        point_size_slider = Slider(
            title="Point Size",
            start=2, 
            end=15,
            value=5,
            step=1,
            width=400
        )
        
        # Density Contour Slider
        contour_level_slider = Slider(
            title="Contour Density Level", 
            start=0.0, 
            end=1.0, 
            value=0.5, 
            step=0.05, 
            width=400
        )

        contour_opacity_slider = Slider(
            title="Contour Line Opacity",
            start=0.1, 
            end=1.0,
            value=0.7, # Default to more visible
            step=0.05,
            width=400
        )

        contour_width_slider = Slider(
            title="Contour Line Width",
            start=1, 
            end=5,
            value=2, # Default line width
            step=0.5,
            width=400
        )
        
        # Stats display
        stats_div = Div(text="", width=400)
        
        # Create JavaScript callback code
        js_code = """
        // Marching Squares function to get contour line segments
        // Z_grid: 2D array of density values
        // X_coords: 1D array of x-coordinates for grid columns
        // Y_coords: 1D array of y-coordinates for grid rows
        // level: contour level
        function getMarchingSquaresSegments(Z_grid, X_coords, Y_coords, level) {
            const lines_xs = []; 
            const lines_ys = []; 
            
            if (!Z_grid || Z_grid.length === 0 || Z_grid[0].length === 0) {
                return {xs: [], ys: []};
            }
            if (!X_coords || X_coords.length === 0 || !Y_coords || Y_coords.length === 0) {
                return {xs: [], ys: []};
            }

            const num_rows = Z_grid.length;
            const num_cols = Z_grid[0].length;

            function interpolate(val1, val2, coord1, coord2, target_level) {
                if (Math.abs(val1 - val2) < 1e-9) return coord1; 
                return coord1 + (coord2 - coord1) * (target_level - val1) / (val2 - val1);
            }

            for (let r = 0; r < num_rows - 1; r++) {
                for (let c = 0; c < num_cols - 1; c++) {
                    const x_p1 = X_coords[c];
                    const x_p2 = X_coords[c+1];
                    const y_p1 = Y_coords[r]; 
                    const y_p2 = Y_coords[r+1];

                    const values = [
                        Z_grid[r][c],     // Top-left (v0)
                        Z_grid[r][c+1],   // Top-right (v1)
                        Z_grid[r+1][c+1], // Bottom-right (v2)
                        Z_grid[r+1][c]    // Bottom-left (v3)
                    ];

                    let case_index = 0;
                    if (values[0] > level) case_index |= 1;
                    if (values[1] > level) case_index |= 2;
                    if (values[2] > level) case_index |= 4;
                    if (values[3] > level) case_index |= 8;
                    
                    const pt0_x = interpolate(values[0], values[1], x_p1, x_p2, level);
                    const pt0 = [pt0_x, y_p1];
                    const pt1_y = interpolate(values[1], values[2], y_p1, y_p2, level);
                    const pt1 = [x_p2, pt1_y];
                    const pt2_x = interpolate(values[3], values[2], x_p1, x_p2, level); // Note: v3, v2 for bottom edge x-interp
                    const pt2 = [pt2_x, y_p2];
                    const pt3_y = interpolate(values[0], values[3], y_p1, y_p2, level);
                    const pt3 = [x_p1, pt3_y];

                    const edge_points = [pt0, pt1, pt2, pt3];
                    
                    const line_definitions = [
                        [],                            // 0
                        [[3,0]],                       // 1
                        [[0,1]],                       // 2
                        [[3,1]],                       // 3
                        [[1,2]],                       // 4
                        [[3,0], [1,2]],                // 5
                        [[0,2]],                       // 6
                        [[3,2]],                       // 7
                        [[2,3]],                       // 8
                        [[2,0]],                       // 9
                        [[0,1], [2,3]],                // 10
                        [[2,1]],                       // 11
                        [[1,3]],                       // 12
                        [[1,0]],                       // 13
                        [[0,3]],                       // 14
                        []                             // 15
                    ];

                    const segments_for_cell = line_definitions[case_index];
                    for (let i = 0; i < segments_for_cell.length; i++) {
                        const point_indices = segments_for_cell[i];
                        const start_point = edge_points[point_indices[0]];
                        const end_point = edge_points[point_indices[1]];
                        lines_xs.push([start_point[0], end_point[0]]);
                        lines_ys.push([start_point[1], end_point[1]]);
                    }
                }
            }
            return {xs: lines_xs, ys: lines_ys};
        }

        // Get selected conditions
        const selected_indices = condition_select.active;
        const selected_conditions = [];
        for (let i = 0; i < selected_indices.length; i++) {
            selected_conditions.push(conditions[selected_indices[i]]);
        }
        
        // Stats tracking
        let total_visible = 0;
        const stats = {};
        
        const current_contour_level = contour_level_slider.value;
        const general_opacity = opacity_slider.value; // For points
        const current_contour_opacity = contour_opacity_slider.value; // For contour lines
        const current_contour_width = contour_width_slider.value; // For contour lines

        // Update each scatter renderer
        for (const condition in scatter_renderers) {
            const renderer_info = scatter_renderers[condition];
            const renderer = renderer_info.renderer;
            const source = renderer_info.source;
            const original_data = renderer_info.original_data.data;
            const contour_source = renderer_info.contour_source;
            const contour_renderer = renderer_info.contour_renderer;
            const density_map_data = renderer_info.density_map_data;
            
            // Set visibility
            renderer.visible = selected_conditions.includes(condition);
            
            if (renderer.visible) {
                // Apply filters to points
                const filtered_area = [];
                const filtered_deform = [];
                const filtered_density = [];
                const filtered_condition = [];
                const filtered_ringratio = []; // Ensure this is declared
                const filtered_timestamp = []; // Ensure this is declared
                
                for (let i = 0; i < original_data.area.length; i++) {
                    if (
                        original_data.area[i] >= area_slider.value[0] &&
                        original_data.area[i] <= area_slider.value[1] &&
                        original_data.deformability[i] >= deform_slider.value[0] &&
                        original_data.deformability[i] <= deform_slider.value[1] &&
                        original_data.density[i] >= density_slider.value[0] &&
                        original_data.density[i] <= density_slider.value[1] &&
                        (!has_ringratio || !ringratio_slider || // Check if ringratio_slider exists
                         (typeof original_data.ringratio[i] === 'number' &&
                          !isNaN(original_data.ringratio[i]) &&
                          original_data.ringratio[i] > 0 && // Ensure positive for log or some calcs
                          original_data.ringratio[i] >= ringratio_slider.value[0] &&
                          original_data.ringratio[i] <= ringratio_slider.value[1])) &&
                        (!has_timestamp || !timestamp_slider || // Check if timestamp_slider exists
                         (typeof original_data.rel_timestamp[i] === 'number' &&
                          !isNaN(original_data.rel_timestamp[i]) &&
                          original_data.rel_timestamp[i] >= timestamp_slider.value[0] &&
                          original_data.rel_timestamp[i] <= timestamp_slider.value[1]))
                    ) {
                        filtered_area.push(original_data.area[i]);
                        filtered_deform.push(original_data.deformability[i]);
                        filtered_density.push(original_data.density[i]);
                        filtered_condition.push(original_data.condition[i]);
                        
                        if (has_ringratio && original_data.ringratio && typeof original_data.ringratio[i] === 'number') {
                            filtered_ringratio.push(original_data.ringratio[i]);
                        }
                        
                        if (has_timestamp && original_data.rel_timestamp && typeof original_data.rel_timestamp[i] === 'number') {
                            filtered_timestamp.push(original_data.rel_timestamp[i]);
                        }
                    }
                }
                
                const new_data = {
                    'area': filtered_area,
                    'deformability': filtered_deform,
                    'density': filtered_density,
                    'condition': filtered_condition
                };
                
                if (has_ringratio && filtered_ringratio.length > 0) {
                    new_data['ringratio'] = filtered_ringratio;
                }
                
                if (has_timestamp && filtered_timestamp.length > 0) {
                    new_data['rel_timestamp'] = filtered_timestamp;
                }
                
                source.data = new_data;
                
                renderer.glyph.fill_alpha = general_opacity;
                renderer.glyph.size = point_size_slider.value;
                
                stats[condition] = {
                    visible: filtered_area.length,
                    total: original_data.area.length
                };
                total_visible += filtered_area.length;

                // Update contours for visible and filtered condition
                if (density_map_data && contour_source && contour_renderer) {
                    const contour_data = getMarchingSquaresSegments(
                        density_map_data.Z_grid, 
                        density_map_data.X_coords, 
                        density_map_data.Y_coords, 
                        current_contour_level
                    );
                    contour_source.data = contour_data;
                    contour_renderer.glyph.line_alpha = current_contour_opacity; // Use dedicated opacity
                    contour_renderer.glyph.line_width = current_contour_width; // Use dedicated width
                    contour_renderer.visible = true;
                } else if (contour_source && contour_renderer) { // Ensure cleanup if no map data
                    contour_source.data = {xs:[], ys:[]};
                    contour_renderer.visible = false;
                }

            } else { // if renderer is not visible
                stats[condition] = { 
                    visible: 0, 
                    total: original_data.area.length 
                };
                // Hide contours if condition scatter is not visible
                if (contour_source && contour_renderer) {
                    contour_source.data = {xs:[], ys:[]};
                    contour_renderer.visible = false;
                }
            }
        }
        
        // Update histograms if they exist (existing code)
        if (has_ringratio && typeof hist_renderers !== 'undefined' && typeof bins_slider !== 'undefined' && ringratio_slider) {
            const bins = bins_slider.value;
            const min_rr_filter = ringratio_slider.value[0]; // Renamed to avoid conflict
            const max_rr_filter = ringratio_slider.value[1]; // Renamed to avoid conflict
            
            for (const condition in hist_renderers) {
                const hist_info = hist_renderers[condition];
                const hist_renderer = hist_info.renderer;
                const hist_source = hist_info.source;
                // Use the already filtered data from the scatter plot's original_data for consistency in filtering for histogram
                // This requires that hist_info.original_data points to the same full dataset as scatter_renderers[condition].original_data
                // And that we filter it here again based on *all* sliders
                const original_hist_data = scatter_renderers[condition].original_data.data; // Use scatter's full original data
                
                hist_renderer.visible = selected_conditions.includes(condition);
                
                if (hist_renderer.visible) {
                    const filtered_hist_ringratio = [];
                    if (original_hist_data.ringratio) {
                        for (let i = 0; i < original_hist_data.ringratio.length; i++) {
                            if (
                                typeof original_hist_data.ringratio[i] === 'number' &&
                                !isNaN(original_hist_data.ringratio[i]) &&
                                original_hist_data.ringratio[i] > 0 &&
                                original_hist_data.area[i] >= area_slider.value[0] &&
                                original_hist_data.area[i] <= area_slider.value[1] &&
                                original_hist_data.deformability[i] >= deform_slider.value[0] &&
                                original_hist_data.deformability[i] <= deform_slider.value[1] &&
                                original_hist_data.density[i] >= density_slider.value[0] &&
                                original_hist_data.density[i] <= density_slider.value[1] &&
                                original_hist_data.ringratio[i] >= min_rr_filter && // Use renamed var
                                original_hist_data.ringratio[i] <= max_rr_filter && // Use renamed var
                                (!has_timestamp || !timestamp_slider ||
                                 (typeof original_hist_data.rel_timestamp[i] === 'number' &&
                                  !isNaN(original_hist_data.rel_timestamp[i]) &&
                                  original_hist_data.rel_timestamp[i] >= timestamp_slider.value[0] &&
                                  original_hist_data.rel_timestamp[i] <= timestamp_slider.value[1]))
                            ) {
                                filtered_hist_ringratio.push(original_hist_data.ringratio[i]);
                            }
                        }
                    }
                    
                    if (filtered_hist_ringratio.length > 0) {
                        const bin_edges = [];
                        // Calculate actual min/max from filtered_hist_ringratio for histogram range, or use slider range
                        let actual_min_rr_for_hist = min_rr_filter;
                        let actual_max_rr_for_hist = max_rr_filter;
                        // Optional: Recalculate min/max from filtered_hist_ringratio if you want bins to adapt to filtered data
                        // actual_min_rr_for_hist = Math.min(...filtered_hist_ringratio);
                        // actual_max_rr_for_hist = Math.max(...filtered_hist_ringratio);
                        // if (actual_min_rr_for_hist >= actual_max_rr_for_hist) actual_max_rr_for_hist = actual_min_rr_for_hist + 1e-9;


                        const bin_width = (actual_max_rr_for_hist - actual_min_rr_for_hist) / bins;
                        
                        for (let i = 0; i <= bins; i++) {
                            bin_edges.push(actual_min_rr_for_hist + i * bin_width);
                        }
                        
                        const bin_counts = Array(bins).fill(0);
                        
                        for (let i = 0; i < filtered_hist_ringratio.length; i++) {
                            const value = filtered_hist_ringratio[i];
                            if (value === actual_max_rr_for_hist && bin_width > 0) { // ensure bin_width > 0
                                bin_counts[bins - 1]++;
                            } else if (bin_width > 0) { // ensure bin_width > 0
                                const bin_index = Math.floor((value - actual_min_rr_for_hist) / bin_width);
                                if (bin_index >= 0 && bin_index < bins) {
                                    bin_counts[bin_index]++;
                                }
                            } else if (bins === 1 && value >= actual_min_rr_for_hist && value <= actual_max_rr_for_hist) { // single bin case
                                bin_counts[0]++;
                            }
                        }
                        
                        const hist_data_new = { // Renamed to avoid conflict
                            'top': bin_counts,
                            'left': bin_edges.slice(0, -1),
                            'right': bin_edges.slice(1),
                            'condition': Array(bins).fill(condition)
                        };
                        hist_source.data = hist_data_new;
                        hist_renderer.glyph.fill_alpha = general_opacity;
                    } else {
                        hist_source.data = {'top': [], 'left': [], 'right': [], 'condition': []}; // Clear histogram
                    }
                } else { // if hist_renderer not visible
                     hist_source.data = {'top': [], 'left': [], 'right': [], 'condition': []}; // Clear histogram
                }
            }
        }
        
        // Update statistics display
        let stats_html = "<h3>Statistics</h3>";
        for (const condition of selected_conditions) {
            if (stats[condition]) {
                const visible = stats[condition].visible;
                const total = stats[condition].total;
                const percentage = total > 0 ? (visible / total * 100).toFixed(1) : 0;
                stats_html += `<p><b>${condition}</b>: ${visible} cells visible out of ${total} total (${percentage}%)</p>`;
            }
        }
        stats_html += `<p><b>Total visible</b>: ${total_visible} cells</p>`;
        stats_div.text = stats_html;
        """
        
        # Create args dict for JS callback
        js_args = {
            'condition_select': condition_select,
            'conditions': conditions,
            'area_slider': area_slider,
            'deform_slider': deform_slider,
            'density_slider': density_slider,
            'opacity_slider': opacity_slider,
            'point_size_slider': point_size_slider,
            'stats_div': stats_div,
            'scatter_renderers': scatter_renderers,
            'has_ringratio': has_ringratio,
            'has_timestamp': has_timestamp,
            'contour_level_slider': contour_level_slider,
            'contour_opacity_slider': contour_opacity_slider,
            'contour_width_slider': contour_width_slider
        }
        
        if has_ringratio:
            js_args['ringratio_slider'] = ringratio_slider
            js_args['hist_renderers'] = hist_renderers
            js_args['bins_slider'] = bins_slider
            
        if has_timestamp:
            js_args['timestamp_slider'] = timestamp_slider
        
        # Create JS callback
        js_callback = CustomJS(args=js_args, code=js_code)
        
        # Connect JS callback to all widgets
        condition_select.js_on_change('active', js_callback)
        area_slider.js_on_change('value', js_callback)
        deform_slider.js_on_change('value', js_callback)
        density_slider.js_on_change('value', js_callback)
        opacity_slider.js_on_change('value', js_callback)
        point_size_slider.js_on_change('value', js_callback)
        
        # Connect contour slider
        contour_level_slider.js_on_change('value', js_callback)
        
        # Connect new contour appearance sliders
        contour_opacity_slider.js_on_change('value', js_callback)
        contour_width_slider.js_on_change('value', js_callback)
        
        if has_ringratio:
            ringratio_slider.js_on_change('value', js_callback)
            bins_slider.js_on_change('value', js_callback)
            
        if has_timestamp:
            timestamp_slider.js_on_change('value', js_callback)
        
        # Initialize the statistics display
        initial_stats_html = "<h3>Statistics</h3>"
        for condition, render_info in scatter_renderers.items():
            count = len(render_info['original_data'].data['area'])
            initial_stats_html += f"<p><b>{condition}</b>: {count} cells visible out of {count} total (100.0%)</p>"
        
        total_count = sum(len(render_info['original_data'].data['area']) for render_info in scatter_renderers.values())
        initial_stats_html += f"<p><b>Total visible</b>: {total_count} cells</p>"
        stats_div.text = initial_stats_html
        
        # Create controls layout
        filters_controls = column(
            Div(text="<h3>Filter Data:</h3>"),
            area_slider,
            deform_slider,
            density_slider
        )
        
        if has_timestamp:
            filters_controls.children.append(timestamp_slider)
            
        if has_ringratio:
            filters_controls.children.append(ringratio_slider)
        
        appearance_controls = column(
            Div(text="<h3>Appearance:</h3>"),
            opacity_slider,
            point_size_slider
        )
        
        if has_ringratio:
            appearance_controls.children.append(bins_slider)
        
        # Add contour slider to controls layout
        contour_controls = column(
            Div(text="<h3>Contour Lines:</h3>"),
            contour_level_slider,
            contour_opacity_slider,
            contour_width_slider
        )

        controls = column(
            Div(text="<h2>Cell Data Visualization</h2>"),
            Div(text="<h3>Select Conditions:</h3>"),
            condition_select,
            filters_controls,
            appearance_controls,
            contour_controls, # Add new contour controls
            stats_div
        )
        
        # Create tabs for scatter plot and histogram
        if has_ringratio and ringratio_plot:
            # Fix the Panel constructor - title is not a valid attribute according to the error
            # Need to check Bokeh version and use the appropriate method
            try:
                # First, try importing with newer API
                from bokeh.models import TabPanel
                
                tabs = Tabs(tabs=[
                    TabPanel(child=scatter_plot, title="Scatter Plot"),
                    TabPanel(child=ringratio_plot, title="Ring Ratio Histogram")
                ])
                print("Using TabPanel for newer Bokeh versions")
            except ImportError:
                # Fallback method for older Bokeh versions
                print("TabPanel not found, trying alternative Bokeh Panel approach")
                try:
                    # Try the older Panel approach
                    tabs = Tabs(tabs=[
                        Panel(child=scatter_plot, name="scatter_plot"),
                        Panel(child=ringratio_plot, name="histogram")
                    ])
                    # Set titles manually if supported
                    try:
                        tabs.tabs[0].title = "Scatter Plot"
                        tabs.tabs[1].title = "Ring Ratio Histogram"
                    except:
                        print("Could not set tab titles, using plain tabs")
                except Exception as e:
                    print(f"Error creating tabs: {e}")
                    # Last resort: just use the scatter plot
                    print("Falling back to scatter plot only due to tab creation error")
                    final_layout = layout([[controls, scatter_plot]])
                    if has_ringratio and ringratio_plot:
                        # Still add the histogram below as another panel
                        final_layout = layout([
                            [controls, scatter_plot],
                            [Div(text="<h2>Ring Ratio Histogram</h2>", width=200), ringratio_plot]
                        ])
                    return final_layout
            
            final_layout = layout([[controls, tabs]])
        else:
            final_layout = layout([[controls, scatter_plot]])
        
        # Handle output
        if output_path:
            output_file(output_path)
            save(final_layout)
            print(f"Plot saved to {output_path}")
        
        if show_plot:
            show(final_layout)
        
        return final_layout
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='Create interactive cell data plots')
    parser.add_argument('--data', type=str, help='Path to CSV or Excel data file')
    parser.add_argument('--dir', type=str, help='Path to directory containing multiple CSV/Excel files to combine')
    parser.add_argument('--db', type=str, help='Path to SQLite database (if data already imported)')
    parser.add_argument('--output', type=str, help='Output HTML file path (defaults to same location as data file)')
    parser.add_argument('--no-show', action='store_true', help='Do not show plot in browser')
    parser.add_argument('--use-gpu', action='store_true', help='Force GPU usage for calculations')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.data and not args.db and not args.dir:
        print("Error: Either --data, --dir, or --db argument is required")
        parser.print_help()
        return
    
    # Set default output path based on input
    if args.data and not args.output:
        data_path = Path(args.data)
        args.output = str(data_path.with_suffix('.html'))
        print(f"Output path not specified. Using: {args.output}")
    elif args.dir and not args.output:
        dir_path = Path(args.dir)
        args.output = str(dir_path / 'combined_plot.html')
        print(f"Output path not specified. Using: {args.output}")
    
    # Determine GPU usage
    use_gpu = None
    if args.use_gpu:
        use_gpu = True
    elif args.no_gpu:
        use_gpu = False
    
    # Create plotter
    try:
        if args.dir:
            plotter = SQLPlotter(data_dir=args.dir, use_gpu=use_gpu)
        else:
            plotter = SQLPlotter(data_path=args.data, db_path=args.db, use_gpu=use_gpu)
        
        # Create and show static plot
        plotter.create_interactive_plot(
            output_path=args.output,
            show_plot=not args.no_show
        )
        
        # Clean up
        plotter.close()
        
        print(f"\nInteractive plot has been created and saved to {args.output}")
        print("You can open this HTML file in any modern web browser to view and interact with the plot.")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
