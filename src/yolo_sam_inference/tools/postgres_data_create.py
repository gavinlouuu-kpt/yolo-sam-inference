'''
This script connects to the postgres database MinIO Tracking PostgreSQL and creates a table to store selected objects to be ingested into the yolo_sam_inference pipeline.

The table is created in the postgres database yolo_sam_inference.

The table is primarily queried with minio_path.

User will provide a partial path and the script will recursively search the MinIO Tracking PostgreSQL database for all objects that match the partial path.

The script will then create a table in the postgres database yolo_sam_inference to store the selected objects.

i.e., user may give minio_path = 'bead/20250226/12%_13um/' and the script will find all objects in the MinIO Tracking PostgreSQL database that match the partial path.
i.e., 'bead/20250226/12%_13um/4/image.0997.tiff', 'bead/20250226/12%_13um/4/image.0998.tiff', 'bead/20250226/12%_13um/2/image.0998.tiff' etc.

All common image file formats (tiff, tif, jpg, jpeg, png, bmp, gif) that reside in the directory matching the partial path will be selected.

Inference Results Schema:
- results: JSONB storing all segmentation and analysis results for an image
  - Each result object contains:
    - mask: Base64-encoded or RLE-encoded segmentation mask
    - deformability: Deformability score for detected object
    - area: Pixel area for detected object
    - area_r: Relative area value
    - circularity: Circularity measure for object
    - ch_area: Convex hull area for object
    - mean_brightness: Mean brightness value for object
    - brightness_std: Standard deviation of brightness for object
    - perimeter: Perimeter measurement for object
    - ch_perimeter: Convex hull perimeter measurement for object

Using JSONB allows storing multiple detection/segmentation results per image and supports
flexible querying of nested properties.
'''

import os
import sys
import argparse
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('postgres_data_create')

# PostgreSQL connection parameters
SOURCE_DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
SOURCE_DB_NAME = os.environ.get("POSTGRES_DB", "mlflowdb")
SOURCE_DB_USER = os.environ.get("POSTGRES_USER", "user")
SOURCE_DB_PASS = os.environ.get("POSTGRES_PASSWORD", "password")
SOURCE_DB_PORT = os.environ.get("POSTGRES_PORT", "5432")

# Target database parameters (can be the same server, different DB)
TARGET_DB_HOST = os.environ.get("TARGET_POSTGRES_HOST", SOURCE_DB_HOST)
TARGET_DB_NAME = os.environ.get("TARGET_POSTGRES_DB", "yolo_sam_inference")
TARGET_DB_USER = os.environ.get("TARGET_POSTGRES_USER", SOURCE_DB_USER)
TARGET_DB_PASS = os.environ.get("TARGET_POSTGRES_PASSWORD", SOURCE_DB_PASS)
TARGET_DB_PORT = os.environ.get("TARGET_POSTGRES_PORT", SOURCE_DB_PORT)

# Define table templates
TABLE_TEMPLATES = {
    "standard": """
        id SERIAL PRIMARY KEY,
        minio_path VARCHAR(1024) NOT NULL UNIQUE,
        size BIGINT,
        last_modified TIMESTAMP,
        content_type VARCHAR(128),
        batch_id VARCHAR(64),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        condition VARCHAR(256),
        description TEXT,
        empty BOOLEAN DEFAULT NULL,
        results JSONB DEFAULT NULL,
        error TEXT
    """,
    "experiment": """
        id SERIAL PRIMARY KEY,
        minio_path VARCHAR(1024) NOT NULL UNIQUE,
        size BIGINT,
        last_modified TIMESTAMP,
        content_type VARCHAR(128),
        experiment_id VARCHAR(64),
        sample_type VARCHAR(64),
        magnification VARCHAR(32),
        condition VARCHAR(256),
        description TEXT,
        batch_id VARCHAR(64),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        empty BOOLEAN DEFAULT NULL,
        results JSONB DEFAULT NULL,
        error TEXT
    """,
    "time_series": """
        id SERIAL PRIMARY KEY,
        minio_path VARCHAR(1024) NOT NULL UNIQUE,
        size BIGINT,
        last_modified TIMESTAMP,
        content_type VARCHAR(128),
        time_point INTEGER,
        channel VARCHAR(32),
        sequence_id VARCHAR(64),
        condition VARCHAR(256),
        description TEXT,
        batch_id VARCHAR(64),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        empty BOOLEAN DEFAULT NULL,
        results JSONB DEFAULT NULL,
        error TEXT
    """
}

def connect_to_source_db():
    """Connect to the source MinIO Tracking PostgreSQL database"""
    try:
        logger.info(f"Connecting to source database {SOURCE_DB_NAME} on {SOURCE_DB_HOST}")
        conn = psycopg2.connect(
            host=SOURCE_DB_HOST,
            database=SOURCE_DB_NAME,
            user=SOURCE_DB_USER,
            password=SOURCE_DB_PASS,
            port=SOURCE_DB_PORT,
            cursor_factory=RealDictCursor
        )
        conn.autocommit = True
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Error connecting to source database: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error connecting to source database: {e}")
        raise

def connect_to_target_db():
    """Connect to the target yolo_sam_inference PostgreSQL database"""
    try:
        logger.info(f"Connecting to target database {TARGET_DB_NAME} on {TARGET_DB_HOST}")
        
        # First connect to default database to check if our target DB exists
        temp_conn = psycopg2.connect(
            host=TARGET_DB_HOST,
            database="postgres",  # Connect to default postgres database
            user=TARGET_DB_USER,
            password=TARGET_DB_PASS,
            port=TARGET_DB_PORT
        )
        temp_conn.autocommit = True
        
        # Check if our target database exists, create it if not
        cur = temp_conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (TARGET_DB_NAME,))
        if cur.fetchone() is None:
            logger.info(f"Creating database {TARGET_DB_NAME}")
            # Close connections to the database before creating a new one
            cur.close()
            # Creating new database
            cur = temp_conn.cursor()
            # Avoid SQL injection by not using string formatting for DB name
            cur.execute(f"CREATE DATABASE {TARGET_DB_NAME}")
            cur.close()
        
        temp_conn.close()
        
        # Now connect to our target database
        conn = psycopg2.connect(
            host=TARGET_DB_HOST,
            database=TARGET_DB_NAME,
            user=TARGET_DB_USER,
            password=TARGET_DB_PASS,
            port=TARGET_DB_PORT,
            cursor_factory=RealDictCursor
        )
        conn.autocommit = True
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Error connecting to target database: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error connecting to target database: {e}")
        raise

def create_purpose_table(conn, table_name, template_type="standard"):
    """Create a condition-specific table using the specified template"""
    try:
        if template_type not in TABLE_TEMPLATES:
            logger.error(f"Template type '{template_type}' not found. Available templates: {', '.join(TABLE_TEMPLATES.keys())}")
            raise ValueError(f"Invalid template type: {template_type}")
        
        logger.info(f"Creating condition table '{table_name}' using {template_type} template")
        cursor = conn.cursor()
        
        # Create the table if it doesn't exist
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {TABLE_TEMPLATES[template_type]}
            )
        """)
        
        # Create indices for faster querying
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_minio_path ON {table_name} (minio_path);
            CREATE INDEX IF NOT EXISTS idx_{table_name}_purpose ON {table_name} (condition);
            CREATE INDEX IF NOT EXISTS idx_{table_name}_batch_id ON {table_name} (batch_id);
            CREATE INDEX IF NOT EXISTS idx_{table_name}_empty ON {table_name} (empty);
            CREATE INDEX IF NOT EXISTS idx_{table_name}_results ON {table_name} USING GIN (results);
        """)
        
        # Create additional indices based on template type
        if template_type == "experiment":
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_experiment_id ON {table_name} (experiment_id);
                CREATE INDEX IF NOT EXISTS idx_{table_name}_sample_type ON {table_name} (sample_type);
            """)
        elif template_type == "time_series":
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_time_point ON {table_name} (time_point);
                CREATE INDEX IF NOT EXISTS idx_{table_name}_sequence_id ON {table_name} (sequence_id);
            """)
        
        cursor.close()
        logger.info(f"Condition table '{table_name}' is ready")
    except Exception as e:
        logger.error(f"Error creating condition table: {e}")
        raise

def find_matching_objects(source_conn, partial_path):
    """
    Find all objects in the source database that match the partial path
    
    Args:
        source_conn: Connection to source database
        partial_path: Partial path to match, e.g. 'bead/20250226/12%_13um/'
    
    Returns:
        List of dictionaries containing object information
    """
    try:
        logger.info(f"Searching for objects matching path: {partial_path}")
        cursor = source_conn.cursor()
        
        # Format search path for SQL LIKE
        search_path = partial_path
        if not search_path.endswith('%'):
            search_path = f"{search_path}%"
        
        logger.info(f"Using search path: {search_path}")
        
        # First check if the table and schema exist
        try:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 
                    FROM information_schema.tables 
                    WHERE table_schema = 'minio_tracking' 
                    AND table_name = 'objects'
                )
            """)
            table_exists = cursor.fetchone()['exists']
            
            if not table_exists:
                logger.error("minio_tracking.objects table does not exist")
                return []
            
            # Get the columns from the table to verify structure
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'minio_tracking' 
                AND table_name = 'objects'
            """)
            columns = [row['column_name'] for row in cursor.fetchall()]
            logger.info(f"Available columns: {columns}")
            
            # Check for required columns - we need minio_path at minimum
            if 'minio_path' not in columns:
                logger.error("Required column 'minio_path' not found in table")
                return []
            
            # Create a simplified query that avoids any parameter binding issues
            # Use explicit column names instead of * for safety
            query = f"""
                SELECT 
                    id,
                    minio_path,
                    size,
                    last_modified,
                    content_type
                FROM minio_tracking.objects
                WHERE minio_path LIKE '{search_path}'
                AND (
                    minio_path LIKE '%.tiff' OR 
                    minio_path LIKE '%.tif' OR 
                    minio_path LIKE '%.jpg' OR 
                    minio_path LIKE '%.jpeg' OR 
                    minio_path LIKE '%.png' OR 
                    minio_path LIKE '%.bmp' OR
                    minio_path LIKE '%.gif'
                )
                ORDER BY minio_path
            """
            
            logger.info(f"Executing query: {query}")
            logger.info("Searching for all common image formats (tiff, tif, jpg, jpeg, png, bmp, gif)")
            cursor.execute(query)
            
            # Initialize an empty list for results
            all_results = []
            
            # Fetch results in batches to handle large result sets
            batch_size = 5000
            logger.info(f"Fetching results in batches of {batch_size}")
            
            # Fetch the first batch
            results = cursor.fetchmany(batch_size)
            batch_count = 1
            total_count = 0
            
            # Process results in batches until no more results
            while results:
                logger.info(f"Processing batch {batch_count}, containing {len(results)} records")
                total_count += len(results)
                all_results.extend(results)
                
                # Fetch the next batch
                results = cursor.fetchmany(batch_size)
                batch_count += 1
            
            logger.info(f"Query returned {total_count} results in total")
            
            # If we didn't find any results, try without the .tiff filter
            if not all_results:
                logger.info("No image files found. Checking if any objects match the path pattern...")
                simple_query = f"""
                    SELECT COUNT(*) as count
                    FROM minio_tracking.objects
                    WHERE minio_path LIKE '{search_path}'
                """
                cursor.execute(simple_query)
                count = cursor.fetchone()['count']
                logger.info(f"Found {count} objects matching the path (including non-image files)")
                
                # Check a sample of what we have in the database
                if count > 0:
                    sample_query = f"""
                        SELECT minio_path 
                        FROM minio_tracking.objects 
                        WHERE minio_path LIKE '{search_path}' 
                        LIMIT 5
                    """
                    cursor.execute(sample_query)
                    samples = cursor.fetchall()
                    logger.info(f"Sample paths: {[s['minio_path'] for s in samples]}")
                    
                    # Check for any image files in the database with a broader search
                    image_query = f"""
                        SELECT COUNT(*) as count 
                        FROM minio_tracking.objects 
                        WHERE minio_path LIKE '%.jpg' OR 
                              minio_path LIKE '%.jpeg' OR 
                              minio_path LIKE '%.png' OR 
                              minio_path LIKE '%.tiff' OR 
                              minio_path LIKE '%.tif' OR 
                              minio_path LIKE '%.bmp' OR 
                              minio_path LIKE '%.gif' 
                        LIMIT 1
                    """
                    cursor.execute(image_query)
                    image_count = cursor.fetchone()['count']
                    logger.info(f"Database contains {image_count} image files in total (based on file extension)")
                    
                    if image_count > 0:
                        logger.warning("There are image files in the database, but none match your path pattern.")
                    else:
                        logger.warning("No files with standard image extensions found in the database.")
            
            # Close cursor
            cursor.close()
            
            # Convert results to standardized objects
            standardized_objects = []
            logger.info(f"Standardizing {len(all_results)} objects")
            
            # Process in batches to avoid memory issues with large datasets
            for batch_start in range(0, len(all_results), batch_size):
                batch_end = min(batch_start + batch_size, len(all_results))
                batch = all_results[batch_start:batch_end]
                
                for row in batch:
                    # Create a new object with only the fields we need
                    obj = {
                        'id': row.get('id'),
                        'minio_path': row.get('minio_path', ''),
                        'size': row.get('size', 0),
                        'last_modified': row.get('last_modified', datetime.now()),
                        'content_type': row.get('content_type', '')
                    }
                    standardized_objects.append(obj)
                
                logger.info(f"Standardized objects {batch_start+1} to {batch_end} of {len(all_results)}")
            
            logger.info(f"Completed standardization of {len(standardized_objects)} objects")
            return standardized_objects
            
        except Exception as e:
            logger.error(f"Error executing database query: {str(e)}")
            
            # Try a last resort query to test basic database access
            try:
                cursor.execute("SELECT 1 as test")
                test_result = cursor.fetchone()
                logger.info(f"Basic database access test: {test_result}")
            except Exception as test_e:
                logger.error(f"Even basic database query failed: {str(test_e)}")
            
            return []
            
    except Exception as e:
        logger.error(f"Error finding matching objects: {str(e)}")
        return []

def insert_objects_to_purpose_table(target_conn, objects, table_name, condition, description=None, batch_id=None, template_type="standard"):
    """
    Insert the matched objects into a condition-specific table using bulk operations
    
    Args:
        target_conn: Connection to target database
        objects: List of dictionaries containing object information
        table_name: Name of the condition table
        condition: Condition identifier for this data set
        description: Optional description of this data set
        batch_id: Optional batch identifier
        template_type: Type of table template used
    """
    if not objects:
        logger.warning("No objects to insert")
        return batch_id
    
    try:
        logger.info(f"Inserting {len(objects)} objects into condition table '{table_name}'")
        cursor = target_conn.cursor()
        
        if batch_id is None:
            # Generate a batch ID based on timestamp if not provided
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create a temporary table for bulk insert
        temp_table_name = f"temp_{table_name}_{int(datetime.now().timestamp())}"
        logger.info(f"Creating temporary table {temp_table_name} for bulk insert")
        
        # Create temporary table with the same structure as target table
        if template_type == "standard":
            cursor.execute(f"""
                CREATE TEMPORARY TABLE {temp_table_name} (
                    minio_path VARCHAR(1024) NOT NULL,
                    size BIGINT,
                    last_modified TIMESTAMP,
                    content_type VARCHAR(128),
                    batch_id VARCHAR(64),
                    condition VARCHAR(256),
                    description TEXT,
                    empty BOOLEAN,
                    results JSONB
                )
            """)
            
            # Prepare data for bulk insert
            from io import StringIO
            import csv
            import json
            
            start_time = datetime.now()
            logger.info(f"Preparing data for bulk insert, started at {start_time}")
            
            # Create a buffer to hold the CSV data
            csv_data = StringIO()
            csv_writer = csv.writer(csv_data, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            # Write data to the buffer
            for obj in objects:
                # Create a null JSON for results
                csv_writer.writerow([
                    obj.get('minio_path', ''),
                    obj.get('size', 0),
                    obj.get('last_modified', datetime.now()),
                    obj.get('content_type', ''),
                    batch_id,
                    condition,
                    description,
                    False,
                    None  # results (JSON)
                ])
            
            # Reset buffer position
            csv_data.seek(0)
            
            # Execute COPY command to quickly load data
            logger.info(f"Executing bulk COPY operation to temporary table")
            cursor.copy_expert(f"COPY {temp_table_name} FROM STDIN WITH CSV DELIMITER E'\\t' QUOTE E'\"'", csv_data)
            
            # Insert from temporary table to target table with ON CONFLICT handling
            logger.info(f"Inserting from temporary table to target table with conflict handling")
            cursor.execute(f"""
                INSERT INTO {table_name} 
                (minio_path, size, last_modified, content_type, batch_id, condition, description, empty, results)
                SELECT 
                    minio_path, size, last_modified, content_type, batch_id, condition, description, empty, results
                FROM {temp_table_name}
                ON CONFLICT (minio_path) 
                DO UPDATE SET 
                    size = EXCLUDED.size,
                    last_modified = EXCLUDED.last_modified,
                    content_type = EXCLUDED.content_type,
                    batch_id = EXCLUDED.batch_id,
                    condition = EXCLUDED.condition,
                    description = EXCLUDED.description,
                    created_at = CURRENT_TIMESTAMP,
                    empty = TRUE,
                    results = COALESCE(EXCLUDED.results, {table_name}.results)
            """)
            
            # Get count of inserted records
            cursor.execute(f"SELECT COUNT(*) as count FROM {temp_table_name}")
            inserted_count = cursor.fetchone()['count']
            
        elif template_type == "experiment":
            # Extract experiment metadata in bulk
            logger.info("Processing experiment metadata in bulk")
            
            cursor.execute(f"""
                CREATE TEMPORARY TABLE {temp_table_name} (
                    minio_path VARCHAR(1024) NOT NULL,
                    size BIGINT,
                    last_modified TIMESTAMP,
                    content_type VARCHAR(128),
                    experiment_id VARCHAR(64),
                    sample_type VARCHAR(64),
                    magnification VARCHAR(32),
                    condition VARCHAR(256),
                    description TEXT,
                    batch_id VARCHAR(64),
                    empty BOOLEAN,
                    results JSONB
                )
            """)
            
            # Create a buffer for CSV data
            csv_data = StringIO()
            csv_writer = csv.writer(csv_data, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            # Write data to the buffer with experiment metadata
            for obj in objects:
                # Extract experiment metadata from path
                path_parts = obj.get('minio_path', '').split('/')
                experiment_id = path_parts[0] if len(path_parts) > 0 else 'unknown'
                sample_type = path_parts[1] if len(path_parts) > 1 else 'default'
                magnification = '40x'  # Default value
                
                csv_writer.writerow([
                    obj.get('minio_path', ''),
                    obj.get('size', 0),
                    obj.get('last_modified', datetime.now()),
                    obj.get('content_type', ''),
                    experiment_id,
                    sample_type,
                    magnification,
                    condition,
                    description,
                    batch_id,
                    False,
                    None  # results (JSON)
                ])
            
            # Reset buffer position
            csv_data.seek(0)
            
            # Execute COPY command
            cursor.copy_expert(f"COPY {temp_table_name} FROM STDIN WITH CSV DELIMITER E'\\t' QUOTE E'\"'", csv_data)
            
            # Insert from temporary table to target table
            cursor.execute(f"""
                INSERT INTO {table_name} 
                (minio_path, size, last_modified, content_type, experiment_id, sample_type, magnification, 
                 condition, description, batch_id, empty, results)
                SELECT 
                    minio_path, size, last_modified, content_type, experiment_id, sample_type, magnification,
                    condition, description, batch_id, empty, results
                FROM {temp_table_name}
                ON CONFLICT (minio_path) 
                DO UPDATE SET 
                    size = EXCLUDED.size,
                    last_modified = EXCLUDED.last_modified,
                    content_type = EXCLUDED.content_type,
                    experiment_id = EXCLUDED.experiment_id,
                    sample_type = EXCLUDED.sample_type,
                    magnification = EXCLUDED.magnification,
                    condition = EXCLUDED.condition,
                    description = EXCLUDED.description,
                    batch_id = EXCLUDED.batch_id,
                    created_at = CURRENT_TIMESTAMP,
                    empty = TRUE,
                    results = COALESCE(EXCLUDED.results, {table_name}.results)
            """)
            
            # Get count of inserted records
            cursor.execute(f"SELECT COUNT(*) as count FROM {temp_table_name}")
            inserted_count = cursor.fetchone()['count']
            
        elif template_type == "time_series":
            logger.info("Processing time series metadata in bulk")
            
            cursor.execute(f"""
                CREATE TEMPORARY TABLE {temp_table_name} (
                    minio_path VARCHAR(1024) NOT NULL,
                    size BIGINT,
                    last_modified TIMESTAMP,
                    content_type VARCHAR(128),
                    time_point INTEGER,
                    channel VARCHAR(32),
                    sequence_id VARCHAR(64),
                    condition VARCHAR(256),
                    description TEXT,
                    batch_id VARCHAR(64),
                    empty BOOLEAN,
                    results JSONB
                )
            """)
            
            # Create a buffer for CSV data
            csv_data = StringIO()
            csv_writer = csv.writer(csv_data, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            # Write data to the buffer with time series metadata
            for obj in objects:
                # Extract time series metadata from filename
                path_parts = obj.get('minio_path', '').split('/')
                filename = path_parts[-1] if path_parts else ''
                
                # Try to extract time point from filename
                time_point = 0
                if '.' in filename:
                    parts = filename.split('.')
                    if len(parts) > 1:
                        try:
                            time_point = int(parts[1])
                        except ValueError:
                            time_point = 0
                
                channel = 'default'
                sequence_id = f"seq_{condition}"
                
                csv_writer.writerow([
                    obj.get('minio_path', ''),
                    obj.get('size', 0),
                    obj.get('last_modified', datetime.now()),
                    obj.get('content_type', ''),
                    time_point,
                    channel,
                    sequence_id,
                    condition,
                    description,
                    batch_id,
                    False,
                    None  # results (JSON)
                ])
            
            # Reset buffer position
            csv_data.seek(0)
            
            # Execute COPY command
            cursor.copy_expert(f"COPY {temp_table_name} FROM STDIN WITH CSV DELIMITER E'\\t' QUOTE E'\"'", csv_data)
            
            # Insert from temporary table to target table
            cursor.execute(f"""
                INSERT INTO {table_name} 
                (minio_path, size, last_modified, content_type, time_point, channel, sequence_id,
                 condition, description, batch_id, empty, results)
                SELECT 
                    minio_path, size, last_modified, content_type, time_point, channel, sequence_id,
                    condition, description, batch_id, empty, results
                FROM {temp_table_name}
                ON CONFLICT (minio_path) 
                DO UPDATE SET 
                    size = EXCLUDED.size,
                    last_modified = EXCLUDED.last_modified,
                    content_type = EXCLUDED.content_type,
                    time_point = EXCLUDED.time_point,
                    channel = EXCLUDED.channel,
                    sequence_id = EXCLUDED.sequence_id,
                    condition = EXCLUDED.condition,
                    description = EXCLUDED.description,
                    batch_id = EXCLUDED.batch_id,
                    created_at = CURRENT_TIMESTAMP,
                    empty = TRUE,
                    results = COALESCE(EXCLUDED.results, {table_name}.results)
            """)
            
            # Get count of inserted records
            cursor.execute(f"SELECT COUNT(*) as count FROM {temp_table_name}")
            inserted_count = cursor.fetchone()['count']
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        records_per_second = inserted_count / duration if duration > 0 else 0
        
        logger.info(f"Bulk insert completed in {duration:.2f} seconds")
        logger.info(f"Inserted {inserted_count} records at {records_per_second:.2f} records per second")
        
        # Drop the temporary table
        cursor.execute(f"DROP TABLE IF EXISTS {temp_table_name}")
        
        cursor.close()
        logger.info(f"Successfully inserted {inserted_count} objects into condition table '{table_name}' with batch_id: {batch_id}")
        return batch_id
    except Exception as e:
        logger.error(f"Error inserting objects to condition table: {e}")
        raise

def list_purpose_tables(target_conn):
    """List all condition-specific tables in the target database"""
    try:
        logger.info("Listing condition tables in the target database")
        cursor = target_conn.cursor()
        
        # Query all tables in the public schema
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = [row['table_name'] for row in cursor.fetchall()]
        cursor.close()
        
        return tables
    except Exception as e:
        logger.error(f"Error listing condition tables: {e}")
        raise

def get_table_summary(target_conn, table_name):
    """Get summary information about a condition table"""
    try:
        logger.info(f"Getting summary for table '{table_name}'")
        cursor = target_conn.cursor()
        
        # Get total count
        cursor.execute(f"SELECT COUNT(*) as total FROM {table_name}")
        total_count = cursor.fetchone()['total']
        
        # Get count by condition
        cursor.execute(f"""
            SELECT condition, COUNT(*) as count
            FROM {table_name}
            GROUP BY condition
            ORDER BY count DESC
        """)
        purpose_counts = cursor.fetchall()
        
        # Get empty status
        cursor.execute(f"""
            SELECT empty, COUNT(*) as count
            FROM {table_name}
            GROUP BY empty
            ORDER BY empty
        """)
        empty_counts = cursor.fetchall()
        
        # Get inference result statistics
        # Check if the results column exists
        cursor.execute(f"""
            SELECT EXISTS (
               SELECT 1 
               FROM information_schema.columns 
               WHERE table_name = '{table_name}' 
               AND column_name = 'results'
            ) as exists_flag
        """)
        results_exists = cursor.fetchone()['exists_flag']
        
        inference_stats = {}
        if results_exists:
            # Count records with results data
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE results IS NOT NULL) as with_results,
                    COUNT(*) FILTER (WHERE results IS NULL) as without_results,
                    COUNT(*) FILTER (WHERE jsonb_array_length(results) > 0) as with_objects
                FROM {table_name}
            """)
            result_stats = cursor.fetchone()
            
            # Get more detailed stats about result properties
            cursor.execute(f"""
                SELECT
                    COUNT(*) FILTER (WHERE EXISTS (
                        SELECT 1 FROM jsonb_array_elements(results) 
                        WHERE jsonb_typeof(jsonb_array_elements.value->'mask') IS NOT NULL
                    )) as with_masks,
                    COUNT(*) FILTER (WHERE EXISTS (
                        SELECT 1 FROM jsonb_array_elements(results) 
                        WHERE jsonb_typeof(jsonb_array_elements.value->'deformability') IS NOT NULL
                    )) as with_deformability,
                    COUNT(*) FILTER (WHERE EXISTS (
                        SELECT 1 FROM jsonb_array_elements(results) 
                        WHERE jsonb_typeof(jsonb_array_elements.value->'area') IS NOT NULL
                    )) as with_area
                FROM {table_name}
                WHERE results IS NOT NULL
            """)
            property_stats = cursor.fetchone()
            
            inference_stats = {
                'total_images': result_stats['total'],
                'with_results': result_stats['with_results'],
                'without_results': result_stats['without_results'],
                'with_detected_objects': result_stats['with_objects'],
                'with_masks': property_stats['with_masks'],
                'with_deformability': property_stats['with_deformability'],
                'with_area': property_stats['with_area'],
                'percent_complete': (result_stats['with_results'] / result_stats['total'] * 100) if result_stats['total'] > 0 else 0
            }
        
        cursor.close()
        
        return {
            'table_name': table_name,
            'total_count': total_count,
            'purpose_counts': purpose_counts,
            'empty_counts': empty_counts,
            'inference_stats': inference_stats
        }
    except Exception as e:
        logger.error(f"Error getting table summary: {e}")
        raise

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='Create and manage condition-specific tables in PostgreSQL for YOLO-SAM inference pipeline')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create table command
    create_parser = subparsers.add_parser('create', help='Create a new condition-specific table')
    create_parser.add_argument('--table', type=str, required=True, help='Name of the condition table to create')
    create_parser.add_argument('--template', type=str, default='standard', choices=['standard', 'experiment', 'time_series'], 
                             help='Template type to use (standard, experiment, or time_series)')
    
    # Add data command
    add_parser = subparsers.add_parser('add', help='Add data to a condition-specific table')
    add_parser.add_argument('--path', type=str, required=True, help='Partial MinIO path to search for objects')
    add_parser.add_argument('--table', type=str, required=True, help='Name of the condition table to add data to')
    add_parser.add_argument('--condition', type=str, required=True, help='Condition identifier for this data set')
    add_parser.add_argument('--description', type=str, help='Description of this data set')
    add_parser.add_argument('--batch-id', type=str, help='Optional batch identifier')
    add_parser.add_argument('--template', type=str, default='standard', choices=['standard', 'experiment', 'time_series'], 
                          help='Template type of the table (standard, experiment, or time_series)')
    
    # List tables command
    list_parser = subparsers.add_parser('list', help='List all condition-specific tables')
    
    # Show table summary command
    summary_parser = subparsers.add_parser('summary', help='Show summary information for a condition table')
    summary_parser.add_argument('--table', type=str, required=True, help='Name of the condition table to summarize')
    
    args = parser.parse_args()
    
    # If no command is specified, show help
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Connect to target database
        target_conn = connect_to_target_db()
        
        if args.command == 'create':
            # Create a new condition table
            create_purpose_table(target_conn, args.table, args.template)
            logger.info(f"Condition table '{args.table}' created successfully")
            
        elif args.command == 'add':
            # Connect to source database
            source_conn = connect_to_source_db()
            
            # Find matching objects
            objects = find_matching_objects(source_conn, args.path)
            
            # Insert objects into the condition table
            batch_id = insert_objects_to_purpose_table(
                target_conn, 
                objects, 
                args.table, 
                args.condition, 
                args.description, 
                args.batch_id, 
                args.template
            )
            
            logger.info(f"Added {len(objects)} objects to condition table '{args.table}' with batch ID: {batch_id}")
            
            # Close source connection
            source_conn.close()
            
        elif args.command == 'list':
            # List all condition tables
            tables = list_purpose_tables(target_conn)
            print("\nAvailable condition tables:")
            for table in tables:
                print(f"  - {table}")
            print()
            
        elif args.command == 'summary':
            # Show summary for a specific table
            summary = get_table_summary(target_conn, args.table)
            print(f"\nSummary for condition table '{summary['table_name']}':")
            print(f"  Total objects: {summary['total_count']}")
            
            print("\n  Objects by condition:")
            for pc in summary['purpose_counts']:
                print(f"    {pc['condition']}: {pc['count']} objects")
            
            print("\n  Processing status:")
            for pc in summary['empty_counts']:
                status = "empty" if pc['empty'] else "Not empty"
                print(f"    {status}: {pc['count']} objects")
            
            if 'inference_stats' in summary and summary['inference_stats']:
                print("\n  Inference result statistics:")
                stats = summary['inference_stats']
                print(f"    Images with results: {stats['with_results']} of {stats['total_images']} ({stats['percent_complete']:.2f}%)")
                print(f"    Images with detected objects: {stats['with_detected_objects']}")
                print(f"    Images with masks: {stats['with_masks']}")
                print(f"    Images with deformability measurements: {stats['with_deformability']}")
                print(f"    Images with area measurements: {stats['with_area']}")
            print()
        
        # Close target connection
        target_conn.close()
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
