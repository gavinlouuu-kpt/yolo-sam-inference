import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
import json
import base64
import io
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any
import cv2
from minio import Minio
from minio.error import S3Error
from urllib.parse import unquote
from yolo_sam_inference.utils.mask_encoding import decode_binary_mask

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('postgres_result_viewer')

# PostgreSQL connection parameters
DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_NAME = os.environ.get("POSTGRES_DB", "yolo_sam_inference")
DB_USER = os.environ.get("POSTGRES_USER", "user")
DB_PASS = os.environ.get("POSTGRES_PASSWORD", "password")
DB_PORT = os.environ.get("POSTGRES_PORT", "5432")

# MinIO connection parameters
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "mibadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "cuhkminio")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "False").lower() == "true"

def connect_to_db():
    """Connect to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT,
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def connect_to_minio():
    """Connect to MinIO server"""
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        return client
    except Exception as e:
        logger.error(f"Error connecting to MinIO: {e}")
        raise

def get_image_from_minio(minio_client, bucket_name: str, object_name: str) -> np.ndarray:
    """Get image from MinIO and convert to numpy array"""
    try:
        # Get the object
        response = minio_client.get_object(bucket_name, object_name)
        image_data = response.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        # Convert from BGR to RGB if it's a color image
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    except S3Error as e:
        logger.error(f"Error getting image from MinIO: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

def convert_tiff_to_png(image: np.ndarray) -> np.ndarray:
    """Convert TIFF image to PNG format"""
    try:
        # Normalize the image to 0-255 range if it's 16-bit
        if image.dtype == np.uint16:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return image
    except Exception as e:
        logger.error(f"Error converting TIFF to PNG: {e}")
        raise

def decode_mask(mask_data: Dict[str, Any]) -> np.ndarray:
    """Decode the base64 encoded mask using the existing implementation"""
    try:
        return decode_binary_mask(mask_data)
    except Exception as e:
        logger.error(f"Error decoding mask: {str(e)}")
        logger.error(f"Mask data: {mask_data}")
        raise

def draw_results(image: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
    """Draw masks and bounding boxes on the image"""
    try:
        # Convert to RGB if grayscale
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Convert to PIL Image for drawing
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        
        for result in results:
            # Draw bounding box in red
            box = result['box']
            draw.rectangle(
                [(box['x_min'], box['y_min']), (box['x_max'], box['y_max'])],
                outline=255,  # Red in grayscale
                width=2
            )
            
            # Draw confidence score in red
            draw.text(
                (box['x_min'], box['y_min'] - 10),
                f"Conf: {result['confidence']:.2f}",
                fill=255  # Red in grayscale
            )
            
            # Draw deformability score in red
            draw.text(
                (box['x_min'], box['y_min'] - 25),
                f"Def: {result['deformability']:.5g}",
                fill=255  # Red in grayscale
            )
        
        return np.array(img)
    except Exception as e:
        logger.error(f"Error drawing results: {str(e)}")
        raise

def visualize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert binary mask to RGB for visualization"""
    try:
        # Convert binary mask to RGB
        mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        mask_rgb[mask] = [0, 255, 0]  # Green color for mask
        
        # Add transparency
        alpha = mask.astype(np.uint8) * 128  # 50% transparency
        mask_rgb = np.dstack((mask_rgb, alpha))
        
        return mask_rgb
    except Exception as e:
        logger.error(f"Error visualizing mask: {str(e)}")
        logger.error(f"Mask shape: {mask.shape}")
        raise

def list_tables(conn):
    """List all tables in the database"""
    try:
        cursor = conn.cursor()
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
        logger.error(f"Error listing tables: {e}")
        raise

def get_table_data(conn, table_name: str, limit: int = 5):
    """Get data from a specific table"""
    try:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT * FROM {table_name}
            WHERE results IS NOT NULL
            LIMIT {limit}
        """)
        data = cursor.fetchall()
        cursor.close()
        return data
    except Exception as e:
        logger.error(f"Error getting table data: {e}")
        raise

def decode_minio_path(path: str) -> tuple[str, str]:
    """Decode URL-encoded MinIO path and split into bucket and object name"""
    try:
        # Decode URL-encoded characters (e.g., %2F -> /)
        decoded_path = unquote(path)
        
        # Split into bucket and object name
        parts = decoded_path.split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid MinIO path format: {path}")
            
        bucket_name = parts[0]
        object_name = parts[1]
        
        return bucket_name, object_name
    except Exception as e:
        logger.error(f"Error decoding MinIO path: {e}")
        raise

def create_mask_overlay(image: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
    """Create an overlay of the original image with masks"""
    try:
        # Create a copy of the image
        overlay = image.copy()
        
        # Convert to RGB if grayscale
        if len(overlay.shape) == 2:  # Grayscale
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        elif overlay.shape[2] == 4:  # RGBA
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGBA2RGB)
        
        # Convert to RGBA for transparency
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2RGBA)
        
        # Overlay each mask
        for result in results:
            mask = decode_mask(result['mask'])
            
            # Create a colored mask (green with transparency)
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            colored_mask[mask] = [0, 255, 0, 128]  # Green with 50% transparency
            
            # Blend the mask with the original image
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
        
        return overlay
    except Exception as e:
        logger.error(f"Error creating mask overlay: {str(e)}")
        raise

def main():
    st.title("YOLO-SAM Inference Results Viewer")
    
    # Connect to databases
    try:
        conn = connect_to_db()
        minio_client = connect_to_minio()
        
        # Get list of tables
        tables = list_tables(conn)
        
        if not tables:
            st.error("No tables found in the database")
            return
        
        # Table selection
        selected_table = st.selectbox("Select a table", tables)
        
        # Get data from selected table (limited to 5 images)
        data = get_table_data(conn, selected_table, limit=5)
        
        if not data:
            st.warning("No results found in the selected table")
            return
        
        # Display table data
        st.subheader("Table Data (First 5 Images)")
        df = pd.DataFrame(data)
        st.dataframe(df)
        
        # Display results for each image
        st.subheader("Image Results (First 5 Images)")
        
        for i, row in enumerate(data, 1):
            if not row['results']:
                continue
                
            st.markdown(f"### Image {i}")
            
            try:
                # Get image from MinIO with URL decoding
                minio_path = row['minio_path']
                bucket_name, object_name = decode_minio_path(minio_path)
                
                image = get_image_from_minio(minio_client, bucket_name, object_name)
                
                # Convert TIFF to PNG if needed
                if minio_path.lower().endswith(('.tiff', '.tif')):
                    image = convert_tiff_to_png(image)
                
                # Parse results
                results = json.loads(row['results'])
                
                # Create columns for all visualizations
                col1, col2, col3, col4 = st.columns(4)
                
                # Create a container for headers to ensure consistent height
                header_container = st.container()
                with header_container:
                    header_cols = st.columns(4)
                    with header_cols[0]:
                        st.markdown("**Original**")
                    with header_cols[1]:
                        st.markdown("**Annotated**")
                    with header_cols[2]:
                        st.markdown("**Masks**")
                    with header_cols[3]:
                        st.markdown("**Overlay**")
                
                # Display images in their respective columns
                with col1:
                    st.image(image, caption=row['minio_path'])
                
                with col2:
                    result_image = draw_results(image, results)
                    st.image(result_image)
                
                with col3:
                    for j, result in enumerate(results, 1):
                        mask = decode_mask(result['mask'])
                        mask_rgb = visualize_mask(mask)
                        st.image(mask_rgb, caption=f"Mask {j}")
                
                with col4:
                    overlay = create_mask_overlay(image, results)
                    st.image(overlay, caption="Original + Masks")
                
                # Display metrics
                st.subheader("Metrics")
                metrics = []
                for result in results:
                    metrics.append({
                        'Confidence': result['confidence'],
                        'Deformability': f"{result['deformability']:.5g}",  # 5 significant figures
                        'Area': result['area'],
                        'Circularity': result['circularity'],
                        'Mean Brightness': result['mean_brightness']
                    })
                
                metrics_df = pd.DataFrame(metrics)
                st.dataframe(metrics_df)
                
                # Plot metrics
                fig, ax = plt.subplots()
                metrics_df.plot(kind='bar', ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error processing image {i}: {str(e)}")
                continue
        
        conn.close()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
