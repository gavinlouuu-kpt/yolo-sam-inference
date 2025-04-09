def process_batch(
    raw_db_data: pd.DataFrame,
    pg_results: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a batch of images, updating the results DataFrame with new predictions.
    
    Args:
        raw_db_data: DataFrame containing raw database records
        pg_results: DataFrame containing existing results
        parameters: Dictionary of parameters
        
    Returns:
        Tuple of (updated raw_db_data, updated pg_results)
    """
    # Create a copy of the results DataFrame to avoid modifying the original
    updated_results = pg_results.copy()
    
    # Get the batch_id from the first record
    batch_id = raw_db_data.iloc[0]['batch_id']
    
    # Filter out records that are already in the results using minio_path
    existing_paths = set(pg_results['minio_path'].unique())
    new_records = raw_db_data[~raw_db_data['minio_path'].isin(existing_paths)]
    
    if len(new_records) == 0:
        logger.info(f"No new records to process for batch {batch_id}")
        return raw_db_data, pg_results
    
    logger.info(f"Processing {len(new_records)} new records for batch {batch_id}")
    
    # Process each new record
    for _, record in new_records.iterrows():
        try:
            # Get the image from MinIO
            image_data = get_image_from_minio(record['minio_path'])
            
            # Process the image
            results = process_image(image_data, parameters)
            
            # Create a new result record
            new_result = {
                'id': record['id'],
                'original_record_id': record['id'],
                'minio_path': record['minio_path'],
                'last_modified': datetime.now(),
                'batch_id': batch_id,
                'purpose': record['purpose'],
                'cell_count': len(results),
                'results': results
            }
            
            # Append the new result to the results DataFrame
            updated_results = pd.concat([updated_results, pd.DataFrame([new_result])], ignore_index=True)
            
        except Exception as e:
            logger.error(f"Error processing record with minio_path {record['minio_path']}: {str(e)}")
            continue
    
    return raw_db_data, updated_results 