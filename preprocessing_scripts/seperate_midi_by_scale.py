import os
import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
import time

# Set up logging
logging.basicConfig(
    filename="../outputs_2/separate_midi_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def process_file(file_entry, key_data):
    """Process a single MIDI file and move it to the appropriate scale directory."""
    try:
        file_path = file_entry['full_path']
        relative_path = file_entry['relative_path']
        file_name = os.path.basename(file_path)
        
        # Skip files already in scale directories
        parent_dir = os.path.basename(os.path.dirname(file_path))
        if parent_dir in ['major', 'minor']:
            return 'skipped (already in scale directory)'

        # Get scale info from JSON data
        if relative_path not in key_data:
            return 'skipped (no scale data in JSON)'
            
        scale = key_data[relative_path].get('scale')
        if not scale:
            return 'skipped (missing scale info)'
            
        if scale not in ['major', 'minor']:
            return f'skipped (invalid scale: {scale})'
        
        # Create output directory path
        output_dir = os.path.join(os.path.dirname(file_path), scale)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)
        
        # Move the file
        shutil.move(file_path, output_path)
        return f'moved to {scale}'
        
    except Exception as e:
        logging.error(f"Error processing {relative_path}: {str(e)}")
        return f'error: {str(e)}'

def main():
    input_base_dir = "../data/transposed_midi"
    json_path = "../outputs_2/key_scales.json"
    
    # Load key/scale data
    try:
        with open(json_path, "r") as f:
            key_data = json.load(f)
        print(f"Loaded key/scale data for {len(key_data)} files")
    except Exception as e:
        logging.error(f"Error loading {json_path}: {e}")
        print(f"Error loading {json_path}: {e}")
        return
    
    # Collect all MIDI files to process
    file_entries = []
    for root, _, files in os.walk(input_base_dir):
        for file in files:
            if file.lower().endswith('.mid'):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, input_base_dir)
                file_entries.append({
                    'full_path': full_path,
                    'relative_path': relative_path
                })
    
    print(f"Found {len(file_entries)} MIDI files to process")
    
    # Process files with progress tracking
    start_time = time.time()
    processed = 0
    stats = {
        'moved_major': 0,
        'moved_minor': 0,
        'skipped': 0,
        'errors': 0
    }
    
    # Use ThreadPoolExecutor for I/O bound operations
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 4)) as executor:
        futures = []
        for entry in file_entries:
            futures.append(executor.submit(process_file, entry, key_data))
        
        for i, future in enumerate(futures, 1):
            result = future.result()
            if 'major' in result:
                stats['moved_major'] += 1
            elif 'minor' in result:
                stats['moved_minor'] += 1
            elif 'skipped' in result:
                stats['skipped'] += 1
            else:
                stats['errors'] += 1
            
            # Print progress every 100 files
            if i % 100 == 0 or i == len(file_entries):
                elapsed = time.time() - start_time
                print(
                    f"Processed {i}/{len(file_entries)} | "
                    f"Major: {stats['moved_major']} | "
                    f"Minor: {stats['moved_minor']} | "
                    f"Skipped: {stats['skipped']} | "
                    f"Errors: {stats['errors']} | "
                    f"Elapsed: {elapsed:.1f}s"
                )
    
    # Final report
    elapsed = time.time() - start_time
    print("\n=== Separation Complete ===")
    print(f"Total files processed: {len(file_entries)}")
    print(f"Files moved to major: {stats['moved_major']}")
    print(f"Files moved to minor: {stats['moved_minor']}")
    print(f"Files skipped: {stats['skipped']}")
    print(f"Files with errors: {stats['errors']}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Files per second: {len(file_entries)/elapsed:.2f}")
    
    logging.info("File separation completed successfully")
    print("\nAll files have been processed and organized by scale.")

if __name__ == "__main__":
    main()