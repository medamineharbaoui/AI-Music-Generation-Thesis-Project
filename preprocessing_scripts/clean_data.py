import os
from music21 import converter, instrument
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Set up logging
logging.basicConfig(
    filename="../outputs/clean_data_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def clean_midi_file(input_path, output_path):
    """Clean a MIDI file by ensuring it has 2 piano tracks."""
    try:
        # Skip if output already exists
        if os.path.exists(output_path):
            return True

        midi = converter.parse(input_path)
        if len(midi.parts) != 2:
            logging.warning(f"Skipping {input_path}: Expected 2 tracks, found {len(midi.parts)}")
            return False
        
        # Ensure tracks are piano
        for part in midi.parts:
            piano = instrument.Piano()
            part.insert(0, piano)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        midi.write('midi', output_path)
        logging.info(f"Cleaned {input_path} -> {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error cleaning {input_path}: {e}")
        return False

def process_file(args, input_dir, output_dir):
    """Process a single file with proper error handling."""
    root, file = args
    if file.endswith(".mid"):
        input_path = os.path.join(root, file)
        relative_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        return clean_midi_file(input_path, output_path)
    return None

def clean_dataset(input_dir="../data/Classical_Music_Midi_dataset", 
                 output_dir="../data/Cleaned_Classical_Midi_dataset"):
    """Clean all MIDI files in the input directory using parallel processing."""
    start_time = time.time()
    files_processed = 0
    files_skipped = 0
    files_failed = 0

    # Collect all files to process
    file_list = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mid"):
                file_list.append((root, file))

    # Process files in parallel
    with Pool(cpu_count()) as pool:
        process_func = partial(process_file, input_dir=input_dir, output_dir=output_dir)
        results = pool.imap_unordered(process_func, file_list)
        
        for result in results:
            if result is True:
                files_processed += 1
            elif result is False:
                files_skipped += 1
            else:
                files_failed += 1

            # Progress reporting
            if (files_processed + files_skipped + files_failed) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed: {files_processed}, Skipped: {files_skipped}, Failed: {files_failed} | Time: {elapsed:.2f}s")

    # Final report
    elapsed = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Total files: {len(file_list)}")
    print(f"Processed: {files_processed}")
    print(f"Skipped: {files_skipped} (wrong track count)")
    print(f"Failed: {files_failed} (errors)")
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Files per second: {len(file_list)/elapsed:.2f}")

if __name__ == "__main__":
    clean_dataset()