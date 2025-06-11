import os
import json
import logging
from music21 import converter, key, pitch
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Set up logging
logging.basicConfig(
    filename="../outputs_2/transpose_midi_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_transposition_interval(current_tonic, target_tonic):
    """Calculate the shortest transposition interval to the nearest target pitch."""
    current_midi = pitch.Pitch(current_tonic).midi
    target_midi = pitch.Pitch(target_tonic).midi
    interval_up = (target_midi - current_midi) % 12
    interval_down = (current_midi - target_midi) % 12
    return interval_up if interval_up <= interval_down else -interval_down

def transpose_midi(file_info, input_base_dir, output_base_dir, key_data):
    """Transpose a single MIDI file with proper error handling."""
    try:
        relative_path = file_info['relative_path']
        input_path = os.path.join(input_base_dir, relative_path)
        output_path = os.path.join(output_base_dir, relative_path)
        
        # Skip if output already exists
        if os.path.exists(output_path):
            return "skipped (exists)"
        
        # Get key info from JSON data
        if relative_path not in key_data:
            return "skipped (no key data)"
            
        info = key_data[relative_path]
        if "key" not in info or "scale" not in info:
            return "skipped (invalid key data)"
            
        scale = info["scale"]
        current_key = info["key"]
        
        if scale not in ["major", "minor"]:
            return "skipped (invalid scale)"
        
        # Handle key notation (e.g., "E-" -> "Eb")
        tonic = current_key.replace("-", "b")
        
        midi = converter.parse(input_path)
        current_key_obj = key.Key(tonic, scale)
        target_key = key.Key("C", "major") if scale == "major" else key.Key("A", "minor")

        semitones = get_transposition_interval(tonic, target_key.tonic.name)
        midi.transpose(semitones, inPlace=True)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        midi.write('midi', output_path)
        
        return f"transposed {semitones} semitones"
    except Exception as e:
        logging.error(f"Error transposing {relative_path}: {e}")
        return f"error: {str(e)}"

def main():
    input_base_dir = "../data/Cleaned_Classical_Midi_dataset"
    json_path = "../outputs_2/key_scales.json"
    output_base_dir = "../data/transposed_midi"
    
    # Load key/scale data
    try:
        with open(json_path, "r") as f:
            key_data = json.load(f)
    except Exception as e:
        logging.error(f"Error loading {json_path}: {e}")
        print(f"Error loading {json_path}: {e}")
        return
    
    # Collect all files to process
    file_list = []
    for root, _, files in os.walk(input_base_dir):
        for file in files:
            if file.endswith(".mid"):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, input_base_dir)
                file_list.append({
                    'full_path': full_path,
                    'relative_path': relative_path
                })
    
    # Process files in parallel
    start_time = time.time()
    stats = {
        'total': len(file_list),
        'transposed': 0,
        'skipped': 0,
        'errors': 0
    }
    
    print(f"Starting transposition of {stats['total']} files...")
    
    with Pool(cpu_count()) as pool:
        process_func = partial(
            transpose_midi,
            input_base_dir=input_base_dir,
            output_base_dir=output_base_dir,
            key_data=key_data
        )
        
        for i, result in enumerate(pool.imap_unordered(process_func, file_list), 1):
            if "transposed" in result:
                stats['transposed'] += 1
            elif "skipped" in result:
                stats['skipped'] += 1
            else:
                stats['errors'] += 1
            
            # Print progress every 100 files
            if i % 100 == 0 or i == stats['total']:
                elapsed = time.time() - start_time
                print(
                    f"Processed {i}/{stats['total']} | "
                    f"Transposed: {stats['transposed']} | "
                    f"Skipped: {stats['skipped']} | "
                    f"Errors: {stats['errors']} | "
                    f"Time: {elapsed:.1f}s"
                )
    
    # Final report
    elapsed = time.time() - start_time
    print("\nTransposition complete!")
    print(f"Total files processed: {stats['total']}")
    print(f"Successfully transposed: {stats['transposed']}")
    print(f"Skipped: {stats['skipped']} (exists/no key data)")
    print(f"Errors: {stats['errors']}")
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Files per second: {stats['total']/elapsed:.2f}")
    
    logging.info(f"Transposed files saved to {output_base_dir}")
    print(f"\nTransposed files saved to {output_base_dir}")

if __name__ == "__main__":
    main()