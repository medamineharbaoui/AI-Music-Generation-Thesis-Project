import os
import json
import logging
from music21 import converter, analysis
from multiprocessing import Pool, cpu_count
from functools import partial

# Set up logging
logging.basicConfig(filename="../outputs_2/analyze_midi_log.txt", 
                    level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

def analyze_midi_key(file_path):
    """Analyze the key and scale of a MIDI file."""
    try:
        midi = converter.parse(file_path)
        ks = analysis.discrete.KrumhanslSchmuckler()
        key = ks.getSolution(midi)
        key_str = key.tonic.name
        scale = key.mode
        return {"key": key_str, "scale": scale}
    except Exception as e:
        logging.error(f"Error analyzing {file_path}: {e}")
        print(f"Error analyzing {file_path}: {e}")
        return None

def process_file(input_dir, file_info):
    """Helper function for parallel processing."""
    subdir, file = file_info
    if file.endswith(".mid"):
        file_path = os.path.join(subdir, file)
        print(f"Analyzing {file_path}")
        result = analyze_midi_key(file_path)
        if result:
            relative_path = os.path.relpath(file_path, input_dir)
            return (relative_path, result)
    return None

def analyze_all_midi_files(input_dir, output_path):
    """Analyze all MIDI files in a directory and its subdirectories."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = {}

    # Collect all files to process
    files_to_process = []
    for subdir, _, files in os.walk(input_dir):
        for file in files:
            files_to_process.append((subdir, file))

    # Process files in parallel
    with Pool(cpu_count()) as pool:
        process_func = partial(process_file, input_dir)
        for result in pool.imap_unordered(process_func, files_to_process):
            if result:
                relative_path, analysis_result = result
                results[relative_path] = analysis_result
                logging.info(f"Analyzed {relative_path}: {analysis_result}")
                print(f"  Result: {analysis_result}")

    # Save results to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    logging.info(f"Results saved to {output_path}")
    print(f"Results saved to {output_path}")

def main():
    input_dir = "../data/Cleaned_Classical_Midi_dataset"
    output_path = "../outputs_2/key_scales.json"
    analyze_all_midi_files(input_dir, output_path)

if __name__ == "__main__":
    main()