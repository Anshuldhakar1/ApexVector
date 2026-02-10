#!/usr/bin/env python3
"""
Batch image processor for ApexVector.

Configuration is set directly in the main() function below.
Run with: python batch_process.py
"""

import subprocess
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Set
import time


def get_image_files(folder: Path, extensions: Optional[Set[str]] = None) -> List[Path]:
    """Get all image files from a folder."""
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")

    images = []
    for ext in extensions:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))

    return sorted(list(set(images)))


def process_image(
    input_path: Path,
    output_folder: Path,
    command_template: str,
    output_ext: str = ".svg",
    create_subfolder: bool = False
) -> Tuple[Path, Path, bool, str]:
    """
    Process a single image using the command template.

    Args:
        input_path: Path to input image
        output_folder: Base output folder
        command_template: Command template string
        output_ext: Output file extension
        create_subfolder: If True, creates a subfolder for each image

    Returns:
        Tuple of (input_path, output_path, success, message)
    """
    # Construct output path
    if create_subfolder:
        # Create a subfolder named after the image (without extension)
        image_folder = output_folder / input_path.stem
        image_folder.mkdir(parents=True, exist_ok=True)
        output_path = image_folder / f"{input_path.stem}{output_ext}"
    else:
        output_path = output_folder / f"{input_path.stem}{output_ext}"

    # Build command from template
    command = command_template.format(
        input=str(input_path),
        output=str(output_path),
        name=input_path.stem,
        ext=input_path.suffix
    )

    try:
        # Run command
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per image
        )

        if result.returncode == 0:
            return input_path, output_path, True, "Success"
        else:
            error_msg = result.stderr.strip() or "Unknown error"
            return input_path, output_path, False, f"Error: {error_msg[:200]}"

    except subprocess.TimeoutExpired:
        return input_path, output_path, False, "Timeout (5min)"
    except Exception as e:
        return input_path, output_path, False, f"Exception: {str(e)[:200]}"


def process_image_with_subfolder(
    input_path_str: str,
    output_folder_str: str,
    command_template: str,
    output_ext: str
) -> Tuple[str, str, bool, str]:
    """
    Process a single image with subfolder creation for save-stages.
    This is a module-level function to work with multiprocessing.

    Args:
        input_path_str: String path to input image
        output_folder_str: String path to base output folder
        command_template: Command template with {input}, {output}, {output_dir}, {name}, {ext}
        output_ext: Output file extension

    Returns:
        Tuple of (input_path_str, output_path_str, success, message)
    """
    input_path = Path(input_path_str)
    output_folder = Path(output_folder_str)

    # Create subfolder
    image_folder = output_folder / input_path.stem
    image_folder.mkdir(parents=True, exist_ok=True)
    output_path = image_folder / f"{input_path.stem}{output_ext}"

    # Replace placeholders in command template
    cmd = command_template.format(
        input=str(input_path),
        output=str(output_path),
        output_dir=str(image_folder),
        name=input_path.stem,
        ext=input_path.suffix
    )

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            return str(input_path), str(output_path), True, "Success"
        else:
            error_msg = result.stderr.strip() or "Unknown error"
            return str(input_path), str(output_path), False, f"Error: {error_msg[:200]}"

    except subprocess.TimeoutExpired:
        return str(input_path), str(output_path), False, "Timeout (5min)"
    except Exception as e:
        return str(input_path), str(output_path), False, f"Exception: {str(e)[:200]}"


def batch_convert(
    input_folder: str,
    output_folder: str,
    command_template: str,
    output_ext: str = ".svg",
    max_workers: int = 10,
    extensions: Optional[Set[str]] = None,
    create_subfolders: bool = False
) -> dict:
    """
    Batch convert all images in a folder.

    Args:
        input_folder: Path to folder containing images
        output_folder: Path to folder for output files
        command_template: Command template with {input}, {output}, {name}, {ext} placeholders
        output_ext: Output file extension
        max_workers: Maximum parallel processes (capped at 10)
        extensions: Set of file extensions to process
        create_subfolders: If True, creates a subfolder for each image output

    Returns:
        Dictionary with results summary
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Validate inputs
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    # Create output folder
    output_path.mkdir(parents=True, exist_ok=True)

    # Get image files
    images = get_image_files(input_path, extensions)
    if not images:
        raise ValueError(f"No images found in: {input_folder}")

    print(f"Found {len(images)} images to process")
    print(f"Output folder: {output_path.absolute()}")
    print(f"Subfolders: {'Yes' if create_subfolders else 'No'}")
    print(f"Max workers: {min(len(images), max_workers)}")
    print(f"Command: {command_template}")
    print("-" * 60)

    # Calculate actual workers (cap at 10)
    actual_workers = min(len(images), max_workers, 10)

    results = {
        'total': len(images),
        'success': 0,
        'failed': 0,
        'results': []
    }

    start_time = time.time()

    # Process images in parallel
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        # Submit all tasks
        future_to_image = {
            executor.submit(
                process_image,
                img,
                output_path,
                command_template,
                output_ext,
                create_subfolders
            ): img for img in images
        }

        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_image)):
            input_file, output_file, success, message = future.result()

            results['results'].append({
                'input': str(input_file),
                'output': str(output_file),
                'success': success,
                'message': message
            })

            if success:
                results['success'] += 1
                status = "[OK]"
            else:
                results['failed'] += 1
                status = "[FAIL]"

            progress = (i + 1) / len(images) * 100
            print(f"[{i+1}/{len(images)}] {status} {Path(input_file).name}: {message}")

    elapsed = time.time() - start_time

    # Print summary
    print("-" * 60)
    print(f"Completed in {elapsed:.2f}s")
    print(f"Total: {results['total']}")
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")

    return results


def batch_convert_with_subfolders(
    input_folder: str,
    output_folder: str,
    command_template: str,
    output_ext: str = ".svg",
    max_workers: int = 10,
    extensions: Optional[Set[str]] = None
) -> dict:
    """
    Batch convert with subfolder creation for save-stages.
    Each image gets its own folder with the SVG and stage outputs inside.

    Args:
        input_folder: Path to folder containing images
        output_folder: Path to folder for output files
        command_template: Command template with {input}, {output}, {output_dir}, {name}, {ext}
        output_ext: Output file extension
        max_workers: Maximum parallel processes (capped at 10)
        extensions: Set of file extensions to process

    Returns:
        Dictionary with results summary
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Validate inputs
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    # Create output folder
    output_path.mkdir(parents=True, exist_ok=True)

    # Get image files
    images = get_image_files(input_path, extensions)
    if not images:
        raise ValueError(f"No images found in: {input_folder}")

    print(f"Found {len(images)} images to process")
    print(f"Output folder: {output_path.absolute()}")
    print(f"Subfolders: Yes (with save-stages)")
    print(f"Max workers: {min(len(images), max_workers, 10)}")
    print(f"Command: {command_template}")
    print("-" * 60)

    # Calculate actual workers (cap at 10)
    actual_workers = min(len(images), max_workers, 10)

    results = {
        'total': len(images),
        'success': 0,
        'failed': 0,
        'results': []
    }

    start_time = time.time()

    # Process images in parallel using the module-level function
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        # Submit all tasks
        future_to_image = {
            executor.submit(
                process_image_with_subfolder,
                str(img),
                str(output_path),
                command_template,
                output_ext
            ): img for img in images
        }

        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_image)):
            input_file_str, output_file_str, success, message = future.result()

            results['results'].append({
                'input': input_file_str,
                'output': output_file_str,
                'success': success,
                'message': message
            })

            if success:
                results['success'] += 1
                status = "[OK]"
            else:
                results['failed'] += 1
                status = "[FAIL]"

            progress = (i + 1) / len(images) * 100
            print(f"[{i+1}/{len(images)}] {status} {Path(input_file_str).name}: {message}")

    elapsed = time.time() - start_time

    # Print summary
    print("-" * 60)
    print(f"Completed in {elapsed:.2f}s")
    print(f"Total: {results['total']}")
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")

    return results


def main():
    """
    Configure your batch processing job here.
    
    Command Template Variables:
        {input}   - Full path to input image file
        {output}  - Full path to output file  
        {name}    - Base filename (without extension)
        {ext}     - File extension (e.g., .jpg)
        {output_dir} - Output folder path (for save-stages with subfolders)
    """
    
    # ============================================
    # CONFIGURATION - Edit these values
    # ============================================
    
    # Input folder containing images
    INPUT_FOLDER = "./test_images"
    
    # Output folder for converted files
    OUTPUT_FOLDER = "./batch_output"
    
    # Command template to run on each image
    # Example: Poster mode with 12 colors and save stages
    COMMAND_TEMPLATE = "python -m apexvec {input} --poster --colors 24 --save-stages {output_dir} -o {output}"
    
    # Alternative commands (uncomment the one you want):
    # COMMAND_TEMPLATE = "python -m apexvec {input} --poster --colors 8 -o {output}"
    # COMMAND_TEMPLATE = "python -m apexvec {input} --contrast --save-stages {output_dir} -o {output}"
    # COMMAND_TEMPLATE = "python -m apexvec {input} --quality -o {output}"
    # COMMAND_TEMPLATE = "python -m apexvec {input} --speed -o {output}"
    
    # Output file extension
    OUTPUT_EXT = ".svg"
    
    # Maximum parallel processes (capped at 10)
    MAX_WORKERS = 10
    
    # Image extensions to process
    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    # CREATE SUBFOLDERS FOR EACH IMAGE
    # Set to True when using --save-stages to organize outputs:
    # batch_output/
    #   img1/
    #     img1.svg
    #     01_ingest.png
    #     02_quantization.png
    #     ...
    #   img2/
    #     img2.svg
    #     ...
    CREATE_SUBFOLDERS = True
    
    # ============================================
    # END CONFIGURATION
    # ============================================
    
    # Check if command template uses {output_dir} and CREATE_SUBFOLDERS is True
    if CREATE_SUBFOLDERS and "{output_dir}" in COMMAND_TEMPLATE:
        # Use the special subfolder-aware processing
        try:
            results = batch_convert_with_subfolders(
                input_folder=INPUT_FOLDER,
                output_folder=OUTPUT_FOLDER,
                command_template=COMMAND_TEMPLATE,
                output_ext=OUTPUT_EXT,
                max_workers=MAX_WORKERS,
                extensions=EXTENSIONS
            )
            
            return 0 if results['failed'] == 0 else 1
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        except Exception as e:
            print(f"Unexpected error: {e}")
            return 1
    else:
        # Standard processing (without subfolders or without save-stages)
        try:
            results = batch_convert(
                input_folder=INPUT_FOLDER,
                output_folder=OUTPUT_FOLDER,
                command_template=COMMAND_TEMPLATE,
                output_ext=OUTPUT_EXT,
                max_workers=MAX_WORKERS,
                extensions=EXTENSIONS,
                create_subfolders=CREATE_SUBFOLDERS
            )
            
            return 0 if results['failed'] == 0 else 1
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        except Exception as e:
            print(f"Unexpected error: {e}")
            return 1


if __name__ == "__main__":
    exit(main())
