# ApexVector

Python image vectorization: raster → optimized SVG via adaptive region classification. It converts raster images (JPG/PNG) into scalable vector graphics (SVG) with a focus on preserving geometric and perceptual quality using advanced curve fitting and aggressive compression.

## Installation

ApexVector requires Python 3.9+ and the following dependencies:

```bash
# Clone the repository
git clone https://github.com/Anshuldhakar1/ApexVector.git
cd ApexVector

# Install dependencies
pip install -r requirements.txt
```

## How to Run

The program is run via the command line interface using the `vectorizer` module.

### Basic Usage

To vectorize an input image with default settings:

```bash
python -m vectorizer input.png -o output.svg
```

### Advanced Modes (Compression and Speed)

Use flags to adjust performance or quality trade-offs.

| Flag | Mode | Description |
|---|---|---|
| `--speed` | Fast | Prioritizes segmentation and faster vector fitting. |
| `--quality` | Quality | Prioritizes more complex segmentation and higher precision curve fitting. |
| `--optimize <MODE>` | Compression | Applies various SVG compression levels (e.g., `ultra`, `merged`, `insane`). |

**Example:** Run in quality mode and apply aggressive merging for small output size.
```bash
python -m vectorizer input.png -o output.svg --quality --optimize merged
```

### Testing

Run the unit tests to verify installation:

```bash
pytest -x                    # Run all tests (stops on first failure)
pytest -v -x path/to/test.py # Verbose for a single file
```

## Development Branch: Curvature-Preserving

The `feature/curvature-preserving` branch is under active development to implement Clothoid-based G² curve synthesis for superior curvature preservation.

To switch to the development branch:

```bash
git checkout feature/curvature-preserving
```
