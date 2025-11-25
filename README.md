# Game Camera Animal Recognition Application

A desktop application that uses computer vision and deep learning to automatically detect, identify, classify, and catalog wildlife from game camera footage.

## Features

- 🦌 **Automated Animal Detection**: YOLOv8-powered object detection
- 🔍 **Species Classification**: Identify 100+ species with confidence scoring
- 📸 **Smart Cropping**: Intelligent cropping around detected animals
- 📊 **CSV Export**: Comprehensive metadata export for analysis
- 🎨 **Modern GUI**: PySide6-based user interface
- ⚡ **Batch Processing**: Handle multiple images simultaneously
- 📁 **Semantic Organization**: Organized output by species, date, and camera

## Requirements

- Python 3.10-3.12 (3.13 not yet fully supported by PyTorch)
- macOS 13+, Windows 10+, or Ubuntu 20.04+
- 8GB RAM minimum (16GB recommended)
- Optional: NVIDIA GPU with CUDA support for faster processing

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/RandyNorthrup/game-camera-analyzer.git
cd game-camera-analyzer
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment with Python 3.12
python3.12 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
game-camera-analyzer/
├── .github/
│   └── copilot-instructions.md  # Code quality standards
├── gui/                          # PySide6 GUI components
├── core/                         # Business logic
├── models/                       # ML model wrappers
├── data/                         # Data management
├── utils/                        # Utilities
├── resources/                    # Icons and themes
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
├── PROJECT_PLAN.md               # Complete technical specification
└── README.md                     # This file
```

## Usage

### Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run the application (once implemented)
python main.py
```

### Workflow

1. **Load Images**: Select folder or individual game camera images
2. **Configure Settings**: Adjust detection thresholds and output preferences
3. **Process**: Click "Start Processing" to analyze images
4. **Review Results**: View detected animals with confidence scores
5. **Export**: Save results and metadata to CSV

## Output Structure

```
output/
├── by_species/
│   ├── white_tailed_deer/
│   │   ├── 2025-11-24/
│   │   │   ├── deer_001_20251124_143022.jpg
│   │   │   └── ...
│   │   └── thumbnails/
│   └── ...
├── by_date/
│   └── 2025/11/24/
├── by_camera/
│   └── camera_01/
├── detections_master.csv         # All detection metadata
├── species_summary.csv           # Aggregate statistics
└── logs/
    └── processing_20251124.log
```

## Development

### Code Quality Standards

This project follows strict code quality standards outlined in `.github/copilot-instructions.md`:

- ✅ Fully implemented code only (no placeholders)
- ✅ Comprehensive error handling
- ✅ Structured logging
- ✅ Type hints and docstrings
- ✅ Unit tests

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
# Format code
black .

# Check style
flake8

# Type checking
mypy .
```

## Technology Stack

- **GUI**: PySide6 (Qt for Python)
- **Detection**: Ultralytics YOLOv8
- **Classification**: timm (PyTorch Image Models)
- **Image Processing**: OpenCV, Pillow
- **Data Management**: Pandas, NumPy
- **Deep Learning**: PyTorch

## Roadmap

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the complete technical specification and development roadmap.

### Phase 1: Foundation ✅
- [x] Project setup
- [x] Documentation
- [x] Dependencies

### Phase 2: Core Detection (In Progress)
- [ ] Model integration
- [ ] Detection engine
- [ ] Image preprocessing

### Phase 3-8: See PROJECT_PLAN.md

## Contributing

1. Review `.github/copilot-instructions.md` for coding standards
2. Create a feature branch
3. Write tests for new functionality
4. Submit a pull request

## License

[License information to be added]

## Acknowledgments

- Ultralytics YOLOv8 for object detection
- PyTorch and timm for deep learning infrastructure
- Qt/PySide6 for the GUI framework

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Project Status**: 🚧 In Development

Built with ❤️ for wildlife researchers and enthusiasts
