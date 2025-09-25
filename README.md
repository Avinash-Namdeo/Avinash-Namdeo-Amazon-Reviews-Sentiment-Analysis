# Amazon Reviews Sentiment Analysis

A machine learning project that analyzes sentiment in Amazon product reviews using LSTM neural networks with an interactive Streamlit web interface.

## Features

- **LSTM-based Sentiment Analysis**: Deep learning model trained on Amazon reviews
- **Interactive Web Interface**: Clean, modern Streamlit UI for real-time sentiment analysis
- **Real-time Analysis**: Instant sentiment prediction with confidence scores
- **Sample Reviews**: Pre-loaded examples for quick testing
- **Analysis Statistics**: Comprehensive metrics and insights

## Setup Instructions

### Prerequisites

- Python 3.7+
- Git with Git LFS support

### Installation

1. **Clone the repository**
   \`\`\`bash
   git clone https://github.com/Avinash-Namdeo/Amazon-Reviews-Sentiment-Analysis.git
   cd Amazon-Reviews-Sentiment-Analysis
   \`\`\`

2. **Install Git LFS** (required for tokenizer.pkl)
   \`\`\`bash
   git lfs install
   git lfs pull
   \`\`\`

3. **Create virtual environment**
   \`\`\`bash
   python -m venv myvenv
   # On Windows:
   myvenv\Scripts\activate
   # On macOS/Linux:
   source myvenv/bin/activate
   \`\`\`

4. **Install dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

## Usage

### Running the Streamlit Web Interface

\`\`\`bash
streamlit run scripts/enhanced_sentiment_ui.py
\`\`\`

The web interface will open in your browser at `http://localhost:8501`

### Features of the Web Interface

- **Text Input**: Enter any review text for sentiment analysis
- **Sample Reviews**: Click buttons to test with pre-loaded positive and negative examples
- **Real-time Results**: Get instant sentiment predictions with confidence scores
- **Analysis Statistics**: View comprehensive metrics about the analysis

### Running the Main Application

\`\`\`bash
python main.py
\`\`\`

## Model Details

- **Architecture**: LSTM (Long Short-Term Memory) neural network
- **Training Data**: Amazon product reviews dataset
- **Preprocessing**: Text cleaning, tokenization, and sequence padding
- **Output**: Sentiment classification (Positive/Negative) with confidence scores

## Dependencies

Key libraries used in this project:

- `streamlit` - Web interface framework
- `tensorflow` - Deep learning model
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scikit-learn` - Machine learning utilities
- `matplotlib` - Data visualization

See `requirements.txt` for complete list of dependencies.

## Important Notes

- **Large File Storage**: The `tokenizer.pkl` file (101MB) is stored using Git LFS
- **Virtual Environment**: Always use a virtual environment to avoid dependency conflicts
- **Memory Requirements**: Model loading requires sufficient RAM (recommended: 4GB+)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

**Avinash Namdeo**
- GitHub: [@Avinash-Namdeo](https://github.com/Avinash-Namdeo)

## Acknowledgments

- Amazon Reviews Dataset
- TensorFlow and Keras communities
- Streamlit framework developers
