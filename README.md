# Solar Panel Dust Detection AI

AI-powered application for detecting dust accumulation on solar panels and providing maintenance recommendations.

## Features

- **Dust Detection**: Uses a Vision Transformer (ViT) model to classify solar panels as Clean or Dusty
- **AI Analysis**: Generates personalized maintenance recommendations using Llama 3.3 70B
- **Energy Impact Estimates**: Calculates potential efficiency loss and financial impact
- **Session History**: Tracks your analysis history during the session

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```
### 2. Add Groq API key
Open your terminal in the project root and use **Vim** to create the file:
```bash
vim .env
```
1. Press i to enter **Insert Mode**
2. Type the following: GROQ_API_KEY=your_actual_groq_api_key_here
3. Press *Esc* to exit **Insert Mode**
4. Type *:wq* to Write and Quit
=> You can check if the file was created correctly by running:
```bash
cat .env
```
### 3. Run the Application

```bash
python app.py
```

### 4. Open in Browser

Navigate to **http://localhost:7860**

## Usage

1. **Upload Image**: Drag and drop or click to upload a solar panel image
2. **Configure** (optional): Expand "System Settings" to customize:
   - Location
   - Panel wattage
   - Number of panels
   - Electricity price
3. **Analyze**: Click the "Analyze Panel" button
4. **Review Results**: View the detection result and AI analysis

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Gradio |
| Image Classification | Vision Transformer (ViT) |
| AI Analysis | Llama 3.3 70B via Groq |
| Deep Learning | PyTorch + HuggingFace Transformers |

## Model

The dust detection model is a fine-tuned Vision Transformer that classifies solar panel images:

- **Clean**: Panel surface is clear, operating at optimal efficiency
- **Dusty**: Panel has dust accumulation that may reduce energy production

## Project Structure

```
.
├── app.py                    # Main application
├── requirements.txt          # Python dependencies
├── config.json              # Model configuration
├── preprocessor_config.json # Image preprocessor config
├── train_model.ipynb        # The training of the model (not needed to run the program, final model was saved on previous files)
├── model.safetensors        # Model weights
├── SAMPLES                  # Samples of solar panels that were not used neither on the training nor test set
├── DEMO.mov                 # Demo of the usage
├── Prototype_Description.docx # Description of the project
└── README.md                # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## Design Principles

This application follows responsible AI practices:

- **Transparency**: Confidence scores shown for all predictions
- **Explainability**: AI provides reasoning for recommendations
- **Accessibility**: Free and open-source tools
- **Privacy**: Images processed locally, not stored
- **User Control**: Configurable parameters and clear feedback

## License

MIT License
