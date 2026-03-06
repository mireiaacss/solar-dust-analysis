import gradio as gr
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from datetime import datetime
from groq import Groq
import os
from dotenv import load_dotenv

# Load model and processor
MODEL_PATH = "."
model = ViTForImageClassification.from_pretrained(MODEL_PATH)
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
model.eval()

# Session data
analysis_history = []

# API client
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# Compact Frutiger Aero CSS
FRUTIGER_AERO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@300;400;600;700&display=swap');

* {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

.gradio-container {
    background: linear-gradient(145deg, #0d4f8c 0%, #1a7f64 25%, #2ecc71 50%, #87ceeb 75%, #a8d8ea 100%) !important;
    background-attachment: fixed !important;
    min-height: 100vh;
    padding: 10px !important;
}

.gradio-container::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(ellipse at 10% 20%, rgba(255,255,255,0.4) 0%, transparent 40%),
        radial-gradient(ellipse at 90% 80%, rgba(80,230,255,0.35) 0%, transparent 45%),
        radial-gradient(ellipse at 50% 50%, rgba(255,255,255,0.15) 0%, transparent 55%);
    pointer-events: none;
    z-index: 0;
}

.block, .form, .panel {
    background: linear-gradient(160deg, rgba(255,255,255,0.95) 0%, rgba(245,252,255,0.9) 50%, rgba(240,250,255,0.85) 100%) !important;
    backdrop-filter: blur(25px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(25px) saturate(180%) !important;
    border: 1px solid rgba(255,255,255,0.7) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0,120,212,0.12), inset 0 1px 0 rgba(255,255,255,0.9) !important;
    margin: 4px !important;
    padding: 12px !important;
}

h1, .markdown h1 {
    background: linear-gradient(135deg, #0078d4 0%, #00b294 50%, #2ecc71 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-weight: 700 !important;
    font-size: 1.8em !important;
    margin: 0 0 5px 0 !important;
    text-align: center;
}

h2, .markdown h2 {
    color: #0078d4 !important;
    font-weight: 600 !important;
    font-size: 1.1em !important;
    margin: 8px 0 !important;
    padding-bottom: 4px;
    border-bottom: 2px solid rgba(0,178,148,0.3);
}

h3, .markdown h3 {
    color: #00796b !important;
    font-weight: 600 !important;
    font-size: 1em !important;
    margin: 6px 0 !important;
}

.subtitle {
    text-align: center;
    color: #0078d4;
    font-size: 0.95em;
    margin-bottom: 10px;
}

/* Buttons */
.primary, button.primary {
    background: linear-gradient(145deg, #00c9a7 0%, #00b294 30%, #0078d4 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0,178,148,0.35), inset 0 1px 0 rgba(255,255,255,0.3) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    width: 100%;
}

.primary:hover, button.primary:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 6px 20px rgba(0,178,148,0.45) !important;
}

.secondary, button.secondary {
    background: linear-gradient(160deg, rgba(255,255,255,0.95) 0%, rgba(240,250,255,0.9) 100%) !important;
    border: 2px solid rgba(0,178,148,0.3) !important;
    border-radius: 10px !important;
    color: #0078d4 !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
}

/* Form inputs */
input, textarea, select {
    background: rgba(255,255,255,0.95) !important;
    border: 1px solid rgba(0,120,212,0.2) !important;
    border-radius: 8px !important;
    padding: 8px !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #00b294 !important;
    box-shadow: 0 0 0 3px rgba(0,178,148,0.15) !important;
}

input[type="range"] {
    accent-color: #00b294 !important;
}

/* Tabs */
.tab-nav {
    background: rgba(255,255,255,0.6) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    margin-bottom: 8px !important;
}

.tab-nav button {
    border-radius: 10px !important;
    padding: 8px 16px !important;
    font-weight: 500 !important;
    font-size: 0.9em !important;
    transition: all 0.3s ease !important;
}

.tab-nav button.selected {
    background: linear-gradient(145deg, #0078d4 0%, #00b294 100%) !important;
    color: white !important;
    box-shadow: 0 3px 12px rgba(0,120,212,0.3) !important;
}

/* Tables */
table {
    background: rgba(255,255,255,0.9) !important;
    border-radius: 10px !important;
    font-size: 0.9em !important;
}

th {
    background: linear-gradient(135deg, rgba(0,120,212,0.1) 0%, rgba(0,178,148,0.1) 100%) !important;
    color: #0078d4 !important;
    font-weight: 600 !important;
    padding: 8px 12px !important;
    font-size: 0.85em;
}

td {
    padding: 6px 12px !important;
}

/* Status badges */
.status-clean {
    background: linear-gradient(145deg, #57e389 0%, #00b294 100%);
    color: white;
    padding: 10px 20px;
    border-radius: 20px;
    display: inline-block;
    font-weight: 700;
    font-size: 1.1em;
    box-shadow: 0 4px 15px rgba(0,178,148,0.35);
    text-transform: uppercase;
    letter-spacing: 1px;
    animation: pulse-green 2s infinite;
}

.status-dusty {
    background: linear-gradient(145deg, #ffc107 0%, #e66100 100%);
    color: white;
    padding: 10px 20px;
    border-radius: 20px;
    display: inline-block;
    font-weight: 700;
    font-size: 1.1em;
    box-shadow: 0 4px 15px rgba(230,97,0,0.35);
    text-transform: uppercase;
    letter-spacing: 1px;
    animation: pulse-orange 2s infinite;
}

@keyframes pulse-green {
    0%, 100% { box-shadow: 0 4px 15px rgba(0,178,148,0.35); }
    50% { box-shadow: 0 4px 25px rgba(0,178,148,0.5); }
}

@keyframes pulse-orange {
    0%, 100% { box-shadow: 0 4px 15px rgba(230,97,0,0.35); }
    50% { box-shadow: 0 4px 25px rgba(230,97,0,0.5); }
}

.result-card {
    text-align: center;
    padding: 15px;
}

.confidence-bar {
    height: 8px;
    background: rgba(0,0,0,0.1);
    border-radius: 4px;
    overflow: hidden;
    margin: 8px 0;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #00b294, #0078d4);
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* Uncertainty box */
.uncertainty-box {
    background: linear-gradient(145deg, rgba(255,193,7,0.1) 0%, rgba(255,152,0,0.1) 100%);
    border: 1px solid rgba(255,152,0,0.3);
    border-left: 4px solid #ff9800;
    border-radius: 8px;
    padding: 12px;
    margin: 10px 0;
    font-size: 0.9em;
}

.info-callout {
    background: linear-gradient(145deg, rgba(0,120,212,0.08) 0%, rgba(0,178,148,0.08) 100%);
    border: 1px solid rgba(0,178,148,0.2);
    border-left: 4px solid #00b294;
    border-radius: 8px;
    padding: 12px;
    margin: 10px 0;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #0078d4 0%, #00b294 100%);
    border-radius: 3px;
}

.accordion { border-radius: 12px !important; }
"""


def get_confidence_interpretation(confidence: float) -> str:
    """Interpret confidence level with uncertainty."""
    if confidence >= 0.95:
        return "Very High", "The model is highly confident, but no AI system is perfect. Visual inspection is still recommended."
    elif confidence >= 0.85:
        return "High", "The model shows good confidence. Results are likely accurate but should be verified for critical decisions."
    elif confidence >= 0.70:
        return "Moderate", "The model has reasonable confidence. Consider taking additional photos from different angles for verification."
    else:
        return "Low", "The model is uncertain. This could be due to image quality, unusual panel appearance, or edge cases. Manual inspection recommended."


def generate_tabbed_analysis(
    detection_result: str,
    confidence: float,
    location: str,
    panel_watts: float,
    num_panels: int,
    electricity_price: float,
    currency: str,
) -> tuple:
    """Generate comprehensive tabbed analysis using Groq."""

    conf_level, conf_note = get_confidence_interpretation(confidence)
    total_capacity = panel_watts * num_panels / 1000

    # Summary prompt
    summary_prompt = f"""You are a solar panel expert. Provide a brief executive summary (3-4 sentences).

Detection: {detection_result} (Confidence: {confidence:.1%} - {conf_level})
System: {total_capacity:.1f} kW ({num_panels} panels × {panel_watts}W)
Location: {location if location else "Not specified"}

Include:
1. What the detection means
2. Immediate implication
3. One key recommendation

IMPORTANT: Always acknowledge uncertainty. Use phrases like "Based on the analysis...", "The model suggests...", "This indicates..." rather than absolute statements."""

    # Detailed analysis prompt
    detailed_prompt = f"""You are a solar panel maintenance expert. Provide a detailed technical analysis.

Detection: {detection_result}
Confidence: {confidence:.1%} ({conf_level})
System: {total_capacity:.1f} kW total
Rate: {electricity_price} {currency}/kWh
Location: {location if location else "Not specified"}

Provide detailed analysis covering:

## Technical Assessment
- What the "{detection_result}" classification means technically
- How confidence level ({confidence:.1%}) should be interpreted
- Potential factors that could affect accuracy

## Energy Impact Analysis
- Estimated efficiency impact (dusty panels typically lose 5-25%)
- Daily/monthly energy loss estimates in kWh
- Financial impact calculation using {electricity_price} {currency}/kWh
- Note: These are estimates based on typical conditions

## Root Cause Considerations
- Common causes of dust accumulation
- Environmental factors to consider
- Seasonal patterns

IMPORTANT: Throughout your analysis, acknowledge uncertainty and limitations. Use hedging language like "approximately", "estimated", "typically", "may", "could". Remind that these are AI-generated estimates."""

    # Recommendations prompt
    recommendations_prompt = f"""You are a solar panel maintenance advisor. Provide actionable recommendations.

Detection: {detection_result} ({confidence:.1%} confidence)
System: {total_capacity:.1f} kW
Location: {location if location else "Not specified"}

Provide recommendations in these categories:

## Immediate Actions
- What should be done now (2-3 bullet points)
- Priority level for each action

## Maintenance Schedule
- Recommended cleaning frequency
- Best times/conditions for maintenance
- DIY vs professional cleaning guidance

## Monitoring Suggestions
- What to watch for going forward
- When to re-analyze
- Signs that indicate urgent attention needed

## Safety Reminders
- Important safety considerations
- When to consult professionals

Keep recommendations practical and actionable. Acknowledge that optimal approach may vary based on local conditions."""

    # Environmental impact prompt
    environmental_prompt = f"""You are a sustainability expert. Analyze the environmental impact.

System: {total_capacity:.1f} kW solar installation
Detection: {detection_result}
Confidence: {confidence:.1%}

Provide environmental analysis:

## Carbon Footprint Impact
- Estimate CO2 offset when panels operate optimally
- Impact of reduced efficiency due to dust
- Comparison to everyday activities (e.g., car miles, trees planted)

## Sustainability Benefits
- Environmental value of maintaining clean panels
- Long-term impact of regular maintenance
- Contribution to renewable energy goals

## Eco-Friendly Maintenance Tips
- Environmentally conscious cleaning methods
- Water conservation considerations
- Avoiding harmful chemicals

Note: Calculations are estimates based on average grid carbon intensity. Actual values vary by location."""

    try:
        # Generate all analyses
        responses = {}

        for name, prompt in [
            ("summary", summary_prompt),
            ("detailed", detailed_prompt),
            ("recommendations", recommendations_prompt),
            ("environmental", environmental_prompt)
        ]:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful expert. Be thorough but concise. Always acknowledge uncertainty and limitations of AI analysis. Use markdown formatting."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            responses[name] = response.choices[0].message.content

        # Add uncertainty disclaimer to summary
        uncertainty_notice = f"""
<div class="uncertainty-box">
⚠️ <strong>Uncertainty Notice</strong>: This analysis is generated by AI with {confidence:.1%} confidence ({conf_level}). {conf_note} All estimates are approximations and should not replace professional assessment for critical decisions.
</div>
"""

        summary_with_notice = uncertainty_notice + "\n\n" + responses["summary"]

        return (
            summary_with_notice,
            responses["detailed"],
            responses["recommendations"],
            responses["environmental"]
        )

    except Exception as e:
        error_msg = f"Analysis error: {str(e)}"
        return error_msg, error_msg, error_msg, error_msg


def analyze_panel(image, location, panel_watts, num_panels, electricity_price, currency):
    """Main analysis function."""

    if image is None:
        placeholder = "*Upload an image to analyze*"
        return (
            "<div style='text-align:center;padding:40px;color:#666;'>Upload an image to analyze</div>",
            "",
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            get_history_html(),
        )

    # Process image
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    label = model.config.id2label[predicted_class]
    conf_level, conf_note = get_confidence_interpretation(confidence)

    # Store history
    analysis_history.append({
        "time": datetime.now().strftime("%H:%M"),
        "result": label,
        "confidence": confidence,
    })

    # Status HTML
    status_class = "status-clean" if label == "Clean" else "status-dusty"
    status_html = f"""
<div class="result-card">
    <div class="{status_class}">{label.upper()}</div>
    <div style="margin-top:12px;">
        <div style="font-size:0.9em;color:#666;">Confidence: {conf_level}</div>
        <div style="font-size:1.4em;font-weight:600;color:#0078d4;">{confidence:.1%}</div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width:{confidence*100}%;"></div>
        </div>
        <div style="font-size:0.75em;color:#888;margin-top:4px;padding:0 10px;">
            {conf_note}
        </div>
    </div>
    <div style="font-size:0.8em;color:#888;margin-top:8px;">
        {datetime.now().strftime("%B %d, %Y at %H:%M")}
    </div>
</div>
"""

    # System summary
    system_html = f"""
| Config | Value |
|--------|-------|
| Panels | {num_panels} × {panel_watts}W |
| Capacity | {panel_watts * num_panels / 1000:.1f} kW |
| Rate | {electricity_price} {currency}/kWh |
| Location | {location if location else "—"} |
"""

    # Generate tabbed analysis
    summary, detailed, recommendations, environmental = generate_tabbed_analysis(
        label, confidence, location, panel_watts, num_panels, electricity_price, currency
    )

    return status_html, system_html, summary, detailed, recommendations, environmental, get_history_html()


def get_history_html():
    """Generate history display."""
    if not analysis_history:
        return "*No analyses yet*"

    md = "| Time | Result | Confidence |\n|------|--------|------------|\n"
    for entry in reversed(analysis_history[-5:]):
        md += f"| {entry['time']} | {entry['result']} | {entry['confidence']:.0%} |\n"
    return md


def clear_history():
    """Clear analysis history."""
    analysis_history.clear()
    return get_history_html()


def create_interface():
    """Create Gradio interface with tabbed analysis."""

    with gr.Blocks(title="Solar Panel AI") as demo:

        # Header
        gr.Markdown("# Solar Panel Dust Detection")
        gr.Markdown("<p class='subtitle'>AI-powered analysis for optimal panel performance</p>")

        # Main layout
        with gr.Row():
            # LEFT: Input
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Panel Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=200,
                )

                with gr.Accordion("System Settings", open=False):
                    location_input = gr.Textbox(
                        label="Location",
                        placeholder="e.g., Barcelona",
                        value="",
                    )
                    with gr.Row():
                        panel_watts = gr.Slider(100, 600, 400, step=25, label="Watts/Panel")
                        num_panels = gr.Slider(1, 50, 10, step=1, label="Panels")
                    with gr.Row():
                        electricity_price = gr.Number(label="Price/kWh", value=0.15, precision=2, step=0.01, minimum=0)
                        currency = gr.Radio(
                            choices=["EUR", "USD", "GBP"],
                            value="EUR",
                            label="Currency",
                        )

                analyze_btn = gr.Button("Analyze Panel", variant="primary", size="lg")

            # RIGHT: Results
            with gr.Column(scale=1):
                status_output = gr.HTML(
                    value="<div style='text-align:center;padding:40px;color:#666;'>Upload an image to analyze</div>"
                )
                system_output = gr.Markdown()

        # Analysis Tabs
        gr.Markdown("## Detailed Analysis")
        with gr.Tabs():
            with gr.TabItem("Summary"):
                summary_output = gr.Markdown("*Analysis summary will appear here*")

            with gr.TabItem("Technical Details"):
                detailed_output = gr.Markdown("*Detailed technical analysis will appear here*")

            with gr.TabItem("Recommendations"):
                recommendations_output = gr.Markdown("*Maintenance recommendations will appear here*")

            with gr.TabItem("Environmental Impact"):
                environmental_output = gr.Markdown("*Environmental analysis will appear here*")

            with gr.TabItem("History"):
                history_output = gr.Markdown(get_history_html())
                clear_btn = gr.Button("Clear History", variant="secondary", size="sm")

        # Disclaimer
        gr.Markdown("""
<div class="info-callout">
<strong>About this tool:</strong> This application uses AI for analysis and all results should be considered estimates.
The dust detection model provides probabilistic classifications, not absolute determinations.
For critical maintenance decisions, always consult with qualified solar panel professionals.
</div>
        """)

        # Events
        analyze_btn.click(
            fn=analyze_panel,
            inputs=[image_input, location_input, panel_watts, num_panels, electricity_price, currency],
            outputs=[status_output, system_output, summary_output, detailed_output,
                    recommendations_output, environmental_output, history_output],
        )

        clear_btn.click(fn=clear_history, outputs=[history_output])

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860,
                theme=gr.themes.Soft(), css=FRUTIGER_AERO_CSS)
