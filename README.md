# HeatLens — Urban Heat Intelligence & Cooling Planner

HeatLens is a polished Streamlit hackathon prototype for exploring urban heat islands, locating hotspots, and planning practical cooling interventions.

## Run locally in VS Code

1. Open the repository folder in VS Code.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows
   .venv\\Scripts\\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start HeatLens:

   ```bash
   streamlit run app.py
   ```

The application uses a deterministic synthetic heat surface by default, so it works offline and is ready for a live demo. Use the sidebar controls to change the study area, time of day, vegetation, surface reflectivity, and water features.

## Prototype features

- Interactive heatmap with hotspot markers and optional cooling overlay
- Simulated temperature field for a campus, neighbourhood, or small city area
- KPI cards for average temperature, hottest zone, cooling opportunity, and vulnerable population
- Clickable intervention plan with cost, impact, and carbon benefits
- Plotly temperature distribution and intervention impact charts
- Downloadable CSV snapshot of the modeled grid

This is an educational planning prototype, not a replacement for calibrated remote-sensing or municipal sensor data.
