# glass-via-simulator
Streamlit-based GUI to simulate micro-drilled glass substrates, estimating Young's modulus, dielectric constant, and dielectric loss.

## Features
- Interactive control over **via diameter**, **pitch**, and **substrate thickness**.
- Material property inputs for different **glass types**:
  - Young’s modulus
  - Poisson’s ratio
  - Relative permittivity (εr)
  - Dielectric loss tangent (tan δ)
- Generates:
  - **Planar maps** (XY) at mid-thickness
  - **Side maps** (XZ) at mid-width
  - **3D topology plots** for:
    - Young’s modulus distribution
    - Effective permittivity (ε′)
    - Dielectric loss (tan δ)

## Requirements
Install dependencies:
```bash
pip install -r requirements_simple.txt
streamlit run via_topology_app_simple.py
