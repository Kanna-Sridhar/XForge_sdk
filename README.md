# XForge SDK 🧠💻

XForge SDK is a specialized Python library for simulating the execution of Deep Learning models (like YOLO) on **Neuromorphic X2** hardware. It provides tools for image preprocessing and post processing, quantization, and hardware performance benchmarking.

---

## 🚀 Features

- **YOLO Integration:** Extract weights directly from YOLOv8 models for hardware analysis.
- **Crossbar Mapping:** Automatically format weights into Signed or Unsigned 64x64/64x32 crossbar grids.
- **Hardware Simulation:** Benchmarks PE (Processing Element) utilization, operation counts, and L2 buffer writes.
- **Image Processing:** Standardized RGB channel splitting and resizing for neuromorphic inputs.

---

## 🛠 Installation

### Install via Git (Recommended)

Install the latest version directly from the GitHub repository:

```bash
pip install git+https://github.com/BMsemi/XForge_sdk.git
```

## 📖 Quick Start


```python
import streamlit as st
import numpy as np
from XForge import NeuromorphicSimulator, ImageProcessor, YOLOWrapper

# Page Config
st.set_page_config(page_title="XForge Dashboard", layout="wide")

# 1. Initialize SDK Components
# These replace the manual logic previously scattered in app.py and user.py
@st.cache_resource
def init_sdk():
    return {
        "sim": NeuromorphicSimulator(num_pes=64),
        "processor": ImageProcessor(),
        "yolo": YOLOWrapper('yolov8n.pt')
    }

sdk = init_sdk()

st.title("🎯 YOLO26N: Neuromorphic Analysis SDK")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file:
    # 2. Use ImageProcessor for standardized loading
    img_array, channels = sdk["processor"].load_rgb_image(uploaded_file)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Object Detection")
        results = sdk["yolo"].predict(img_array)
        st.image(results[0].plot(), caption="Detection Results")
        
```



## 🧪 Development & Testing

Run the test suite after installation:

If using the Streamlit Dashboard:

```bash
streamlit run app.py
```

---

## 🤝 Contributing

For internal use by **BM Labs**. Please ensure all hardware-specific logic is validated against the NeuromorphicX2 core modules before pushing updates.
