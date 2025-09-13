import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from liver_preprocessing import LiverDataPreprocessor
from liver_3d_model import Liver3DAutoencoder

st.set_page_config(
    page_title="LiverScan AI",
    page_icon="🫘",
    layout="wide"
)

st.title("🫘 LiverScan AI - Liver Anomaly Detection")
st.markdown("### Advanced 3D Liver Anomaly Detection System")

@st.cache_resource
def load_liver_system():
    model_path = "../models/best_liver_3d_autoencoder.pth"
    if not Path(model_path).exists():
        return None, None, None, None, False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Liver3DAutoencoder()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    preprocessor = LiverDataPreprocessor("C:\\Users\\saura\\unetp_3d_liver\\Task03_Liver")
    threshold = 0.013188
    
    return model, preprocessor, threshold, device, True

def analyze_liver_volume(preprocessor, model, device, threshold, volume_idx):
    volume_path = preprocessor.image_files[volume_idx]
    mask_path = preprocessor.label_files[volume_idx]
    
    volume, mask = preprocessor.preprocess_liver_volume(volume_path, mask_path)
    if volume is None:
        return None
    
    liver_mask = mask > 0
    liver_volume = volume.copy()
    liver_volume[~liver_mask] = 0
    
    volume_tensor = torch.FloatTensor(liver_volume[np.newaxis, np.newaxis, ...]).to(device)
    
    with torch.no_grad():
        reconstructed = model(volume_tensor)
        reconstruction_error = torch.mean((volume_tensor - reconstructed) ** 2).item()
    
    is_anomaly = reconstruction_error > threshold
    confidence = reconstruction_error / threshold
    
    return {
        'file_name': volume_path.name,
        'reconstruction_error': reconstruction_error,
        'is_anomaly': is_anomaly,
        'confidence': confidence,
        'liver_voxels': np.sum(liver_mask),
        'volume': liver_volume
    }

def create_matplotlib_visualization(volume):
    """Create visualization using matplotlib instead of plotly"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show three slices
    mid_z = volume.shape[2] // 2
    axes[0].imshow(volume[:, :, mid_z//2], cmap='hot')
    axes[0].set_title('Liver Slice 1')
    axes[0].axis('off')
    
    axes[1].imshow(volume[:, :, mid_z], cmap='hot')
    axes[1].set_title('Liver Slice 2 (Middle)')
    axes[1].axis('off')
    
    axes[2].imshow(volume[:, :, mid_z + mid_z//2], cmap='hot')
    axes[2].set_title('Liver Slice 3')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    model, preprocessor, threshold, device, loaded = load_liver_system()
    
    if not loaded:
        st.error("❌ Liver model not found!")
        return
    
    st.success(f"✅ Liver system loaded on {device}")
    
    # Sidebar
    st.sidebar.title("🫘 Liver Controls")
    st.sidebar.info(f"""
    **LiverScan AI:**
    - Model: 3D Liver Autoencoder
    - Training: 201 liver volumes
    - Validation Loss: 0.017603
    - Detection: 66.7% on extreme cases
    - Hardware: RTX 4050 optimized
    """)
    
    threshold = st.sidebar.slider("Detection Threshold", 0.005, 0.025, threshold, 0.001)
    
    # Main interface
    st.markdown("#### Analyze Liver Volumes")
    
    volume_idx = st.selectbox(
        "Select Liver Volume",
        range(min(10, len(preprocessor.image_files))),
        format_func=lambda x: f"Liver {x+1}: {preprocessor.image_files[x].name}"
    )
    
    if st.button("🔍 Analyze Liver", type="primary"):
        with st.spinner("Analyzing liver volume..."):
            result = analyze_liver_volume(preprocessor, model, device, threshold, volume_idx)
            
            if result:
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if result['is_anomaly']:
                        st.error("🚨 LIVER ANOMALY")
                    else:
                        st.success("✅ NORMAL LIVER")
                
                with col2:
                    st.metric("Reconstruction Error", f"{result['reconstruction_error']:.6f}")
                
                with col3:
                    st.metric("Confidence", f"{result['confidence']:.2f}x")
                
                with col4:
                    st.metric("Liver Voxels", f"{result['liver_voxels']:,}")
                
                # Visualization using matplotlib
                st.markdown("#### Liver Volume Slices")
                fig = create_matplotlib_visualization(result['volume'])
                st.pyplot(fig)
                
                # Details
                st.markdown("#### Analysis Results")
                st.write(f"**File:** {result['file_name']}")
                st.write(f"**Classification:** {'🚨 Anomalous Liver' if result['is_anomaly'] else '✅ Normal Liver'}")
                st.write(f"**Error vs Threshold:** {result['reconstruction_error']:.6f} vs {threshold:.6f}")
                st.write(f"**Confidence Level:** {result['confidence']:.2f}x threshold")
            else:
                st.error("Failed to analyze liver volume")

if __name__ == "__main__":
    main()
