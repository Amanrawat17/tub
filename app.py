import streamlit as st
from rdkit import Chem
from PIL import Image
import pickle
from main import drawing_molecule, generate_morgan_fingerprint, compute_all_descriptors, molecular_properties, model_predict

# Load pre-trained Random Forest model
rf_model_path = r'C:\Users\aman\Desktop\web\xgb_model.pkl'

# Streamlit app layout
def main():
    st.title("Molecule Activity Prediction App")

    # Input section (text box in upper middle part)
    st.markdown("<h3 style='text-align: center;'>Input SMILES string</h3>", unsafe_allow_html=True)
    smile_input = st.text_input(label="", placeholder="Enter SMILES string")

    if st.button("Submit"):
        if smile_input:
            # Generate molecular image and descriptors
            mol_image = drawing_molecule(smile_input)
            fingerprint = generate_morgan_fingerprint(smile_input)
            descriptors = compute_all_descriptors(smile_input)

            # Handle valid SMILES and proceed
            mol = Chem.MolFromSmiles(smile_input)
            if mol:
                # Get molecular properties
                try:
                    properties = molecular_properties(smile_input)
                    mol_weight = properties["Molecular Weight"]
                    mol_formula = properties["Chemical Formula"]
                except Exception as e:
                    mol_weight = "N/A"
                    mol_formula = "N/A"
                    st.error(f"Error calculating molecular properties: {e}")

                # Prediction using Random Forest
                try:
                    rf_pred, rf_conf = model_predict(rf_model_path, smile_input)
                except Exception as e:
                    st.error(f"Error predicting with Random Forest model: {e}")
                    return

                # Output section (two-column layout)
                col1, col2 = st.columns([1, 2])

                # Left column for molecule image
                with col1:
                    st.image(mol_image, caption="Molecule Image", use_column_width=True)

                # Right column for molecule properties and predictions
                with col2:
                    st.markdown(f"**Molecular Weight:** {mol_weight:.2f} g/mol" if mol_weight != "N/A" else "**Molecular Weight:** N/A")
                    st.markdown(f"**Chemical Formula:** {mol_formula}")
                    st.markdown(f"---")
                    st.markdown(f"### **Random Forest Prediction:**")
                    st.markdown(f"**Prediction:** {'Active' if rf_pred == 'Active' else 'Inactive'}")
                    st.markdown(f"**Confidence:** {rf_conf * 100:.2f}%")
            else:
                st.error("Invalid SMILES string. Please check the input.")
        else:
            st.error("Please enter a valid SMILES string.")

if __name__ == "__main__":
    main()
