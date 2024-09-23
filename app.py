from flask import Flask, render_template, request
from rdkit import Chem
from rdkit.Chem import Draw
import pickle
import numpy as np
import io
from PIL import Image

# Load pre-trained Random Forest model
rf_model_path = r'C:\Users\aman\Desktop\web\xgb_model.pkl'
with open(rf_model_path, 'rb') as file:
    rf_model = pickle.load(file)

app = Flask(__name__)

def drawing_molecule(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        img = Draw.MolToImage(mol)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf
    return None

def generate_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    return np.zeros(n_bits)

def compute_all_descriptors(smiles):
    from rdkit.ML.Descriptors import MoleculeDescriptors
    descriptor_names = [desc[0] for desc in Chem.Descriptors.descList[:208]]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        descriptors = calculator.CalcDescriptors(mol)
        return np.array(descriptors)
    return np.zeros(len(descriptor_names))

def model_predict(model, smiles):
    morgan = generate_morgan_fingerprint(smiles)
    descriptors = compute_all_descriptors(smiles)
    feature_vector = np.concatenate((morgan, descriptors))
    prediction = model.predict([feature_vector])
    confidence = model.predict_proba([feature_vector]).max()
    return 'Active' if prediction[0] == 1 else 'Inactive', confidence

def molecular_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    properties = {"Molecular Weight": None, "Chemical Formula": None}
    if mol is not None:
        properties["Molecular Weight"] = Chem.Descriptors.MolWt(mol)
        properties["Chemical Formula"] = Chem.rdMolDescriptors.CalcMolFormula(mol)
    return properties

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        smile_input = request.form.get("smile_input")
        if smile_input:
            mol_image = drawing_molecule(smile_input)
            properties = molecular_properties(smile_input)
            mol_weight = properties.get("Molecular Weight", "N/A")
            mol_formula = properties.get("Chemical Formula", "N/A")

            rf_pred, rf_conf = model_predict(rf_model, smile_input)

            return render_template("index.html", 
                                   mol_image=mol_image,
                                   mol_weight=mol_weight,
                                   mol_formula=mol_formula,
                                   rf_pred=rf_pred,
                                   rf_conf=rf_conf * 100)  # Convert to percentage
        else:
            error = "Please enter a valid SMILES string."
            return render_template("index.html", error=error)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
