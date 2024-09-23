import numpy as np
import pickle as pkl
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors # type: ignore 
from PIL import Image
import io
from sklearn.ensemble import VotingClassifier, StackingClassifier

def drawing_molecule(smile: str):
    '''
    Draws molecule from SMILES string and displays it on Streamlit.

    Parameters:
    smile: str: SMILES string of the molecule

    Returns:
    None
    '''
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        img = Draw.MolToImage(mol)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        img = Image.open(buf)
        return img
    return None


def generate_morgan_fingerprint(smiles: str, radius=2, n_bits=2048) -> np.ndarray:
    '''
    Generates Morgan fingerprints for a given SMILES string.

    Parameters:
    smiles: str: SMILES string of the molecule
    radius: int: radius of the fingerprint
    nBits: int: number of bits in the fingerprint

    Returns:
    np.ndarray: array of fingerprint bits
    '''
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    return np.zeros(n_bits)


def compute_all_descriptors(smiles: str) -> np.ndarray:
    '''
    Generates 208 molecular descriptors for a given SMILES string.

    Parameters:
    smiles: str: SMILES string of the molecule

    Returns:
    np.ndarray: array of molecular descriptors
    '''
    descriptor_names = [desc[0] for desc in Descriptors.descList[:208]]  # Only first 208 descriptors
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  
        return np.zeros(len(descriptor_names))
    descriptors = calculator.CalcDescriptors(mol)
    return np.array(descriptors)


def predict_activity(smiles: str, model_path: str) -> (str, float):
    '''
    Predicts whether the given compound is active or not using a pre-trained model.

    Parameters:
    smiles: str: SMILES string of the molecule
    model_path: str: path to the pre-trained model (.pkl file)

    Returns:
    str: Prediction ("Active" or "Inactive")
    float: Confidence score of the prediction
    '''
    # Load pre-trained model
    with open(model_path, 'rb') as file:
        model = pkl.load(file)

    # Generate feature vector (Morgan + Descriptors)
    morgan = generate_morgan_fingerprint(smiles)
    descriptors = compute_all_descriptors(smiles)
    feature_vector = np.concatenate((morgan, descriptors))

    # Predict
    prediction = model.predict([feature_vector])[0]
    confidence = np.max(model.predict_proba([feature_vector]))  # Get the highest probability

    if prediction == 1:
        return "Active", confidence
    else:
        return "Inactive", confidence


def molecular_properties(smiles: str) -> dict:
    '''
    Computes basic molecular properties such as molecular weight and chemical formula.

    Parameters:
    smiles: str: SMILES string of the molecule

    Returns:
    dict: Dictionary containing molecular weight and chemical formula
    '''
    mol = Chem.MolFromSmiles(smiles)
    properties = {"Molecular Weight": None, "Chemical Formula": None}

    if mol is not None:
        try:
            mol_weight = Descriptors.MolWt(mol)
            chem_formula = rdMolDescriptors.CalcMolFormula(mol)
            properties["Molecular Weight"] = mol_weight
            properties["Chemical Formula"] = chem_formula
        except Exception as e:
            print(f"Error in calculating molecular properties: {e}")
    else:
        print("Invalid SMILES string.")

    return properties


def load_voting_model(voting_model_path: str):
    '''
    Loads a pre-trained VotingClassifier model from a .pkl file.
    
    Parameters:
    voting_model_path: str: Path to the VotingClassifier model file
    
    Returns:
    VotingClassifier: Loaded voting model
    '''
    with open(voting_model_path, 'rb') as file:
        voting_model = pkl.load(file)
    return voting_model


def load_stacking_model(stacking_model_path: str):
    '''
    Loads a pre-trained StackingClassifier model from a .pkl file.
    
    Parameters:
    stacking_model_path: str: Path to the StackingClassifier model file
    
    Returns:
    StackingClassifier: Loaded stacking model
    '''
    with open(stacking_model_path, 'rb') as file:
        stacking_model = pkl.load(file)
    return stacking_model


def predict_with_voting(smiles: str, voting_model: VotingClassifier) -> (str, float):
    '''
    Predicts the activity of a compound using a VotingClassifier.

    Parameters:
    smiles: str: SMILES string of the molecule
    voting_model: VotingClassifier: Loaded voting model

    Returns:
    str: Prediction ("Active" or "Inactive")
    float: Confidence score of the prediction
    '''
    morgan = generate_morgan_fingerprint(smiles)
    descriptors = compute_all_descriptors(smiles)
    feature_vector = np.concatenate((morgan, descriptors))

    # Predict
    prediction = voting_model.predict([feature_vector])[0]
    confidence = np.max(voting_model.predict_proba([feature_vector]))  # Get the highest probability

    if prediction == 1:
        return "Active", confidence
    else:
        return "Inactive", confidence


def predict_with_stacking(smiles: str, stacking_model: StackingClassifier) -> (str, float):
    '''
    Predicts the activity of a compound using a StackingClassifier.

    Parameters:
    smiles: str: SMILES string of the molecule
    stacking_model: StackingClassifier: Loaded stacking model

    Returns:
    str: Prediction ("Active" or "Inactive")
    float: Confidence score of the prediction
    '''
    morgan = generate_morgan_fingerprint(smiles)
    descriptors = compute_all_descriptors(smiles)
    feature_vector = np.concatenate((morgan, descriptors))

    # Predict
    prediction = stacking_model.predict([feature_vector])[0]
    confidence = np.max(stacking_model.predict_proba([feature_vector]))  # Get the highest probability

    if prediction == 1:
        return "Active", confidence
    else:
        return "Inactive", confidence


# Function to load model and predict activity with confidence
def model_predict(model_path, smiles):
    # Load the model from the provided file path
    with open(model_path, 'rb') as f:
        model = pkl.load(f)

    morgan = generate_morgan_fingerprint(smiles)
    descriptors = compute_all_descriptors(smiles)
    vector = np.concatenate((morgan, descriptors))

    # Make prediction and get probability
    prediction = model.predict([vector])
    probability = model.predict_proba([vector])

    # Return the predicted activity (1 for Active, 0 for Inactive) and the maximum confidence score
    activity = 'Active' if prediction[0] == 1 else 'Inactive'
    confidence = probability.max()

    return activity, confidence


if __name__ == '__main__':
    smiles = "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"
    drawing_molecule(smiles)
    morgan = generate_morgan_fingerprint(smiles)
    print(morgan)
    desc = compute_all_descriptors(smiles)
    print(desc)
    # Example: Predict with Voting Model (use correct path for your model)
    voting_model = load_voting_model(r'C:\Users\saatvik\Desktop\FINAL_PAPER\website\best_voting_classifier.pkl')
    prediction, confidence = predict_with_voting(smiles, voting_model)
    print(prediction, confidence)
    stacking_model = load_stacking_model(r'C:\Users\saatvik\Desktop\FINAL_PAPER\website\best_stacking_classifier.pkl')
    prediction, confidence = predict_with_stacking(smiles, stacking_model)
    print(prediction, confidence)
