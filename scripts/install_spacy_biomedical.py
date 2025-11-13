"""
Script to install spaCy biomedical model (en_core_sci_md).

This model is required for accurate biomedical entity recognition.
Run this script if you see warnings about missing en_core_sci_md model.
"""

import subprocess
import sys
import os

def install_scispacy():
    """Install scispacy package."""
    print("Installing scispacy package...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scispacy"])
        print("✓ scispacy installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install scispacy: {e}")
        return False

def install_biomedical_model():
    """Install en_core_sci_md biomedical model."""
    print("Installing en_core_sci_md biomedical model...")
    print("This may take a few minutes...")
    
    model_url = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz"
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", model_url
        ])
        print("✓ en_core_sci_md model installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install model: {e}")
        print("\nAlternative installation method:")
        print("Try running: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz")
        return False

def verify_installation():
    """Verify that the model can be loaded."""
    print("\nVerifying installation...")
    try:
        import spacy
        nlp = spacy.load("en_core_sci_md")
        print(f"✓ Model loaded successfully: {nlp.meta.get('name', 'en_core_sci_md')}")
        
        # Test biomedical entity recognition
        test_text = "metformin treats diabetes and prevents cardiovascular disease"
        doc = nlp(test_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"✓ Test extraction: Found {len(entities)} entities: {entities}")
        
        return True
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("spaCy Biomedical Model Installation")
    print("=" * 60)
    
    # Step 1: Install scispacy
    if not install_scispacy():
        print("\nInstallation failed at scispacy step.")
        sys.exit(1)
    
    # Step 2: Install biomedical model
    if not install_biomedical_model():
        print("\nInstallation failed at model step.")
        sys.exit(1)
    
    # Step 3: Verify
    if verify_installation():
        print("\n" + "=" * 60)
        print("Installation completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Installation completed but verification failed.")
        print("The model may still work - try using it in your code.")
        print("=" * 60)




