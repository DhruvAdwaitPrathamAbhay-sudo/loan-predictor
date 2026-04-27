"""
Download the Lending Club dataset from Kaggle.
Requires Kaggle API credentials (kaggle.json).
"""
import os

try:
    import opendatasets as od
except ImportError:
    print("Installing opendatasets...")
    os.system("python -m pip install opendatasets -q")
    import opendatasets as od

os.makedirs('./data', exist_ok=True)

print("Downloading Lending Club dataset from Kaggle...")
print("You will be prompted for your Kaggle username and API key.")
print("Get them from: https://www.kaggle.com/settings -> API -> Create New Token\n")

od.download("https://www.kaggle.com/datasets/wordsforthewise/lending-club", data_dir='./data')

# Move files if nested
src = './data/lending-club'
if os.path.exists(src):
    import shutil
    for f in os.listdir(src):
        shutil.move(os.path.join(src, f), os.path.join('./data', f))
    os.rmdir(src)

print("\nFiles in ./data:")
for f in os.listdir('./data'):
    size_mb = os.path.getsize(os.path.join('./data', f)) / (1024*1024)
    print(f"  {f} ({size_mb:.1f} MB)")
print("\nDone! Now run: python clean_loan_data.py")
