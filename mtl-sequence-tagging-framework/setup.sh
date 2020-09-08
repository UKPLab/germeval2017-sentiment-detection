# Create virtual environment and activate it
virtualenv .
. bin/activate

# Install dependencies
pip install -r requirements.txt

# Add a symbolic link to the library folder (`shared_modules`)
ln -s ../../../shared_modules src/shared_modules
