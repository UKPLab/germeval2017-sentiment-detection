virtualenv .

# Install common dependencies (helper tools)
. bin/activate
pip install yolk
pip install numpy
pip install tensorflow
# Do not install Keras from pip because it does not contain the
# CRF layer
# pip install keras
pip install ruamel.yaml
pip install pydot-ng
pip install tabulate
pip install sklearn

# Save dependencies and create a requirements management file
pip freeze > requirements.txt

# Install Keras from CRF fork
pip install git+https://github.com/phipleg/keras.git@crf
# Append reference to CRF fork to requirements.txt
echo "git+https://github.com/phipleg/keras.git@crf" >> requirements.txt

git add .

git commit -sm "Added dependency management for experiment '12_tensorflow_port'"

# Add a symbolic link to the library folder (`shared_modules`)
ln -s ../../../shared_modules src/shared_modules
