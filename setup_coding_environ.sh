# This is my personal setup. It does not mean it will work for you. This also might not run that nicely, I mostly use it to keep track of what I do rather than for fresh setups 

# Install pyenv 
brew install pyenv
brew install pyenv-virtualenv
brew install pyenv-virtualenvwrapper

# Change directory names if desired
mkdir ~/.ve
mkdir ~/workspace
echo 'export WORKON_HOME=~/.ve' >> ~/.bashrc
echo 'export PROJECT_HOME=~/workspace' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Update to latest python3
pyenv install 3.7.4
pyenv install 2.7.13

# Setup 4 virtual envs to use 
pyenv virtualenv 3.7.4 jupyter3
pyenv virtualenv 3.7.4 tools3 
pyenv virtualenv 2.7.13 ipython2 
pyenv virtualenv 2.7.13 tools2

# Start with python 3 setup 
pyenv activate jupyter3
pip install --upgrade pip 
pip install jupyter # sometimes this won't install right
python -m ipykernel install --user 
pyenv deactivate # may need to restart shell with exec $SHELL after this

# Now python 2
pyenv activate ipython2 
pip install --upgrade pip 
pip install ipykernel
python -m ipykernel install --user
pyenv deactivate

# Install python 3 packages
pyenv activate tools3
pip install -r requirements.txt
pyenv deactivate

# install python 2 packages
# pip install -e ~/Github/nomics

# Make everything play nice
pyenv global 3.7.4 2.7.13 jupyter3 ipython2 tools3 tools2

# Will isntall dependences on start 
echo 'pyenv virtualenvwrapper_lazy' >> ~/.bashrc

# Restart shell
exec $SHELL