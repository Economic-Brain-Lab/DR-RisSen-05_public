# configure paths and variables
PROJECT=$1

# create environment
nohup \
mamba create \
--name $PROJECT \
--override-channels \
--channel conda-forge \
--channel ejolly \
--strict-channel-priority \
--no-default-packages \
--verbose \
--yes \
screen \
python">=3.8" \
pip \
mne \
numpy \
scipy \
matplotlib \
seaborn \
numba \
pandas \
xlrd \
scikit-learn \
h5py \
pillow \
pingouin \
statsmodels \
jupyter \
joblib \
psutil \
numexpr \
traits \
pyface \
traitsui \
imageio \
tqdm \
imageio-ffmpeg">=0.4.1" \
vtk">=9.0.1" \
pyvista">=0.24" \
pyvistaqt">=0.2.0" \
mayavi \
python-picard \
pyqt \
mffpy">=0.5.7" \
openpyxl \
pymer4 \
pytables">=3.8.0" \
> ./create_environment_$PROJECT.log &

echo "DONE"
