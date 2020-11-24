FROM pypy:3

WORKDIR /usr/src/app

# Install system dependencies
RUN apt update
RUN apt install -y libblas-dev liblapack-dev gfortran libatlas-base-dev libhdf5-serial-dev
RUN apt install -y libopenmpi-dev

# Install python dependencies
RUN pip install --no-cache-dir numpy scipy matplotlib ase pandas
RUN pip install --no-cache-dir mpi4py tables

# Modify arraywrapper util to enforce compatibility with PyPy
RUN wget https://pastebin.com/raw/Xm281ZpK
RUN mv Xm281ZpK "/opt/pypy/site-packages/ase/utils/arraywrapper.py"

# Copy source into app folder
COPY . .

# Run pypy with the test script
ENV PYTHONPATH "/usr/src/app"
CMD [ "pypy3", "./examples/gd_ion_example/gd_ion_simulation.py" ]
