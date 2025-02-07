# log_generator

Generate a summary file from solar simulator measurement data.

## Installation and Usage

### Windows (non-Python users)

Download and run the latest release for you operating system from [here](https://github.com/jmball/log_generator/releases).

### Windows and MacOS (Python users)

Create and activate a new Python (version 3.13) virtual environment e.g. using [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), [venv](https://docs.python.org/3/library/venv.html) etc. Then clone this repository using [git](https://git-scm.com) and navigate to its newly created directory:

```
git clone https://github.com/jmball/log_generator.git
cd log_generator
```

Install the dependencies into the virtual environment using:

```
pip install -r requirements.txt
```

To run the program with a GUI on Windows (or Linux) call:

```
python log_generator.py
```

or on a MacOS call:

```
pythonw log_generator.py
```

To skip the GUI use:

```
python log_generator.py --ignore-gooey "[folder]" --stack_ivs
```

where `[folder]` is the absolute path to the folder containing data and `--stack_ivs` is an optional flag indicating whether you want all IV files for a given slot, label, device, and illumination condition grouping stacked into new single files. The flag can be omitted to avoid creating stacked files.

### Linux

First, install the wxPython prerequisites listed [here](https://github.com/wxWidgets/Phoenix#prerequisites).

In addition, if your distribution's package manager doesn't include tkinter with your Python installation (e.g. Ubuntu), it must be installed separately (e.g. `sudo apt install python3.x-tk`, where x denotes your version of python3).

Then follow the instructions for 'Windows and MacOS (Python users)' above.

## Build instructions

To compile the program into a standalone binary file first follow the 'Installation and Usage' instructions for 'Windows and MacOS (Python users)' or 'Linux' above until you have installed the dependencies from the `requirements.txt` file. Then run:

```
pyinstaller log_generator.spec
```

This will create two new folders in the current directory called `build` and `dist`. The binary file is in the `dist` folder and will be called `log_generator.exe` on Windows, `log_generator.app` on MacOSX, and just `log_generator` on Linux.

## Notes

It is strongly recommended that users of this program read and understand the code to ensure that the processing methods it uses are appropriate for their use case.

Parameters derived from I-V curves (Isc, Voc, Pmax, Imp, Vmp, FF, Jsc, Pdmax, Jmp, Rs, and Rsh) are estimated using linear interpolation. While this often works well, irregular I-V curves, e.g. with noise, can lead to inaccurate parameter estimation. Cross-checking the I-V curves with the summary data is highly recommended.

Rs and Rsh are estimated from evaluating the interpolated gradient (2nd order central differences method) of the I-V curve at open-circuit and at short-circuit, respectively, following the conventional approach. However, these estimates only represent real resistances in a meaningful way for idealised, well-behaved devices. Particularly for lab-based solar cell research based on emerging technologies, a more detailed analysis/modelling is often required to determine real resistances. In these cases, Rs and Rsh estimated using this program should probably only be considered as representative of the shape of the I-V curve.

Parameters derived from time-dependent measurements (Iss, Vss, Pss, Jss, Pdss) are estimated by averaging the final N points (default N=10) of the measurement.
