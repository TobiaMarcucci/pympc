Repeatability Evaluation Package (REP) for the paper "Mixed-Integer Formulations for Optimal Control of Piecewise-Affine Systems".

# Anonymity warning

The results reported in the paper leverage on the toolbox "pympc".
Since the download of this library could reveal the identity of the authors, the whole toolbox has been submitted to keep the submission anonymous.

# Introduction

The REP can be used to reproduce the results presented in Section 7.2 of the paper (in particular Figures 4, 5, 6 and Tables 2, 3).
(Note that the example in Section 7.1 is not numeric, and hence is not included in the REP.)
This REP is written in Python 2.7.
The results presented in the paper can be reproduced running the files in the folder pympc/repeatability_evaluation (see below for more details).
The results are obtained exploiting the main tools of pympc and the additional files collected in the folder pympc/pympc/control/hscc.

# Dependencies

In order to use the pympc library the following python packages are required:
- numpy, available at http://www.numpy.org
- sympy, available at https://www.sympy.org/en/index.html
- scipy, available at https://www.scipy.org
- matplotlib, available at https://matplotlib.org

Together with these, the commercial solver Gurobi is used for the solution of optimization problems.
The solver can be downloaded at https://www.gurobi.com.
The results presented in the paper are obtained with the version Gurobi 8.0.0.
The download requires registration; the license is free for academic use.
Detailed instructions for the installation of the solver can be found in the Quick Start Guide: https://www.gurobi.com/documentation/.
For more details on the installation of Gurobi's Python interface please refer to Section 12 of the Quick Start Guide.
(The guide suggests to install the Anaconda Python distribution, alternatively one can consider to install the gurobipy module by following the simpler instructions in Section 12.3.)

# Use of the pympc toolbox

Once the dependencies are installed, the pympc directory must be added to the PYTHONPATH.
If you are using bash (on Mac or Linux), this can be done adding
export PYTHONPATH="${PYTHONPATH}:/path_to_pympc_main_directory"
to the ~/.bashrc file.

# REP structure

The folder pympc/repeatability_evaluation contains the following python files (see the headers of the files for more details).
- table_3_data.py:
	This file can be used to reproduce the computations analyzed in Table 3.
	Note that this file takes a very long time to run (~8 hours) since it needs to solve 12 very hard mixed-integer programs.
	The results of these computations are also saved in the paper_data folder so that they are readily available for the other files.
	(One can consider to reduce the time limit of the solver to speed up the running time of this code, see the file's header.)
- table_3_print_results.py:
	This file prints the results computed in the previous.
	It can be modified to directly load the results of table_3_data.py (see the file's header).
- figure_5.py:
	This file can be used to reproduce Figure 5.
- figure_6.py:
	This file can be used to reproduce Figure 6.
- table_2_row_1_data.py:
	This file can be used to reproduce the figures in the first row of Table 2.
	Running this code takes an EXTREMELY long time (~3 days).
	However the results fo this code are not crucial for the comparison between the formulations analyzed in the paper.
	The results of these computations can be loaded as explained in the file's header.
- table_2_row_1_plot.py:
	This file plots the results computed in the previous.
	It can be modified to directly load the results of table_2_row_1_plot.py (see the file's header).
- table_2_row_2_to_5_data.py:
	This file can be used to reproduce the figures in Table 2 from row 2 to row 5.
	Running this file takes ~10 minutes.
	Alternatively, results can be loaded as described in the file's header.
- table_2_row_2_to_5_plot.py:
	This file plots the results computed in the previous.
	It can be modified to directly load the results of table_2_row_2_to_5_data.py (see the file's header).

All the above results can also be reproduced using the Jupyter notebook ball_and_paddle.ipynb.
In addition, this notebook also includes the animation shown in Figure 4 in the paper.
Jupyter is available at https://jupyter.org.
The animation requires the package meshcat-python available at https://github.com/rdeits/meshcat-python.

The code to synthesize the controllers used in the file table_3_data.py can be found in pympc/pympc/control/hscc/controllers.py.
Unit tests for this file are in pympc/pympc/test/test_control/test_hscc/test_controllers.py.