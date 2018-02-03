"""
Functions for loading and formatting data files as datastats.DataStats objects.
[1] Spirtes, Glymour, and Scheines (SGS), Causation, Prediction, and Search, 2nd ed.
"""

import numpy as np
import sys
import os.path
import datastats

#base_dir = 'C:\\Users\\Dave\\Documents\\causal\\'

def load_file(fname, **kwargs):
    """Create DataStats object from sample data in file.
    
    :param fname: file name
    :param kwargs: pass through kwargs to numpy.loadtext
    :return: DataStats object
    """

#    exec_dir = os.path.dirname(sys.argv[0])
#    file_name = exec_dir + '\\' + fname
    raw_data = np.loadtxt(fname, **kwargs)
    return datastats.DataStats(raw_data=raw_data)

def lucas0_train():
    """Load LUCAS0 training dataset, retrieved from http://www.causality.inf.ethz.ch/challenge.php?page=datasets
    
    :return: DataStats object containing LUCAS0 sample data
    """

    file_name = 'lucas0_text\\lucas0_train.data'
    file_path = base_dir+file_name
    
    var_names =   {
                0: 'Smoking',
                1: 'Yellow_Fingers',
                2: 'Anxiety',
                3: 'Peer_Pressure',
                4: 'Genetics',
                5: 'Attention_Disorder',
                6: 'Born_an_Even_Day',
                7: 'Car_Accident',
                8: 'Fatigue',
                9: 'Allergy',
                10: 'Coughing'
                }
    
    try:
        imported_data.any()
    except NameError:
        imported_data = np.genfromtxt(file_path,dtype=int)
        # genfromtxt doesn't throw error if file not read properly
        try:
            imported_data[0]
        except IndexError:
            raise Exception('Data failed to read!')
    lucas0_train_data = datastats.DataStats(var_names,imported_data)
    return lucas0_train_data

def pubprod():
    """Load example statistics on publication productivity from [1], sec. 5.8.1, p. 97. Note that raw sample data is not provided - only correlation matrix.
    
    :return: DataStats object containing statistics
    """

    file_name = 'sgs_pubprod.txt'
    file_path = base_dir+file_name
    
    try:
        imported_data.any()
    except NameError:
        imported_data = np.genfromtxt(file_path,dtype=float,delimiter='\t',names=True)
        # genfromtxt doesn't throw error if file not read properly
        try:
            imported_data[0]
        except IndexError:
            raise Exception('Data failed to read!')
    n_samp = 86+76 # see SGS p. 98
    names_list = [it[0] for it in imported_data.dtype.descr]
    var_names = dict(zip(range(len(names_list)),names_list))
    pub_prod_data = datastats.DataStats(var_names)
    corr_mat = imported_data.view(dtype=float).reshape(len(imported_data),-1)
    # fill in NaN values; SGS only give lower triangular correlation matrix
    for it1 in range(np.shape(corr_mat)[0]):
        for it2 in range(np.shape(corr_mat)[1]):
            if np.isnan(corr_mat[it1,it2]):
                corr_mat[it1,it2] = corr_mat[it2,it1]
    pub_prod_data.set_corr_mat(corr_mat)
    pub_prod_data.set_n_samp(n_samp)
    return pub_prod_data
  
