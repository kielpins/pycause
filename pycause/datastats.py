"""
Define DataStats class for managing statistical data.
"""

import numpy as np

class CachedAttribute(object):
    '''
    Computes attribute value and caches it in instance.
    Not actually used in code at present, but could be useful for later refactoring.
    Author: Denis Otkidach

    Example of use:
        class MyClass(object):
            def myMethod(self):
                # ...
            myMethod = CachedAttribute(myMethod)
    Use "del inst.myMethod" to clear cache.
    '''

    def __init__(self, method, name=None):
        self.method = method
        self.name = name or method.__name__

    def __get__(self, inst, cls):
        if inst is None:
            return self
        result = self.method(inst)
        setattr(inst, self.name, result)
        return result

class DataStats(object):
    """Manages statistical information on causal model.
    
    Allows for specification of statistics "by hand" when sample data is not available.
    """

    _PREC = 1e-9
    
    def __init__(self, var_names=None, raw_data=None):
        """Create DataStats object. Each column of raw_data corresponds to a variable named in var_names.
        
        TODO 3: replace this with a more standard data structure (named numpy array? pandas?)
        :param var_names: names of data variables
        :param raw_data: sample data
        """

        self.var_names = var_names
        if raw_data is not None:
            self.raw_data = raw_data
        
    def get_corr_mat(self):
        """Get correlation matrix, either by returning cached value or calculating from raw data.
        
        :return: correlation matrix
        """

        if not hasattr(self,'corr_mat'): # if corr_mat doesn't exist, calculate it
            # by default, numpy.corrcoef assumes each *row* corresponds to a variable
            # for our *column* variables, need to set rowvar = False
            if hasattr(self,'raw_data'):
                self.corr_mat = np.corrcoef(self.raw_data,rowvar=False)
            else:
                Exception('Raw data not present, cannot compute correlation matrix.')
        return self.corr_mat
    
    def set_corr_mat(self, corr_mat):
        """Set correlation matrix by hand. Use when raw data is not present.
        
        :param corr_mat: correlation matrix
        """

        if not hasattr(self,'raw_data'):
            self.corr_mat = corr_mat
        else:
            Exception('Raw data present! Compute correlation matrix using get_corr_mat().')

    def get_prec_mat(self, var_set_key):
        """Get precision matrix for a desired conditioning set.
        
        The precision matrix is an intermediate variable for calculating partial correlation.
        
        :param var_set_key: set of conditioning variables
        :return: precision matrix for conditioning variables specified in var_set_key
        """

        # initialize dictionary of precision matrices
        if not hasattr(self,'prec_mat'):
            self.prec_mat = dict()

        # if the relevant precision matrix is not in the dictionary, calculate it
        if var_set_key not in self.prec_mat.keys():
            corr_mat = self.get_corr_mat()
            var_list = list(var_set_key)
            var_corr = corr_mat[np.ix_(var_list,var_list)]
            self.prec_mat[var_set_key] = np.linalg.pinv(var_corr)

        return self.prec_mat[var_set_key]
       
    def get_part_corr(self, pair, cond_set):
        """Get partial correlation and corresponding Fisher's z for a pair of variables, conditioned on a set of other variables.
        
        :param pair: variables to correlate
        :param cond_set: conditioning variables
        :return: part_corr = partial correlation, fisher_z = Fisher's z statistic
        """

        # load the relevant precision matrix
        var_set = list(pair)+list(cond_set)
        var_set_keys = tuple(sorted(var_set))
        prec_mat = self.get_prec_mat(var_set_keys)

        # compute partial correlation
        var_range = range(len(var_set_keys))
        xi = [ind for ind in var_range if var_set_keys[ind]==pair[0]][0]
        yi = [ind for ind in var_range if var_set_keys[ind]==pair[1]][0]
        part_corr = -prec_mat[xi,yi]/np.sqrt(prec_mat[xi,xi]*prec_mat[yi,yi])

        # compute Fisher z statistic for comparison to standard normal distribution
        nsamp = self.get_nsamp()
        ncond = len(cond_set)
        with np.errstate(divide='ignore'):
            if abs(part_corr) > 1-self._PREC:
                part_corr = 1-2*self._PREC
            fisher_z = np.sqrt(nsamp-ncond-3)*np.arctanh(part_corr)
        return part_corr,fisher_z    

    def get_nsamp(self):
        """Get the number of samples for computation of Fisher's z.
        
        The number of samples is computed from the sample data, if available.
        
        :return: number of samples
        """

        if hasattr(self,'raw_data'):
            nsamp = np.shape(self.raw_data)[0]
        elif hasattr(self,'nsamp'):
            nsamp = self.nsamp
        else:
            Exception('Number of samples is unknown and cannot be computed from data.')
        return nsamp
    
    def set_nsamp(self,nsamp):
        """Set the number of samples.
        
        Use only in the absence of actual sample data.
        
        :param nsamp: number of samples
        """

        if not hasattr(self,'raw_data'):
            self.nsamp = nsamp
        else:
            Exception('Raw data present! Compute number of samples using get_nsamp().')
