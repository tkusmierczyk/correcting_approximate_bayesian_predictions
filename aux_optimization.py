import numpy as np


def yield_minibatch_rows(i, N, MINIBATCH):
    """ Minibatch optimization via rows subset selection.
    
        Args:
          i  Iteration number 0,1,2,...
    """
    if MINIBATCH>N: MINIBATCH=N

    nbatches_per_epoch = int( np.ceil(N/MINIBATCH) )
    batch_no = i%nbatches_per_epoch    
    if batch_no==0: # shuffle order
        yield_minibatch_rows.rows_order = np.random.permutation(range(N))
    six, eix = batch_no*MINIBATCH, (batch_no+1)*MINIBATCH
    rows = yield_minibatch_rows.rows_order[six: eix] # batch rows
    
    # makes sure that for full-batch the order of rows is preserved
    if MINIBATCH>=N: rows = list(range(N)) 
      
    sgd_scale = N/len(rows) 
    epoch_no = i//nbatches_per_epoch
    return rows, epoch_no, sgd_scale
yield_minibatch_rows.rows_order = None
