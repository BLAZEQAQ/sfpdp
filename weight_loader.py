from datetime import datetime
import numpy as np




def load_weight(name, p):
    weights_SFPDP = np.loadtxt('./Weights/{}/SFP-DP/p={}/weights-SFP-DP.txt'.format(name, p), dtype=float, delimiter=' ')
    weights_SFPDP_MMI = np.loadtxt('./Weights/{}/SFP-DP-MMI/weights-SFP-DP-MMI.txt'.format(name), dtype=float, delimiter=' ')
    weights_SFPDP_LMLT = np.loadtxt('./Weights/{}/SFP-DP-LMLT/weight_LMLT.txt'.format(name), dtype=float,
                                   delimiter=' ')

    weights = {'SFPDP': weights_SFPDP, 'SFPDP_MMI': weights_SFPDP_MMI, 'SFPDP_LMLT': weights_SFPDP_LMLT}
    return weights







