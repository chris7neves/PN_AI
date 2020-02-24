#---------------------------------------------------------------------
# This file will accept a list with all the hyperparameters entered #
# by the user in main. It will then use this list of hyperparameters#
# to define as many networks as necessary, train them, then return  #
# either the individual network objects or just the accuracy and    #
# performance of each network.                                      #
#---------------------------------------------------------------------


class NetworkFactory:
    def __init__(self, params, dloader):

