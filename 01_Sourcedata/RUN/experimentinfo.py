# -*- coding: utf-8 -*-
# @Author: Dragan Rangelov <uqdrange>
# @Date:   07-3-2019
# @Email:  d.rangelov@uq.edu.au
# @Last modified by:   uqdrange
# @Last modified time: 21-3-2019
# @License: CC-BY-4.0
#===============================================================================
# importing libraries
#===============================================================================
from __future__ import division, print_function
from psychopy import gui
#===============================================================================
# defining class
#===============================================================================
class ExperimentInfo():
    '''
    convenience class for preseting a GUI dialog and
    getting information about experimental details
    '''
    def __init__(self, title = None, data = [[], [], []]):
        '''
        Params:
            - title: the title of the dialog
            - data: the data types to be collected
                has to be a list containing three lists
                for info_attrs, info_labels and info_values
        '''
        # info_atts - attributes of ExperimentInfo class
        # info_labels - descriptive labels of these attributes to be used in GUI
        # info_values - default attribute values
        info_attrs, info_labels, info_values = list(zip(*data))

        dlgInfo = dict(list(zip(info_labels, info_values)))
        self.__dict__['dlg'] = gui.DlgFromDict(dlgInfo, title = title)

        # assigning attributes and attribute values
        for key, value in dlgInfo.items():
                self.__dict__[info_attrs[info_labels.index(key)]] = value
# test the class
if __name__ == '__main__':
    ExperimentInfo(
        title='Test', 
        data=[
            [
                'testAttr_A',
                'testLabl_A',
                ''
            ],
            [
                'testAttr_B',
                'testLabl_B',
                ''  
            ]
        ]
    )
