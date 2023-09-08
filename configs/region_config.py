REGION_CONFIG = {
    "4_4_ADJACENCY1": [['intersection_2_2', 'intersection_1_1', 'intersection_2_1', 'intersection_3_1', 'dummy'],
                       ['intersection_1_4', 'dummy', 'intersection_1_3', 'intersection_2_3', 'intersection_1_2'],
                       ['intersection_4_3', 'intersection_3_2', 'intersection_4_2', 'dummy', 'intersection_4_1'],
                       ['dummy', 'intersection_2_4', 'intersection_3_4', 'intersection_4_4', 'intersection_3_3']],
    "4_4_ADJACENCY2": [['intersection_1_3', 'dummy', 'intersection_1_2', 'intersection_2_2', 'intersection_1_1'],
                       ['intersection_3_2', 'intersection_2_1', 'intersection_3_1', 'intersection_4_1', 'dummy'],

                       ['dummy', 'intersection_1_4', 'intersection_2_4', 'intersection_3_4', 'intersection_2_3'],
                       ['intersection_4_4', 'intersection_3_3', 'intersection_4_3', 'dummy', 'intersection_4_2']],
    "4_4_GRID": [['dummy', 'intersection_1_4', 'intersection_2_4',
                  'dummy', 'intersection_1_3', 'intersection_2_3'],
                 ['dummy', 'intersection_1_2', 'intersection_2_2',
                  'dummy', 'intersection_1_1', 'intersection_2_1'],
                 ['intersection_3_4', 'intersection_4_4', 'dummy',
                  'intersection_3_3', 'intersection_4_3', 'dummy'],
                 ['intersection_3_2', 'intersection_4_2', 'dummy',
                  'intersection_3_1', 'intersection_4_1', 'dummy']],
    "16_3_ADJACENCY1": [['intersection_2_2', 'intersection_1_1', 'intersection_2_1', 'intersection_3_1', 'dummy'],
                       ['intersection_1_4', 'dummy', 'intersection_1_3', 'intersection_2_3', 'intersection_1_2'],
                       ['intersection_3_4', 'dummy', 'intersection_3_3', 'dummy', 'intersection_3_2'],
                       ['dummy', 'intersection_1_5', 'intersection_2_5', 'intersection_3_5', 'intersection_2_4'],
                       ['intersection_2_7', 'intersection_1_6', 'intersection_2_6', 'intersection_3_6', 'dummy'],
                       ['intersection_1_9', 'dummy', 'intersection_1_8', 'intersection_2_8', 'intersection_1_7'],
                       ['intersection_3_9', 'dummy', 'intersection_3_8', 'dummy', 'intersection_3_7'],
                       ['dummy', 'intersection_1_10', 'intersection_2_10', 'intersection_3_10', 'intersection_2_9'],
                       ['intersection_2_12', 'intersection_1_11', 'intersection_2_11', 'intersection_3_11', 'dummy'],
                       ['intersection_1_14', 'dummy', 'intersection_1_13', 'intersection_2_13', 'intersection_1_12'],
                       ['intersection_3_14', 'dummy', 'intersection_3_13', 'dummy', 'intersection_3_12'],
                       ['dummy', 'intersection_1_15', 'intersection_2_15', 'intersection_3_15', 'intersection_2_14'],
                       ['dummy', 'intersection_1_16', 'intersection_2_16', 'intersection_3_16', 'dummy'], ],
    "16_3_ADJACENCY2": [['dummy', 'intersection_1_5', 'intersection_2_5', 'intersection_3_5', 'intersection_2_4'],
                        ['intersection_3_4', 'intersection_2_3', 'intersection_3_3', 'dummy', 'intersection_3_2'],
                        ['intersection_3_14', 'intersection_2_13', 'intersection_3_13', 'dummy', 'intersection_3_12'],
                        ['intersection_2_12', 'intersection_1_11', 'intersection_2_11', 'intersection_3_11', 'dummy'],
                        ['intersection_2_2', 'intersection_1_1', 'intersection_2_1', 'intersection_3_1', 'dummy'],
                        ['dummy', 'intersection_1_16', 'intersection_2_16', 'intersection_3_16', 'dummy'],
                        ['dummy', 'intersection_1_15', 'intersection_2_15', 'intersection_3_15', 'intersection_2_14'],
                        ['intersection_1_9', 'dummy', 'intersection_1_8', 'intersection_2_8', 'intersection_1_7'],
                        ['dummy', 'intersection_1_10', 'intersection_2_10', 'intersection_3_10', 'intersection_2_9'],
                        ['intersection_1_14', 'dummy', 'intersection_1_13', 'dummy', 'intersection_1_12'],
                        ['intersection_3_9', 'dummy', 'intersection_3_8', 'dummy', 'intersection_3_7'],
                        ['intersection_2_7', 'intersection_1_6', 'intersection_2_6', 'intersection_3_6', 'dummy'],
                        ['intersection_1_4', 'dummy', 'intersection_1_3', 'dummy', 'intersection_1_2']],
    "16_3_ADJACENCY3": [['intersection_3_9', 'intersection_2_8', 'intersection_3_8', 'dummy', 'intersection_3_7'],
                        ['dummy', 'intersection_1_10', 'intersection_2_10', 'intersection_3_10', 'intersection_2_9'],
                        ['intersection_1_9', 'dummy', 'intersection_1_8', 'dummy', 'intersection_1_7'],
                        ['intersection_2_7', 'intersection_1_6', 'intersection_2_6', 'intersection_3_6', 'dummy'],
                        ['intersection_1_4', 'dummy', 'intersection_1_3', 'intersection_2_3', 'intersection_1_2'],
                        ['intersection_2_2', 'intersection_1_1', 'intersection_2_1', 'intersection_3_1', 'dummy'],
                        ['dummy', 'intersection_1_5', 'intersection_2_5', 'intersection_3_5', 'intersection_2_4'],
                        ['intersection_1_14', 'dummy', 'intersection_1_13', 'intersection_2_13', 'intersection_1_12'],
                        ['dummy', 'intersection_1_16', 'intersection_2_16', 'intersection_3_16', 'dummy'],
                        ['intersection_3_14', 'dummy', 'intersection_3_13', 'dummy', 'intersection_3_12'],
                        ['intersection_2_12', 'intersection_1_11', 'intersection_2_11', 'intersection_3_11', 'dummy'],
                        ['intersection_3_4', 'dummy', 'intersection_3_3', 'dummy', 'intersection_3_2'],
                        ['dummy', 'intersection_1_15', 'intersection_2_15', 'intersection_3_15', 'intersection_2_14']],
    "16_3_ADJACENCY4": [['intersection_2_2', 'intersection_1_1', 'intersection_2_1', 'intersection_3_1', 'dummy'],
                        ['intersection_1_4', 'dummy', 'intersection_1_3', 'intersection_2_3', 'intersection_1_2'],
                        ['intersection_3_4', 'dummy', 'intersection_3_3', 'dummy', 'intersection_3_2'],
                        ['dummy', 'intersection_1_16', 'intersection_2_16', 'intersection_3_16', 'dummy'],
                        ['intersection_1_9', 'dummy', 'intersection_1_8', 'intersection_2_8', 'intersection_1_7'],
                        ['dummy', 'intersection_1_15', 'intersection_2_15', 'intersection_3_15', 'intersection_2_14'],
                        ['dummy', 'intersection_1_5', 'intersection_2_5', 'intersection_3_5', 'intersection_2_4'],
                        ['dummy', 'intersection_1_10', 'intersection_2_10', 'intersection_3_10', 'intersection_2_9'],
                        ['intersection_2_7', 'intersection_1_6', 'intersection_2_6', 'intersection_3_6', 'dummy'],
                        ['intersection_2_12', 'intersection_1_11', 'intersection_2_11', 'intersection_3_11', 'dummy'],
                        ['intersection_3_14', 'intersection_2_13', 'intersection_3_13', 'dummy', 'intersection_3_12'],
                        ['intersection_1_14', 'dummy', 'intersection_1_13', 'dummy', 'intersection_1_12'],
                        ['intersection_3_9', 'dummy', 'intersection_3_8', 'dummy', 'intersection_3_7']],
    "16_3_ADJACENCY5": [['intersection_1_4', 'dummy', 'intersection_1_3', 'intersection_2_3', 'intersection_1_2'],
                        ['intersection_1_9', 'dummy', 'intersection_1_8', 'intersection_2_8', 'intersection_1_7'],
                        ['dummy', 'intersection_1_15', 'intersection_2_15', 'intersection_3_15', 'intersection_2_14'],
                        ['dummy', 'intersection_1_10', 'intersection_2_10', 'intersection_3_10', 'intersection_2_9'],
                        ['intersection_1_14', 'dummy', 'intersection_1_13', 'intersection_2_13', 'intersection_1_12'],
                        ['dummy', 'intersection_1_5', 'intersection_2_5', 'intersection_3_5', 'intersection_2_4'],
                        ['intersection_3_9', 'dummy', 'intersection_3_8', 'dummy', 'intersection_3_7'],
                        ['dummy', 'intersection_1_16', 'intersection_2_16', 'intersection_3_16', 'dummy'],
                        ['intersection_3_14', 'dummy', 'intersection_3_13', 'dummy', 'intersection_3_12'],
                        ['intersection_2_7', 'intersection_1_6', 'intersection_2_6', 'intersection_3_6', 'dummy'],
                        ['intersection_3_4', 'dummy', 'intersection_3_3', 'dummy', 'intersection_3_2'],
                        ['intersection_2_12', 'intersection_1_11', 'intersection_2_11', 'intersection_3_11', 'dummy'],
                        ['intersection_2_2', 'intersection_1_1', 'intersection_2_1', 'intersection_3_1', 'dummy']],
    "16_3_ADJACENCY6": [['dummy', 'intersection_1_5', 'intersection_2_5', 'intersection_3_5', 'intersection_2_4'],
                        ['intersection_1_14', 'dummy', 'intersection_1_13', 'intersection_2_13', 'intersection_1_12'],
                        ['dummy', 'intersection_1_16', 'intersection_2_16', 'intersection_3_16', 'dummy'],
                        ['intersection_2_7', 'intersection_1_6', 'intersection_2_6', 'intersection_3_6', 'dummy'],
                        ['intersection_1_4', 'dummy', 'intersection_1_3', 'intersection_2_3', 'intersection_1_2'],
                        ['dummy', 'intersection_1_15', 'intersection_2_15', 'intersection_3_15', 'intersection_2_14'],
                        ['intersection_3_9', 'intersection_2_8', 'intersection_3_8', 'dummy', 'intersection_3_7'],
                        ['dummy', 'intersection_1_10', 'intersection_2_10', 'intersection_3_10', 'intersection_2_9'],
                        ['intersection_3_14', 'dummy', 'intersection_3_13', 'dummy', 'intersection_3_12'],
                        ['intersection_2_2', 'intersection_1_1', 'intersection_2_1', 'intersection_3_1', 'dummy'],
                        ['intersection_1_9', 'dummy', 'intersection_1_8', 'dummy', 'intersection_1_7'],
                        ['intersection_3_4', 'dummy', 'intersection_3_3', 'dummy', 'intersection_3_2'],
                        ['intersection_2_12', 'intersection_1_11', 'intersection_2_11', 'intersection_3_11', 'dummy']],
}