import numpy as np


def return_manual_confusion_matrix(threshold, X_no_input, X_yes_input, yes_above_threshold):
    
    if yes_above_threshold:
        tn_result = np.sum(X_no_input <= threshold)
        fn_result = np.sum(X_yes_input <= threshold)
        fp_result = np.sum(X_no_input > threshold)
        tp_result = np.sum(X_yes_input > threshold)
    else:
        tn_result = np.sum(X_no_input > threshold)
        fn_result = np.sum(X_yes_input > threshold)
        fp_result = np.sum(X_no_input <= threshold)
        tp_result = np.sum(X_yes_input <= threshold)
    
    return tn_result, fn_result, fp_result, tp_result

def return_manual_number_of_confusion_matrix_elements(
    thresholds, X_no_input, X_yes_input, yes_above_threshold):

    manual_number_of_TN_result = []
    manual_number_of_FN_result = []
    manual_number_of_TP_result = []
    manual_number_of_FP_result = []

    for threshold in thresholds:

        tn, fn, fp, tp = return_manual_confusion_matrix(threshold, X_no_input, X_yes_input, yes_above_threshold)

        manual_number_of_TN_result.append(tn)
        manual_number_of_FN_result.append(fn)
        manual_number_of_TP_result.append(tp)
        manual_number_of_FP_result.append(fp)

    manual_number_of_TN_result = np.array(manual_number_of_TN_result)
    manual_number_of_FN_result = np.array(manual_number_of_FN_result)
    manual_number_of_TP_result = np.array(manual_number_of_TP_result)
    manual_number_of_FP_result = np.array(manual_number_of_FP_result)
    
    return manual_number_of_TN_result, manual_number_of_FN_result, \
        manual_number_of_TP_result, manual_number_of_FP_result

def entropy(p):
    q = 1 - p
    return -p* np.log2(p) - q* np.log2(q)

def gini(p):
    q = 1 - p
    return 1 - p**2 - q**2

def return_entropies(thresholds, tn_list, fn_list, fp_list, tp_list):

    entropies_group1_result = []
    entropies_group2_result = []
    entropies_average_result = []
    size_total = tn_list[0] + fn_list[0] + fp_list[0] + tp_list[0]

    for i, threshold in enumerate(thresholds):

        group_1_no = tn_list[i]
        group_1_yes = fn_list[i]
        group_2_no = fp_list[i]
        group_2_yes = tp_list[i]

        if group_1_no > 0 and group_1_yes > 0:
            group_1_entropy = entropy(group_1_no / (group_1_no + group_1_yes))
        else:
            group_1_entropy = 0

        if group_2_no > 0 and  group_2_yes > 0:
            group_2_entropy = entropy(group_2_no / (group_2_no + group_2_yes))
        else: 
            group_2_entropy = 0

        entropies_group1_result.append(group_1_entropy)
        entropies_group2_result.append(group_2_entropy)
        entropies_average_result.append(
            ((group_1_no + group_1_yes) * group_1_entropy + \
            (group_2_no + group_2_yes) * group_2_entropy) / \
            size_total
        )

    return entropies_average_result


def return_ginis(thresholds, tn_list, fn_list, fp_list, tp_list):

    ginis_group1_result = []
    ginis_group2_result = []
    ginis_average_result = []
    size_total = tn_list[0] + fn_list[0] + fp_list[0] + tp_list[0]

    for i, threshold in enumerate(thresholds):

        group_1_no = tn_list[i]
        group_1_yes = fn_list[i]
        group_2_no = fp_list[i]
        group_2_yes = tp_list[i]

        if group_1_no > 0 and group_1_yes > 0:
            group_1_gini = gini(group_1_no / (group_1_no + group_1_yes))
        else:
            group_1_gini = 0

        if group_2_no > 0 and  group_2_yes > 0:
            group_2_gini = gini(group_2_no / (group_2_no + group_2_yes))
        else: 
            group_2_gini = 0

        ginis_group1_result.append(group_1_gini)
        ginis_group2_result.append(group_2_gini)
        ginis_average_result.append(
            ((group_1_no + group_1_yes) * group_1_gini + \
            (group_2_no + group_2_yes) * group_2_gini) / \
            size_total
        )
        
    return ginis_average_result 