# Utilities for QCE

import numpy as np

def verify_record(record, ce, coeffs, intercept, new_atom):
    """
    Verify if the obtained record is correct
    record: extracted dictionary
    ce: ICET ce model
    coeffs: linear regression coefficients
    intercept: offset to the CE predictions
    new_atom: template atoms object
    """
    Zs = new_atom.get_atomic_numbers()
    calculate = []

    for k in record.keys():
        if k==0:
            continue
        sums = 0.0
        for entry in record[k][2]:
            product = 1.0
            for product_term in entry:
                product *= product_term[1][Zs[product_term[0]]]
            sums += product
        sums /= record[k][1] 
        calculate.append(sums)

    print('Calcuated via record: ',np.dot([1]+calculate, coeffs) + intercept)
    print('Calculated via ICET: ',ce.predict(new_atom) + intercept)
    
    
def ExtractDataFromLists(records, weights, elements):
    """
    Converts all data to an extracted form that can be used by any of the search processes [GA, BO, DA or Gurobi]
    
    records: List of record type data
    weights: list of weights associated with each of the records (in the same order)
    elements: subset of elements we want to search over
    """
    extracted_data = []
    
    for record,weight in zip(records,weights):

        for k in record.keys():
            if k != 0:
                data = record[k]
                outer_coeff = weight * data[0] / data[1]

                for entry in data[2]: #cluster_orbits
                    term = [outer_coeff, []]
                    for product_term in entry: #(cluster_alphas, cluster_atomic_idxs)
                        coef = []
                        for idx,i in enumerate(elements):
                            coef.append(product_term[1][i])
                        term[1].append([product_term[0],coef])    
                    extracted_data.append(term)
    return extracted_data

def Res2Atoms(state, elements, Allowed_Zs, site_num, template_atom, ce):
        """
        Converts results from DA output to an Atoms object
        state: bit vector from DA
        elements: subset of elements under consideration
        Allowed_Zs: list of atomic numbers that are allowed to occupy sites
        site_num: Number of sites
        template_atom: atom object used for decoration
        ce: list of ce-models

        returns:
            (new-atoms-object, properties)
        """
        shaped_state = state[:len(elements)*site_num].reshape(site_num,len(elements))
        element_indices = np.argmax(shaped_state,axis=1)
        outcome_Z = np.array(elements)[element_indices]
        at = template_atom.copy()

        final_Z, tracker = [], 0
        for Z in at.get_atomic_numbers():
            if Z not in Allowed_Zs:
                final_Z.append(Z)
            else:
                final_Z.append(outcome_Z[tracker])
                tracker += 1

        at.set_atomic_numbers(final_Z)
        return at, [model.predict(at) for model in ce]
