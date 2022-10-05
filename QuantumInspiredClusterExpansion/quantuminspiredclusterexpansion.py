from icet.core.local_orbit_list_generator import LocalOrbitListGenerator
from icet.core.structure import Structure

def ICET2List(cs, ce, new_atom):
    """
    Converts all the information contained in ICET CE to nested lists of Python
    This enables much faster processing down-the-line than relying simply on ICET or Sympy
    
    cs: ClusterSpace object that was used to train CE model
    ce: ICET cluster expansion model that has been trained
    new_atom: instance of a decorated lattice on which we want to perform DFT
    """
    new_atom #= df.loc[0,'atoms']

    ClusterVector = [1.0] # first entry is for zerolet
    Species_map = ce._cluster_space.species_maps

    SuperOrbit = LocalOrbitListGenerator(cs.orbit_list, Structure.from_atoms(new_atom), 1e-5)
    orbitList = SuperOrbit.generate_full_orbit_list().get_orbit_list()

    record = {}
    record[0] = [ce.parameters[0],[]]
    counter = 0 # keeping track of zero and non-zero ECIs

    # first loop on OrbitList
    for orbit_idx,orbit in enumerate(orbitList):

        # Identify sites
        ClusterSites = []
        ClusterCounts = []
        indices = []
        size_of_each_cluster = orbit.order
        oc = orbit.clusters
        for j in range(len(oc)):
            site = tuple([oc[j].lattice_sites[i].index for i in range(size_of_each_cluster)])
            ClusterSites.append(site)
            ClusterCounts.append(tuple([new_atom.get_atomic_numbers()[j] for j in site]))

        # Allowed number of elements at a site [make this generalize better]
        AllowedOccupations = len(ce._cluster_space.species_maps[0].items())

        for site in orbit.representative_cluster.lattice_sites:
            indices.append(site.index)

        # Second loop on ClusterVectorElements
        for cve_index, cvelement in enumerate(orbit.cluster_vector_elements):

            ClusterVectorElement = 0.0
            Sum = [] # Sum(Product)
            counter += 1

            # Third loop on ClusterCounts
            for k,clustersite in zip(ClusterCounts,ClusterSites):

                # Fourth loop on permutations of multi-component vector
                for permutation in cvelement['site_permutations']:

                    Product = []
                    for i in range(len(k)):
                        single_basis_function = {}
                        index = permutation[i]
                        LocalSpeciesMap = ce._cluster_space.species_maps[0]
                        #LocalSpeciesMap = np.argwhere(Species_map_mapper == set(substitutions[indices[index]]))[0,0]
                        for ks,vs in LocalSpeciesMap.items():
                            single_basis_function[ks] = ce._cluster_space.evaluate_cluster_function(AllowedOccupations,
                                                    cvelement['multicomponent_vector'][index], vs)
                        Product.append([clustersite[i], single_basis_function])

                    Sum.append(Product)

            ClusterVector.append(ClusterVectorElement/cvelement['multiplicity'])
            record[len(ClusterVector)-1] = [ce.parameters[len(ClusterVector)-1], cvelement['multiplicity'], Sum]
            
    return record

def mk_converter_from_atomid_to_siteid(extracted_data):
    '''
    assumption: All sites appear in extracted_data
    '''
    idx_list = set()
    for e in extracted_data:
        if len(e[1])!=0:
            for site in e[1]:
                idx_list.add(site[0])
    atom2site = {atom_idx:i for i, atom_idx in enumerate(idx_list)}
    site2atom = {i:atom_idx for i, atom_idx in enumerate(idx_list)}   
    return atom2site, site2atom, len(atom2site)

def generate_quboinput(extracted_data, atom2site, site_num, bit_per_site, rt = {'table':{}, 'next_new_site_id':0}):
    # replace atomid by siteid and sort internal site-info (id + coeff) list 
    for e in extracted_data:
        for el in e[1]:
            el[0]=atom2site[el[0]]
        e[1].sort(key=lambda x: x[0]) 
    # sort by number of interactions
    extracted_data.sort(key=lambda x: -len(x[1]))

    # decrese order by replacing and generate DA3 input. Addtion is done by python_fjda
    replace_table = rt['table'] # Note: 1.keys/values are site ids 2.each entry generates bit_per_site*2 variables
    next_new_site_id=rt['next_new_site_id']
    ce_bp = []
    rp_bp = []
    oh_bp = []

    base = bit_per_site * site_num
    for e in extracted_data:
        site_id = [e[1][i][0] for i in range(len(e[1]))]
        if e[0]==0:
            assert True, 'outer_coeff is 0!'
        elif len(e[1])==4:
            site1 = replace_table.get((site_id[0],site_id[1]))
            if site1 is None:
                site1 = next_new_site_id
                next_new_site_id +=1
                replace_table[(site_id[0],site_id[1])]=site1
            site2 = replace_table.get((site_id[2],site_id[3]))
            if site2 is None:
                site2 = next_new_site_id
                next_new_site_id +=1
                replace_table[(site_id[2],site_id[3])]=site2

            for i in range(bit_per_site):
                for j in range(bit_per_site):
                    pos1 = base + site1 * bit_per_site**2 + i * bit_per_site + j 
                    for k in range(bit_per_site):
                        for l in range(bit_per_site):
                            pos2 = base + site2 * bit_per_site**2 + k * bit_per_site + l 
                            coeff = e[0] * e[1][0][1][i] * e[1][1][1][j] * e[1][2][1][k] * e[1][3][1][l]
                            if coeff != 0:
                                ce_bp.append({"c":coeff, "p":[pos1,pos2]})
        elif len(e[1])==3:
            site1 = replace_table.get((site_id[0],site_id[1]))
            if site1 is None:
                site1 = next_new_site_id
                next_new_site_id +=1
                replace_table[(site_id[0],site_id[1])]=site1
            for i in range(bit_per_site):
                for j in range(bit_per_site):
                    pos1 = base + site1 * bit_per_site**2 + i * bit_per_site + j 
                    for k in range(bit_per_site):
                        pos2 = site_id[2] * bit_per_site + k  
                        coeff = e[0] * e[1][0][1][i] * e[1][1][1][j] * e[1][2][1][k]
                        if coeff != 0:
                            ce_bp.append({"c":coeff, "p":[pos1,pos2]})
        elif len(e[1])==2:
            for i in range(bit_per_site):
                pos1 = site_id[0] * bit_per_site + i 
                for j in range(bit_per_site):
                    pos2 = site_id[1] * bit_per_site + j
                    coeff = e[0] * e[1][0][1][i] * e[1][1][1][j]
                    if coeff != 0:
                        ce_bp.append({"c":coeff, "p":[pos1,pos2]})
        elif len(e[1])==1:
            for i in range(bit_per_site):
                coeff = e[0] * e[1][0][1][i]
                pos = site_id[0] * bit_per_site + i
                if coeff != 0:
                    ce_bp.append({"c":coeff, "p":[pos]})        
        elif len(e[1])==0:
            coeff = e[0]
            if coeff != 0:
                ce_bp.append({"c":coeff, "p":[]})
        else:
            assert True, 'The orders of more than fifth are not supported yet'

    # Penalties for variable replacements
    for k,v in replace_table.items():
        for i in range(bit_per_site):
            x1 = k[0] * bit_per_site + i  
            for j in range(bit_per_site):
                x2 = k[1] * bit_per_site + j
                y  = base + v * bit_per_site**2 + i * bit_per_site + j
                rp_bp.append({"c":3, "p":[y]})
                rp_bp.append({"c":1, "p":[x1,x2]})
                rp_bp.append({"c":-2,"p":[y,x1]})
                rp_bp.append({"c":-2,"p":[y,x2]})

    # Penalties for one-hot-constraint
    for s in range(site_num):
        for i in range(bit_per_site):
            for j in range(i,bit_per_site):
                x1 = bit_per_site * s + i
                x2 = bit_per_site * s + j
                if i==j:
                    oh_bp.append({"c":-1, "p":[x1]})
                else:
                    oh_bp.append({"c":2, "p":[x1,x2]})
    oh_bp.append({"c":site_num, "p":[]})
    
    return ce_bp, rp_bp, oh_bp, {'table':replace_table,'next_new_site_id':next_new_site_id}
