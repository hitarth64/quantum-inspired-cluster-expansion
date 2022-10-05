from collections import Counter
from ase.io import read
import pandas as pd
import os

def find_mixing_energy(st, energy, references):
    """
    Args:
        st: ASE atoms object
        energy: total energy of the structure st (in eV)
        references: references to consider for calculation of mixing energy
    returns:
        mixing energy: Units[eV/atom]
    """
    dict_representation = dict(Counter(st.get_chemical_symbols()))
    NumAtoms = len(st)
    return ( energy - sum([v/NumAtoms*references[k] for k,v in dict_representation.items()]) )/ NumAtoms

module_dir = os.path.dirname(os.path.abspath(__file__))

class Dataset:
    """
    Dataset class for interfacing QCE
    3 datasets are provided:
    * d1: CuNiPdAg alloys
    * d2: [Hf-V-Y-Ce-Fe-Co-Ru-Zr]O2 oxide alloys
    * d3: [W-Cr-Mn-Ti-Co-Sb-V-Ru]O2 oxide alloys
    """
    def __init__(self, AtomPropertyKeyValuePairs=None):
        """
        source: either d1 OR d2 OR d3
        Read the description above
        Please use 'd4' to use custom data
        """
        
    def GetInfo(self):
        return {'source="d1"': 'CuNiPdAg', 'source="d2"': '[Hf-V-Y-Ce-Fe-Co-Ru-Zr]O2', 'source="d3"': '[W-Cr-Mn-Ti-Co-Sb-V-Ru]O2'}
		
    def GetData(self, source, AtomPropertyKeyValuePairs=None):
        """
        source: either d1 OR d2 OR d3
        Read the description above
        Please use 'd4' to use custom data
        """
        
        self.AtomPropertyKeyValuePairs = AtomPropertyKeyValuePairs
        
        if source == 'd1':
            return self.QuaternaryDataset()
        elif source == 'd2':
            return self.ZrDataset()
        elif source == 'd3':
            return self.RuDataset()
        elif source == 'd5':
            return self.RuOldDataset()
        elif source == 'd4':
            assert self.AtomPropertyKeyValuePairs is not None, "Provide Atom's property-key value pairs"
            return self.CustomDataset()
            
    def QuaternaryDataset(self):
        """
        Returns pandas dataframe with data on CuNiPdAg quaternary alloy system
        """
        references = {'Cu':-240.11121086, 'Ni':-352.62417749, 'Pd':-333.69496589, 'Ag':-173.55506507}
        db = read(module_dir+'/dataset/structures_CuNiPdAg.json',':')
        dft_data = pd.read_csv(module_dir+'/dataset/properties_CuNiPdAg.csv',index_col=0)
        indices = dft_data.index.values
        mixing_energies = []
        DataFrame = pd.DataFrame(columns=['atoms','MixingEnergy','TotalEnergy'])
        
        for idx,i in enumerate(indices):
            mixEnergy = find_mixing_energy(db[i], dft_data.loc[i,'final_energy'], references)
            DataFrame.loc[idx] = [db[i], mixEnergy, dft_data.loc[i,'final_energy']]
        
        return DataFrame
    
    def ZrDataset(self):
        """
        Returns pandas dataframe with data on [Hf-V-Y-Ce-Fe-Co-Ru-Zr]O2 oxide alloys
        """
        db = read(module_dir+'/dataset/structures_Zr.json',':')
        dft_data = pd.read_csv(module_dir+'/dataset/properties_Zr.csv', index_col=0)
        indices = dft_data.index.values
        DataFrame = pd.DataFrame(columns=['atoms','MixingEnergy','pup','pdown','dup','ddown'])
        
        for idx,i in enumerate(indices):
            DataFrame.loc[idx] = [db[i], dft_data.loc[i,'mixing'], dft_data.loc[i,'pup'], dft_data.loc[i,'pdown'], \
                                 dft_data.loc[i,'dup'], dft_data.loc[i,'ddown']]
        return DataFrame
    
    def RuDataset(self):
        """
        Returns pandas dataframe with data on [W-Cr-Mn-Ti-Co-Sb-V-Ru]O2 oxide alloys
        """
        db = read(module_dir+'/dataset/structures_Ru.json',':')
        dft_data = pd.read_csv(module_dir+'/dataset/properties_Ru.csv', index_col=0)
        indices = dft_data.index.values
        DataFrame = pd.DataFrame(columns=['atoms','MixingEnergy', 'pband','dband','O-p'])
        
        for idx,i in enumerate(indices):
            DataFrame.loc[idx] = [db[i], dft_data.loc[i,'mixing_energy'], \
                                  dft_data.loc[i,'pband'], dft_data.loc[i,'dband'], dft_data.loc[i,'O-p']]
        return DataFrame
    
    def RuOldDataset(self):
        """
        Returns pandas dataframe with data on [W-Cr-Mn-Ti-Co-Sb-V-Ru]O2 oxide alloys
        """
        db = read(module_dir+'/dataset/structures_Ru.json',':')
        dft_data = pd.read_csv(module_dir+'/dataset/properties_Ru_old.csv', index_col=0)
        indices = dft_data.index.values
        DataFrame = pd.DataFrame(columns=['atoms','MixingEnergy', 'TotalEnergy', 'pband','dband'])
        
        for idx,i in enumerate(indices):
            DataFrame.loc[idx] = [db[i], dft_data.loc[i,'MixingEnergy'], dft_data.loc[i,'TotalEnergy'], \
                                  dft_data.loc[i,'pup'], dft_data.loc[i,'dup']]
        return DataFrame
        
    def CustomDataset(self):
        """
        AtomPropertyKeyValuePairs is a dictionary of dictionary
        Returns pandas dataframe using the provided custom data
        """
        DataFrame = pd.DataFrame(columns=['atoms'])
        for idx,k in enumerate(self.AtomPropertyKeyValuePairs.keys()):
            for Property, PropertyVal in self.AtomPropertyKeyValuePairs[k].items():
                DataFrame.loc[idx,Property] = PropertyVal
        return DataFrame
