def residuehydrophobicity(rname):
    if(rname == 'ARG'):
        return -4.5
    if(rname == 'LYS'):
        return -3.9
    if(rname == 'ASN'):
        return -3.5
    if(rname == 'ASP'):
        return -3.5
    if(rname == 'GLN'):
        return -3.5
    if(rname == 'GLU'):
        return -3.5
    if(rname == 'HIS'):
        return -3.2 
    if(rname == 'PRO'):
        return -1.6
    if(rname == 'TYR'):
        return -1.3
    if(rname == 'TRP'):
        return -0.9
    if(rname == 'SER'):
        return -0.8
    if(rname == 'THR'):
        return -0.7
    if(rname == 'GLY'):
        return -0.4
    if(rname == 'ALA'):
        return 1.8
    if(rname == 'ILE'):
        return 4.5
    if(rname == 'MET'):
        return 1.9
    if(rname == 'CYS'):
        return 2.5
    if(rname == 'PHE'):
        return 2.8
    if(rname == 'LEU'):
        return 3.8
    if(rname == 'VAL'):
        return 4.2
    return 0.0
