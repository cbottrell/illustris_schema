import os,sys,time,h5py
import illustris_python as il
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15 as cosmo

def query(dbcmd,database,
          host='nantai.ipmu.jp',user='bottrell',
          cnf_path='/home/connor.bottrell/.mysql/nantai.cnf'):
    '''Query database and return results as numpy array.'''
    import pymysql
    import numpy as np
    db = pymysql.connect(host=host,
                         user=user,
                         database=database,
                         read_default_file=cnf_path)
    c = db.cursor()
    c.execute(dbcmd)
    data = np.asarray(c.fetchall())
    c.close()
    db.close()
    return data

def submit(dbcmd,database,
           host='nantai.ipmu.jp',user='bottrell',
           cnf_path='/home/connor.bottrell/.mysql/nantai.cnf'):
    '''Submit command to database.'''
    import pymysql
    import numpy as np
    db = pymysql.connect(host=host,
                         user=user,
                         database=database,
                         read_default_file=cnf_path,
                         autocommit=True
                        )
    c = db.cursor()
    c.execute(dbcmd)
    c.close()
    db.close()
    return 

def Environment_Catalogue(basePath,massCatPath,sim,snap):
    '''
    mstar_lower and mstar_upper have log10(Mstar/Msun) units and are physical (i.e. Msun, not Msun/h). Requires that maximum masses have already been computed for every galaxy and are stored in the catalogues. These catalogues include masses for all subhalos with a stellar mass component.
    '''
    
    df = pd.DataFrame()
    
    little_h = cosmo.H0.value/100.
    ptNumStars = il.snapshot.partTypeNum('stars')
    ptNumGas = il.snapshot.partTypeNum('gas')
    ptNumDM = il.snapshot.partTypeNum('dm')
    
    # get list of halos to measure environments
    db_sim = sim.replace('-','_')
    database = f'Illustris{db_sim}'
    dbcmd = f'SELECT SubfindID FROM Subhalos WHERE SnapNum={snap}'
    db_subs = np.sort(query(dbcmd,database).flatten())
    
    # get full list of subhalos from groupcats
    fields = ['SubhaloPos','SubhaloMassType','SubhaloFlag']
    cat = il.groupcat.loadSubhalos(basePath,snap,fields=fields)
    hdr = il.groupcat.loadHeader(basePath,snap)
    redshift,a2 = hdr['Redshift'],hdr['Time']
    boxSize = hdr['BoxSize']
    # periodic coordinates
    cat_coords = cat['SubhaloPos']
    
    # get maximum masses from catalogue
    massmax = pd.read_csv(
        f'{massCatPath}/MassMax/{sim}_MassMax_{snap:03}.csv'
    )
    
    # keep only subs in massmax catalogue
    # (i.e. flagged as cosmological and with stellar component)
    cat_subs = massmax['SubhaloID']
    cat_coords = cat_coords[cat_subs]
    cat_maxmass = massmax['MaxSubhaloMassType_stars'].values
    
    for sub in db_subs:
        dbid = f'{snap}_{sub}'
        mass_idx = massmax.SubhaloID==sub
        mass_rec = massmax.loc[mass_idx]
        sub_coords = cat_coords[cat_subs==sub]
        sub_maxmass = mass_rec['MaxSubhaloMassType_stars'].values[0]
        
        # initialize record
        r = {
            'dbID': dbid,
            'SnapNum': snap,
            'SubfindID': sub,
            'MaxMassType_stars': np.log10(sub_maxmass*1e10/little_h),
        }
        
        # periodic coordinate fix relative to subhalo
        rel_coords = cat_coords-sub_coords
        rel_coords[rel_coords>boxSize/2] -= boxSize
        rel_coords[rel_coords<-boxSize/2] += boxSize
        
        # compute mass ratios for every halo 
        mu = cat_maxmass/sub_maxmass
        
        # compute 3d radial distances for every halo (physical kpc)
        rad_coords = np.sqrt(np.sum(rel_coords**2,axis=1))*a2/little_h
        
        for i,mu_min in zip(['Tiny','Minor','Major'],[0.01,0.10,0.25]):
                
            # find matches in mass ratio and 3D distance
            idxs = mu>mu_min
            idxs *= rad_coords<2e3
            # omit target halo (only flagged in major)
            idxs[cat_subs==sub]=0
            # numbers within sphere of radius
            r[f'{i}_Nin2Mpc']   = sum(idxs)
            r[f'{i}_Nin1Mpc']   = sum(idxs*(rad_coords<1e3))
            r[f'{i}_Nin500kpc'] = sum(idxs*(rad_coords<5e2))
            r[f'{i}_Nin100kpc'] = sum(idxs*(rad_coords<1e2))
            r[f'{i}_Nin50kpc']  = sum(idxs*(rad_coords<5e1))
            r[f'{i}_MassIn2Mpc_stars']   = np.log10((sub_maxmass + np.sum(cat_maxmass[idxs]))*1e10/little_h)
            r[f'{i}_MassIn1Mpc_stars']   = np.log10((sub_maxmass + np.sum(cat_maxmass[idxs*(rad_coords<1e3)]))*1e10/little_h)
            r[f'{i}_MassIn500kpc_stars'] = np.log10((sub_maxmass + np.sum(cat_maxmass[idxs*(rad_coords<5e2)]))*1e10/little_h)
            r[f'{i}_MassIn100kpc_stars'] = np.log10((sub_maxmass + np.sum(cat_maxmass[idxs*(rad_coords<1e2)]))*1e10/little_h)
            r[f'{i}_MassIn50kpc_stars']  = np.log10((sub_maxmass + np.sum(cat_maxmass[idxs*(rad_coords<5e1)]))*1e10/little_h)
            # nearest neighbour distances and masses
            sort_rad = np.argsort(rad_coords[idxs])
            rad_coords_sorted = np.array(rad_coords[idxs][sort_rad])
            cat_maxmass_sorted = cat_maxmass[idxs][sort_rad]
            for nn in range(5):
                if len(rad_coords_sorted)>=nn+1:
                    r[f'{i}_R{nn+1}'] = rad_coords_sorted[nn]
                    r[f'{i}_M{nn+1}_stars'] = np.log10(
                        cat_maxmass_sorted[nn]*1e10/little_h)
                else:
                    r[f'{i}_R{nn+1}'] = -1
                    r[f'{i}_M{nn+1}_stars'] = -1
                    
        df = df.append(r,ignore_index=True)
        
    df = df.astype({'SnapNum': int,'SubfindID': int})
            
    return df

def Environment_SQL(cat,sim,snap,
                 host='nantai.ipmu.jp',user='bottrell',
                 cnf_path='/home/connor.bottrell/.mysql/nantai.cnf'):
    import pymysql
    database = f'Illustris{sim}'.replace('-','_')
    columns = list(cat.columns)
    db = pymysql.connect(host=host,
                         user=user,
                         database=database,
                         read_default_file=cnf_path,
                         autocommit=True
                        )
    c = db.cursor()
    for idx in cat.index:
        rec = cat.loc[idx]
        values = [str(rec[col]) for col in columns]
        values = ','.join(values)
        values = values.replace('nan','-99')
        values = values.replace('-inf','-99')
        values = values.replace('inf','-99')
        dbID = f'{rec["SnapNum"]}_{rec["SubfindID"]}'
        values = values.replace(f'{dbID}',f'"{dbID}"')
        dbcmd = [
            f'INSERT INTO Environments',
            '(',','.join(columns),')',
            f'VALUES',
            f'(',values,')',
        ]
        dbcmd = ' '.join(dbcmd)
        try:
            c.execute(dbcmd)
        except:
            continue
    c.close()
    db.close()
    return
    
def main():
    
    sim,snap=sys.argv[1],int(sys.argv[2])
    
    # group cat path
    basePath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/{sim}/output'
    # output schema path
    schemaCatPath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/Scripts/Schema/Catalogues'
    # catalogue of maximum masses
    massCatPath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/Scripts/Mergers/illustris_mergers/Catalogues'
    
    if not os.access(schemaCatPath,0):
        os.system(f'mkdir -p {schemaCatPath}')
        
    catName = f'{schemaCatPath}/{sim}_{snap:03}_Environments.csv'
    
    if not os.access(catName,0):
        cat = Environment_Catalogue(
            basePath, massCatPath, sim=sim, snap=snap 
        )
        cat.to_csv(catName,index=False)
    else:
        cat = pd.read_csv(catName)
        
    Environment_SQL(cat,sim,snap)

if __name__=='__main__':
    
    main()