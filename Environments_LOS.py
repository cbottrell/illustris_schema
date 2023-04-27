import os,sys,time,h5py
import illustris_python as il
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15 as cosmo
import matplotlib.pyplot as plt

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

def Rotate(x,axis):
    '''return x-vector in instrument frame'''
    Xinst, Yinst, Zinst = _get_inst(axis)
    rx = np.zeros_like(x)
    rx[:,0] = np.dot(x,Xinst)
    rx[:,1] = np.dot(x,Yinst)
    rx[:,2] = np.dot(x,Zinst)
    return rx

def _get_angles(axis):
    '''return instrument coordinates in spherical coordinates'''
    angles = {'v0': (109.5,   0.,  0.),
              'v1': (109.5, 120.,  0.),
              'v2': (109.5,-120.,  0.),
              'v3': (   0.,   0.,  0.)}
    if len(axis) == 2:
        inclination, azimuth, posang = angles[axis]
    elif len(axis) == 5:
        inclination = float(axis[1:3])
        azimuth     = 0.
        posang      = float(axis[3: ])
    return inclination,azimuth,posang

def _get_inst(axis):
    '''return instrument coordinates in box coordinates '''
    inclination,azimuth,posang = _get_angles(axis)
    costheta= np.cos(np.radians(inclination))
    sintheta= np.sin(np.radians(inclination))
    cosphi  = np.cos(np.radians(azimuth))
    sinphi  = np.sin(np.radians(azimuth))
    cospa   = np.cos(np.radians(posang))
    sinpa   = np.sin(np.radians(posang)) 
    
    Xinst   = np.array([-cosphi*costheta*cospa - sinphi*sinpa,
                        -sinphi*costheta*cospa + cosphi*sinpa,
                         sintheta*cospa])
    Yinst   = np.array([ cosphi*costheta*sinpa - sinphi*cospa,
                         sinphi*costheta*sinpa + cosphi*cospa,
                        -sintheta*sinpa]) 
    Zinst   = np.array([sintheta * cosphi, 
                        sintheta * sinphi, 
                        costheta])

    Xinst /= np.sqrt(np.sum(Xinst**2))
    Yinst /= np.sqrt(np.sum(Yinst**2))
    Zinst /= np.sqrt(np.sum(Zinst**2))
    return Xinst,Yinst,Zinst

def Environment_Catalogue(basePath,massCatPath,sim,snap,axis,delv):
    '''
    mstar_lower and mstar_upper have log10(Mstar/Msun) units and are physical (i.e. Msun, not Msun/h). Requires that maximum masses have already been computed for every galaxy and are stored in the catalogues. These catalogues include masses for all subhalos with a stellar mass component.
    '''
    
    df = pd.DataFrame()
    
    little_h = cosmo.h
    ptNumStars = il.snapshot.partTypeNum('stars')
    ptNumGas = il.snapshot.partTypeNum('gas')
    ptNumDM = il.snapshot.partTypeNum('dm')
    
    # get list of halos to measure environments
    db_sim = sim.replace('-','_')
    database = f'Illustris{db_sim}'
    dbcmd = f'SELECT SubfindID FROM Subhalos WHERE SnapNum={snap}'
    db_subs = np.sort(query(dbcmd,database).flatten())
    db_subs = db_subs
    
    # get full list of subhalos from groupcats
    fields = ['SubhaloPos','SubhaloVel','SubhaloMassType','SubhaloFlag']
    cat = il.groupcat.loadSubhalos(basePath,snap,fields=fields)
    hdr = il.groupcat.loadHeader(basePath,snap)
    redshift,a2 = hdr['Redshift'],hdr['Time']
    Hz = cosmo.H(redshift).value/little_h # km/s/(Mpc/h)

    # kpc/h proxy for spectroscopic velocity cut (cosmological)
    delz_lim = delv/Hz*1000 # kpc/h
    
    boxSize = hdr['BoxSize']
    # periodic coordinates
    cat_coords = cat['SubhaloPos']
    cat_vels = cat['SubhaloVel']
    
    # get maximum masses from catalogue
    massmax = pd.read_csv(
        f'{massCatPath}/MassMax/{sim}_MassMaxInf_{snap:03}.csv'
    )
    
    # keep only subs in massmax catalogue
    # (i.e. flagged as cosmological and with stellar component)
    cat_subs = massmax['SubhaloID']
    cat_coords = cat_coords[cat_subs]
    cat_vels = cat_vels[cat_subs]
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
        # coordinate change to instrument LOS and convert to kpc/h
        rel_coords = Rotate(rel_coords,axis)*a2
        rel_vels = Rotate(cat_vels,axis)
        sub_vels = rel_vels[cat_subs==sub]
        rel_vels = (rel_vels-sub_vels)
        
        # compute mass ratios for every halo 
        mu = cat_maxmass/sub_maxmass
        
        # fig,ax = plt.subplots(figsize=(10,10))
        # ax.scatter(rel_coords[:,0],rel_coords[:,-1],s=0.05)
        # ax.set_xlim(-17500,17500)
        # ax.set_ylim(-17500,17500)
        
        # compute 2d projected distance to every subhalo (physical kpc/h)
        proj_coords = np.sqrt(np.sum(rel_coords[:,:-1]**2,axis=1))
        # los distances and velocity to every galaxy in new coords
        delz_coords = rel_coords[:,-1]
        los_vels = rel_vels[:,-1]
        print(np.median(np.abs(los_vels)))
        
        for i,mu_min in zip(['Tiny','Minor','Major'],[0.01,0.10,0.25]):
                
            # find matches in mass ratio and cylinder
            idxs = mu>mu_min
            idxs = idxs & (proj_coords<8e3)
            # effective cosmological velocity cut
            idxs = idxs & (np.abs(delz_coords)<delz_lim)
            # peculiar velocity cut
            idxs = idxs & (np.abs(los_vels)<delv)
            # omit target halo (only flagged in major)
            idxs = idxs & (cat_subs!=sub)
            # numbers within sphere of radius
            r[f'{i}_Nin8Mpc']   = sum(idxs)
            r[f'{i}_Nin4Mpc']   = sum(idxs & (proj_coords<4e3))
            r[f'{i}_Nin2Mpc']   = sum(idxs & (proj_coords<2e3))
            r[f'{i}_Nin1Mpc']   = sum(idxs & (proj_coords<1e3))
            r[f'{i}_Nin500kpc'] = sum(idxs & (proj_coords<5e2))
            r[f'{i}_Nin100kpc'] = sum(idxs & (proj_coords<1e2))
            r[f'{i}_Nin50kpc']  = sum(idxs & (proj_coords<5e1))
            r[f'{i}_MassIn8Mpc_stars']   = np.log10((sub_maxmass + np.sum(cat_maxmass[idxs]))*1e10/little_h)
            r[f'{i}_MassIn4Mpc_stars']   = np.log10((sub_maxmass + np.sum(cat_maxmass[idxs & (proj_coords<4e3)]))*1e10/little_h)
            r[f'{i}_MassIn2Mpc_stars']   = np.log10((sub_maxmass + np.sum(cat_maxmass[idxs & (proj_coords<2e3)]))*1e10/little_h)
            r[f'{i}_MassIn1Mpc_stars']   = np.log10((sub_maxmass + np.sum(cat_maxmass[idxs & (proj_coords<1e3)]))*1e10/little_h)
            r[f'{i}_MassIn500kpc_stars'] = np.log10((sub_maxmass + np.sum(cat_maxmass[idxs & (proj_coords<5e2)]))*1e10/little_h)
            r[f'{i}_MassIn100kpc_stars'] = np.log10((sub_maxmass + np.sum(cat_maxmass[idxs & (proj_coords<1e2)]))*1e10/little_h)
            r[f'{i}_MassIn50kpc_stars']  = np.log10((sub_maxmass + np.sum(cat_maxmass[idxs & (proj_coords<5e1)]))*1e10/little_h)
            # nearest neighbour distances and masses
            sort_proj = np.argsort(proj_coords[idxs])
            proj_coords_sorted = np.array(proj_coords[idxs][sort_proj])
            cat_maxmass_sorted = cat_maxmass[idxs][sort_proj]
            for nn in range(5):
                if len(proj_coords_sorted)>=nn+1:
                    r[f'{i}_R{nn+1}'] = proj_coords_sorted[nn]
                    r[f'{i}_M{nn+1}_stars'] = np.log10(
                        cat_maxmass_sorted[nn]*1e10/little_h)
                else:
                    r[f'{i}_R{nn+1}'] = -1
                    r[f'{i}_M{nn+1}_stars'] = -1
                    
        for min_mass in [8,9,10]:
            idxs = cat_maxmass>(10**min_mass/1e10*little_h)
            idxs = idxs & (proj_coords<8e3)
            # effective cosmological velocity cut
            idxs = idxs & (np.abs(delz_coords)<delz_lim)
            # peculiar velocity cut
            idxs = idxs & (np.abs(los_vels)<delv)
            # omit target halo (only flagged in major)
            idxs = idxs & (cat_subs!=sub)
            # ax.scatter(rel_coords[idxs,0],rel_coords[idxs,-1],s=5)
            
            # numbers within sphere of radius
            r[f'MstarAbove{min_mass}_Nin8Mpc'] = sum(idxs)
            r[f'MstarAbove{min_mass}_Nin4Mpc'] = sum(idxs & (proj_coords<4e3))
            r[f'MstarAbove{min_mass}_Nin2Mpc'] = sum(idxs & (proj_coords<2e3))
            r[f'MstarAbove{min_mass}_Nin1Mpc'] = sum(idxs & (proj_coords<1e3))
            r[f'MstarAbove{min_mass}_Nin500kpc']= sum(idxs & (proj_coords<5e2))
            r[f'MstarAbove{min_mass}_Nin100kpc']= sum(idxs & (proj_coords<1e2))
            r[f'MstarAbove{min_mass}_Nin50kpc'] = sum(idxs & (proj_coords<5e1))
            r[f'MstarAbove{min_mass}_MassIn8Mpc_stars']= np.log10((sub_maxmass + np.sum(cat_maxmass[idxs]))*1e10/little_h)
            r[f'MstarAbove{min_mass}_MassIn4Mpc_stars']= np.log10((sub_maxmass + np.sum(cat_maxmass[idxs & (proj_coords<4e3)]))*1e10/little_h)
            r[f'MstarAbove{min_mass}_MassIn2Mpc_stars']= np.log10((sub_maxmass + np.sum(cat_maxmass[idxs & (proj_coords<2e3)]))*1e10/little_h)
            r[f'MstarAbove{min_mass}_MassIn1Mpc_stars']= np.log10((sub_maxmass + np.sum(cat_maxmass[idxs & (proj_coords<1e3)]))*1e10/little_h)
            r[f'MstarAbove{min_mass}_MassIn500kpc_stars']= np.log10((sub_maxmass + np.sum(cat_maxmass[idxs & (proj_coords<5e2)]))*1e10/little_h)
            r[f'MstarAbove{min_mass}_MassIn100kpc_stars']= np.log10((sub_maxmass + np.sum(cat_maxmass[idxs & (proj_coords<1e2)]))*1e10/little_h)
            r[f'MstarAbove{min_mass}_MassIn50kpc_stars']= np.log10((sub_maxmass + np.sum(cat_maxmass[idxs & (proj_coords<5e1)]))*1e10/little_h)
            # nearest neighbour distances and masses
            sort_proj = np.argsort(proj_coords[idxs])
            proj_coords_sorted = np.array(proj_coords[idxs][sort_proj])
            cat_maxmass_sorted = cat_maxmass[idxs][sort_proj]
            for nn in range(5):
                if len(proj_coords_sorted)>=nn+1:
                    r[f'MstarAbove{min_mass}_R{nn+1}'] = proj_coords_sorted[nn]
                    r[f'MstarAbove{min_mass}_M{nn+1}_stars'] = np.log10(
                        cat_maxmass_sorted[nn]*1e10/little_h)
                else:
                    r[f'MstarAbove{min_mass}_R{nn+1}'] = -1
                    r[f'MstarAbove{min_mass}_M{nn+1}_stars'] = -1

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
            f'INSERT INTO Environments_LOS',
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
    
    axis = 'v3' # z-axis
    delv = 1000 # km/s
    
    # group cat path
    basePath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/{sim}/output'
    # output schema path
    schemaCatPath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/Scripts/Schema/Catalogues'
    # catalogue of maximum masses
    massCatPath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/Scripts/Mergers/illustris_mergers/Catalogues'
    
    if not os.access(schemaCatPath,0):
        os.system(f'mkdir -p {schemaCatPath}')
        
    catName = f'{schemaCatPath}/{sim}_{snap:03}_Environments_LOS.csv'
    
    if not os.access(catName,0):
        cat = Environment_Catalogue(
            basePath, massCatPath, sim=sim, snap=snap, 
            axis=axis, delv=delv,
        )
        cat.to_csv(catName,index=False)
    else:
        cat = pd.read_csv(catName)
        
    Environment_SQL(cat,sim,snap)

if __name__=='__main__':
    
    main()
