import os,sys,string,time,glob,h5py
import illustris_python as il
import numpy as np
import pandas as pd

def Subhalos_Catalogue(basePath,snapnum,mstar_lower=0,mstar_upper=np.inf,
                       little_h=0.6774):
    '''
    mstar_lower and mstar_upper have log10(Mstar/Msun) units and are physical (i.e. Msun, not Msun/h).
    '''
    # convert to units used by TNG (1e10 Msun/h)
    mstar_lower = 10**(mstar_lower)/1e10*little_h
    mstar_upper = 10**(mstar_upper)/1e10*little_h

    ptNumStars = il.snapshot.partTypeNum('stars')
    ptNumGas = il.snapshot.partTypeNum('gas')
    ptNumDM = il.snapshot.partTypeNum('dm')

    hdr = il.groupcat.loadHeader(basePath,snapnum)
    redshift = hdr['Redshift']
    a2 = hdr['Time']
    boxsize = hdr['BoxSize']

    # get subhalo info

    subhalo_fields = [
        'SubhaloFlag',
        'SubhaloGrNr',
        'SubhaloPos',
        'SubhaloBHMass',
        'SubhaloBHMdot',
        'SubhaloMass',
        'SubhaloMassType',
        'SubhaloMassInRadType',
        'SubhaloSFR',
        'SubhaloSFRinHalfRad',
        'SubhaloSFRinRad',
        'SubhaloHalfmassRadType',
        'SubhaloVel',
        'SubhaloVelDisp',
        'SubhaloVmax',
        'SubhaloVmaxRad',
    ]

    subhalos = il.groupcat.loadSubhalos(basePath,snapnum,
                                        fields=subhalo_fields)
    nsubhalos = subhalos['count']
    subfind_ids = np.arange(nsubhalos,dtype='int32')
    subhalos['SubfindID'] = subfind_ids
    subhalos['SnapNum'] = np.ones_like(subfind_ids)*snapnum
    del subhalos['count']
    
    # set mass and cosmology flag cuts
    mstar = subhalos['SubhaloMassType'][:,ptNumStars]
    flags = subhalos['SubhaloFlag']
    subhalo_msk = (mstar>=mstar_lower) & (mstar<mstar_upper) & (flags==1)
    del subhalos['SubhaloFlag']

    # get group info

    group_fields = [
        'GroupFirstSub',
        'GroupMass',
        'GroupMassType',
        'GroupNsubs',
        'GroupCM',
        'GroupPos',
        'Group_M_Crit200',
        'Group_M_Crit500',
        'Group_R_Crit200',
        'Group_R_Crit500',
        'GroupVel',
    ]

    groups = il.groupcat.loadHalos(basePath,snapnum,
                                   fields=group_fields)
    central_msk = groups['GroupFirstSub']>=0
    central_ids = groups['GroupFirstSub'][central_msk]

    # make placeholders for group info in subhalo catalogue 
    for group_field in group_fields:
        dims = groups[group_field].shape
        dtype = groups[group_field].dtype
        if len(dims)>1:
            dims = (nsubhalos,dims[-1])
        else:
            dims = (nsubhalos,)
        subhalos[group_field]=np.zeros(dims).astype(dtype)

    # central/satellite classification field
    subhalos['GroupFirstSub'][central_ids] = 1

    # add other group properties to subhalo catalogue
    for subfind_id in subfind_ids:
        group_id = subhalos['SubhaloGrNr'][subfind_id]
        for group_field in group_fields[1:]:
            subhalos[group_field][subfind_id]=groups[group_field][group_id]

    # fillable arrays
    subhalos['GroupVelDisp'] = np.zeros((nsubhalos,))
    subhalos['GroupVelDispMassWeighted'] = np.zeros((nsubhalos,))
    subhalos['GroupVelDisp_Z'] = np.zeros((nsubhalos,))
    subhalos['GroupVelDispMassWeighted_Z'] = np.zeros((nsubhalos,))

    # compute 3d position/velocity offsets for subhalos in each group
    # also compute x,y position and los velocity wrt z axis
    
    # reduce to groups containing galaxies satisfying subhalo_msk
    group_ids = np.unique(subhalos['SubhaloGrNr'][subhalo_msk])
    
    for group_id in group_ids:
        
        # bulk properties of group using CM
        group_coords = groups['GroupCM'][group_id]
        group_vel = groups['GroupVel'][group_id]

        # Use Pos for Subhalos due to numerical stripping
        grsub_msk = (subhalos['SubhaloGrNr']==group_id)
        rel_coords = subhalos['SubhaloPos'][grsub_msk] - group_coords
        rel_vels = subhalos['SubhaloVel'][grsub_msk] - group_vel/a2

        # apply periodic fix to coords
        rel_coords[rel_coords>boxsize/2] -= boxsize
        rel_coords[rel_coords<-boxsize/2] += boxsize

        subhalos['SubhaloPos'][grsub_msk] = rel_coords
        subhalos['SubhaloVel'][grsub_msk] = rel_vels

        # compute velocity dispersion (+ mass weighted)
        weights = subhalos['SubhaloMassInRadType'][grsub_msk,ptNumStars]

        if np.sum(weights)>0.:
            subhalos['GroupVelDisp'][grsub_msk] = np.sqrt(np.average(
                np.sum(rel_vels**2,axis=1)))/np.sqrt(3)
            subhalos['GroupVelDisp_Z'][grsub_msk] = np.std(
                rel_vels,axis=0)[-1]
            subhalos['GroupVelDispMassWeighted'][grsub_msk] = np.sqrt(
                np.average(np.sum(rel_vels**2,axis=1),
                           weights=weights))/np.sqrt(3)
            subhalos['GroupVelDispMassWeighted_Z'][grsub_msk] = np.sqrt(
                np.average(rel_vels**2,weights=weights,axis=0))[-1]
        else:
            # no subhalos with stars in group
            subhalos['GroupVelDisp'][grsub_msk]=-1.
            subhalos['GroupVelDisp_Z'][grsub_msk]=-1.
            subhalos['GroupVelDispMassWeighted'][grsub_msk]=-1.
            subhalos['GroupVelDispMassWeighted_Z'][grsub_msk]=-1.
            
    # split x,y,z in-group coordinates
    subhalos['SubhaloPosInGroup_X'] = subhalos['SubhaloPos'][:,0]*a2/little_h
    subhalos['SubhaloPosInGroup_Y'] = subhalos['SubhaloPos'][:,1]*a2/little_h
    subhalos['SubhaloPosInGroup_Z'] = subhalos['SubhaloPos'][:,2]*a2/little_h
    del subhalos['SubhaloPos']
    subhalos['SubhaloVelInGroup_X'] = subhalos['SubhaloVel'][:,0]
    subhalos['SubhaloVelInGroup_Y'] = subhalos['SubhaloVel'][:,1]
    subhalos['SubhaloVelInGroup_Z'] = subhalos['SubhaloVel'][:,2]
    del subhalos['SubhaloVel']

    # update other fields to physical units
    
    subhalos['SubhaloBHMass'] = np.log10(subhalos[
        'SubhaloBHMass']*1e10/little_h)
    subhalos['SubhaloBHMdot'] = subhalos[
        'SubhaloBHMdot']*1e10/0.978 # Msun/yr
    
    subhalos['SubhaloMass'] = np.log10(subhalos[
        'SubhaloMass']*1e10/little_h)
    subhalos['SubhaloMassType'] = np.log10(subhalos[
        'SubhaloMassType']*1e10/little_h)
    subhalos['SubhaloMassType_stars'] = subhalos[
        'SubhaloMassType'][:,ptNumStars]
    subhalos['SubhaloMassType_gas'] = subhalos[
        'SubhaloMassType'][:,ptNumGas]
    subhalos['SubhaloMassType_dm'] = subhalos[
        'SubhaloMassType'][:,ptNumDM]
    del subhalos['SubhaloMassType']
    
    subhalos['SubhaloMassInRadType'] = np.log10(subhalos[
        'SubhaloMassInRadType']*1e10/little_h)
    subhalos['SubhaloMassInRadType_stars'] = subhalos[
        'SubhaloMassInRadType'][:,ptNumStars]
    subhalos['SubhaloMassInRadType_gas'] = subhalos[
        'SubhaloMassInRadType'][:,ptNumGas]
    subhalos['SubhaloMassInRadType_dm'] = subhalos[
        'SubhaloMassInRadType'][:,ptNumDM]
    del subhalos['SubhaloMassInRadType']
    
    subhalos['SubhaloHalfmassRadType'] *= a2/little_h # kpc
    subhalos['SubhaloHalfMassRadType_stars']=subhalos[
        'SubhaloHalfmassRadType'][:,ptNumStars]
    subhalos['SubhaloHalfmassRadType_gas']=subhalos[
        'SubhaloHalfmassRadType'][:,ptNumGas]
    subhalos['SubhaloHalfmassRadType_dm']=subhalos[
        'SubhaloHalfmassRadType'][:,ptNumDM]
    del subhalos['SubhaloHalfmassRadType']
    
    subhalos['SubhaloVmaxRad'] *= a2/little_h
    
    subhalos['GroupMass'] = np.log10(subhalos[
        'GroupMass']*1e10/little_h)
    
    subhalos['GroupMassType'] = np.log10(subhalos[
        'GroupMassType']*1e10/little_h)
    subhalos['GroupMassType_stars']=subhalos[
        'GroupMassType'][:,ptNumStars]
    subhalos['GroupMassType_gas']=subhalos[
        'GroupMassType'][:,ptNumGas]
    subhalos['GroupMassType_dm']=subhalos[
        'GroupMassType'][:,ptNumDM]
    del subhalos['GroupMassType']
    
    subhalos['GroupCM'] = subhalos['GroupCM']*a2/little_h
    subhalos['GroupCM_X'] = subhalos['GroupCM'][:,0]
    subhalos['GroupCM_Y'] = subhalos['GroupCM'][:,1]
    subhalos['GroupCM_Z'] = subhalos['GroupCM'][:,2]
    del subhalos['GroupCM']
    
    subhalos['GroupPos'] = subhalos['GroupPos']*a2/little_h
    subhalos['GroupPos_X'] = subhalos['GroupPos'][:,0]
    subhalos['GroupPos_Y'] = subhalos['GroupPos'][:,1]
    subhalos['GroupPos_Z'] = subhalos['GroupPos'][:,2]
    del subhalos['GroupPos']
    
    subhalos['GroupVel']/=a2
    subhalos['GroupVel_X']=subhalos['GroupVel'][:,0]
    subhalos['GroupVel_Y']=subhalos['GroupVel'][:,1] 
    subhalos['GroupVel_Z']=subhalos['GroupVel'][:,2]
    del subhalos['GroupVel']
    
    subhalos['Group_M_Crit200'] = np.log10(subhalos[
        'Group_M_Crit200']*1e10/little_h)
    subhalos['Group_M_Crit500'] = np.log10(subhalos[
        'Group_M_Crit500']*1e10/little_h)
    subhalos['Group_R_Crit200'] = subhalos['Group_R_Crit200']*a2/little_h
    subhalos['Group_R_Crit500'] = subhalos['Group_R_Crit500']*a2/little_h
    
    # make dataframe
    df = pd.DataFrame.from_dict(subhalos)
    df = df.loc[subhalo_msk]
    return df

def Subhalos_SQL(cat,sim,snap,
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
        rec = cat.loc[[idx]]
        values = [str(rec[col].values[0]) for col in columns]
        values = ','.join(values)
        values = values.replace('nan','-99')
        values = values.replace('-inf','-99')
        values = values.replace('inf','-99')
        dbID = f'{rec["SnapNum"].values[0]}_{rec["SubfindID"].values[0]}'
        dbcmd = [
            f'INSERT INTO Subhalos',
            '(dbID,',','.join(columns),')',
            f'VALUES',
            f'("{dbID}",',values,')',
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
    
    from astropy.cosmology import Planck15 as cosmo
    
    sim = os.getenv('SIM')
    snap = int(os.getenv('SNAP'))
    mstar_lower = 9.

    basePath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/{sim}/output'
    catPath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/Scripts/Schema/Catalogues'
    
    if not os.access(catPath,0):
        os.system(f'mkdir -p {catPath}')
        
    catName = f'{catPath}/{sim}_{snap:03}_Subhalos.csv'
    
    if not os.access(catName,0):
        cat = Subhalos_Catalogue(basePath,snapnum=snap,
                                 mstar_lower=mstar_lower,
                                 little_h = cosmo.H0.value/100)
        cat.to_csv(catName,index=False)
    else:
        cat = pd.read_csv(catName)
        
    Subhalos_SQL(cat,sim,snap)

if __name__=='__main__':
    main()
