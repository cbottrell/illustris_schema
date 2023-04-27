import os,sys,string,time,glob,h5py
import illustris_python as il
import numpy as np
import pandas as pd

def Mergers_Catalogue(basePath,sim,snap,little_h=0.6774):
    
    massmax = pd.read_csv(
        f'{basePath}/MassMax/{sim}_MassMax_{snap:03}.csv'
    )
    mergers = np.load(
        f'{basePath}/Mergers/{sim}_MergersInf_{snap:03}.npz',
        allow_pickle=True
    )
    # all files have same keys 
    subs = [int(key) for key in mergers['major'][()].keys()]
    
    for type_merger in ['major','minor','tiny']:
        history = mergers[type_merger][()]
        
    cat = pd.DataFrame()
    
    for sub in subs:
        dbID = f'{snap}_{sub}'
        mass_idx = massmax.SubhaloID==sub
        mass_rec = massmax.loc[mass_idx]
        
        # append massmax catalogue fields
        r = {
            'dbID': f'{snap}_{sub}',
            'SnapNum': snap,
            'SubfindID': sub,
            'MaxMassSnap_stars': mass_rec[
                'SnapMaxSubhaloMassType_stars'
            ].values[0],
            'MaxMassType_stars': mass_rec[
                'MaxSubhaloMassType_stars'
            ].values[0],
            'MaxMassInRadSnap_stars': mass_rec[
                'SnapMaxSubhaloMassInRadType_stars'
            ].values[0],
            'MaxMassInRadType_stars': mass_rec[
                'MaxSubhaloMassInRadType_stars'
            ].values[0],
        }
        
        # append merger catalogue fields
        for i in ['Major','Minor','Tiny']:
            h = mergers[i.lower()][()][f'{sub}']
            for key in h.keys():
                if type(h[key])==list:
                    if len(h[key])>0:
                        r[f'{i}_{key}'] = h[key][0]
                    else:
                        r[f'{i}_{key}'] = -1
                else:
                    r[f'{i}_{key}'] = h[key]

        cat = cat.append(r,ignore_index=True)
        
    cat = cat.astype({'SnapNum': int, 'SubfindID': int})
        
    return cat

def Mergers_SQL(cat,sim,snap,
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
        values = values.replace('nan','-2')
        values = values.replace('-inf','-2')
        values = values.replace('inf','-2')
        dbID = f'{rec["SnapNum"]}_{rec["SubfindID"]}'
        values = values.replace(f'{dbID}',f'"{dbID}"')
        dbcmd = [
            f'INSERT INTO MergersInf',
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
    
    from astropy.cosmology import Planck15 as cosmo
    
    sim,snap=sys.argv[1],int(sys.argv[2])
    
    mergerCatPath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/Scripts/Mergers/illustris_mergers/Catalogues'
    
    schemaCatPath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/Scripts/Schema/Catalogues'
    
    if not os.access(schemaCatPath,0):
        os.system(f'mkdir -p {schemaCatPath}')
        
    catName = f'{schemaCatPath}/{sim}_{snap:03}_MergersInf.csv'
    
    if not os.access(catName,0):
        cat = Mergers_Catalogue(mergerCatPath,sim,snap,
                                little_h = cosmo.H0.value/100)
        
        cat.to_csv(catName,index=False)
    else:
        cat = pd.read_csv(catName)

    Mergers_SQL(cat,sim,snap)

if __name__=='__main__':
    main()