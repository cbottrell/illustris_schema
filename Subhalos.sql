CREATE TABLE IF NOT EXISTS Subhalos(

    dbID VARCHAR(255),
    SnapNum INT(16),
    SubfindID INT(32),
    
    SubhaloMassType_stars FLOAT,
    SubhaloMassType_gas FLOAT,
    SubhaloMassType_dm FLOAT,
    
    SubhaloMassInRadType_stars FLOAT,
    SubhaloMassInRadType_gas FLOAT,
    SubhaloMassInRadType_dm FLOAT,
    
    SubhaloMassInHalfRadType_stars FLOAT,
    SubhaloMassInHalfRadType_gas FLOAT,
    SubhaloMassInHalfRadType_dm FLOAT,

    SubhaloHalfMassRadType_stars FLOAT,
    SubhaloHalfMassRadType_gas FLOAT,
    SubhaloHalfMassRadType_dm FLOAT,
    
    SubhaloSFR FLOAT,
    SubhaloSFRinRad FLOAT,
    SubhaloSFRinHalfRad FLOAT,

    PRIMARY KEY (dbID)
);
