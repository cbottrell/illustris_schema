CREATE TABLE IF NOT EXISTS Mergers(

    dbID VARCHAR(255),
    SnapNum INT(16),
    SubfindID INT(32),
    
    MaxMassSnap_stars INT(16),
    MaxMassType_stars FLOAT,
    MaxMassInRadSnap_stars INT(16),
    MaxMassInRadType_stars FLOAT,
    
    Major_CountSinceMainLeafProgenitor INT(32),
    Major_CountSinceHalfScaleFactor INT(32),
    Major_CountSince250Myr INT(32),
    Major_CountSince500Myr INT(32),
    Major_CountSince1Gyr INT(32),
    Major_CountSince2Gyr INT(32),
    Major_CountSince3Gyr INT(32),
    Major_FreqencySinceMainLeafProgenitor FLOAT,
    Major_PastSnapNum INT(16),
    Major_PastSubfindID INT(32),
    Major_PastMassRatio FLOAT,
    Major_TimeSinceMerger FLOAT,
    Major_PastMainProgenitorSnapNum INT(16),
    Major_PastMainProgenitorSubfindID INT(32),
    Major_PastMainProgenitorMaxMass_stars FLOAT,
    Major_PastMainProgenitorMaxMass_gas FLOAT,
    Major_PastMainProgenitorMaxMassInRad_stars FLOAT,
    Major_PastMainProgenitorMaxMassInRad_gas FLOAT,
    Major_PastNextProgenitorSnapNum INT(16),
    Major_PastNextProgenitorSubfindID INT(32),
    Major_PastNextProgenitorMaxMass_stars FLOAT,
    Major_PastNextProgenitorMaxMass_gas FLOAT,
    Major_PastNextProgenitorMaxMassInRad_stars FLOAT,
    Major_PastNextProgenitorMaxMassInRad_gas FLOAT,
    Major_CountUntilRootDescendent INT(32),
    Major_CountUntil250Myr INT(32),
    Major_CountUntil500Myr INT(32),
    Major_CountUntil1Gyr INT(32),
    Major_CountUntil2Gyr INT(32),
    Major_CountUntil3Gyr INT(32),
    Major_FutureSnapNum INT(16),
    Major_FutureSubfindID INT(32),
    Major_FutureMassRatio FLOAT,
    Major_TimeUntilMerger FLOAT,
    Major_FutureMainProgenitorSnapNum INT(16),
    Major_FutureMainProgenitorSubfindID INT(32),
    Major_FutureMainProgenitorMaxMass_stars FLOAT,
    Major_FutureMainProgenitorMaxMass_gas FLOAT,
    Major_FutureMainProgenitorMaxMassInRad_stars FLOAT,
    Major_FutureMainProgenitorMaxMassInRad_gas FLOAT,
    Major_FutureNextProgenitorSnapNum INT(16),
    Major_FutureNextProgenitorSubfindID INT(32),
    Major_FutureNextProgenitorMaxMass_stars FLOAT,
    Major_FutureNextProgenitorMaxMass_gas FLOAT,
    Major_FutureNextProgenitorMaxMassInRad_stars FLOAT,
    Major_FutureNextProgenitorMaxMassInRad_gas FLOAT,
    Major_TreeFlag INT(16),
    
    Minor_CountSinceMainLeafProgenitor INT(32),
    Minor_CountSinceHalfScaleFactor INT(32),
    Minor_CountSince250Myr INT(32),
    Minor_CountSince500Myr INT(32),
    Minor_CountSince1Gyr INT(32),
    Minor_CountSince2Gyr INT(32),
    Minor_CountSince3Gyr INT(32),
    Minor_FreqencySinceMainLeafProgenitor FLOAT,
    Minor_PastSnapNum INT(16),
    Minor_PastSubfindID INT(32),
    Minor_PastMassRatio FLOAT,
    Minor_TimeSinceMerger FLOAT,
    Minor_PastMainProgenitorSnapNum INT(16),
    Minor_PastMainProgenitorSubfindID INT(32),
    Minor_PastMainProgenitorMaxMass_stars FLOAT,
    Minor_PastMainProgenitorMaxMass_gas FLOAT,
    Minor_PastMainProgenitorMaxMassInRad_stars FLOAT,
    Minor_PastMainProgenitorMaxMassInRad_gas FLOAT,
    Minor_PastNextProgenitorSnapNum INT(16),
    Minor_PastNextProgenitorSubfindID INT(32),
    Minor_PastNextProgenitorMaxMass_stars FLOAT,
    Minor_PastNextProgenitorMaxMass_gas FLOAT,
    Minor_PastNextProgenitorMaxMassInRad_stars FLOAT,
    Minor_PastNextProgenitorMaxMassInRad_gas FLOAT,
    Minor_CountUntilRootDescendent INT(32),
    Minor_CountUntil250Myr INT(32),
    Minor_CountUntil500Myr INT(32),
    Minor_CountUntil1Gyr INT(32),
    Minor_CountUntil2Gyr INT(32),
    Minor_CountUntil3Gyr INT(32),
    Minor_FutureSnapNum INT(16),
    Minor_FutureSubfindID INT(32),
    Minor_FutureMassRatio FLOAT,
    Minor_TimeUntilMerger FLOAT,
    Minor_FutureMainProgenitorSnapNum INT(16),
    Minor_FutureMainProgenitorSubfindID INT(32),
    Minor_FutureMainProgenitorMaxMass_stars FLOAT,
    Minor_FutureMainProgenitorMaxMass_gas FLOAT,
    Minor_FutureMainProgenitorMaxMassInRad_stars FLOAT,
    Minor_FutureMainProgenitorMaxMassInRad_gas FLOAT,
    Minor_FutureNextProgenitorSnapNum INT(16),
    Minor_FutureNextProgenitorSubfindID INT(32),
    Minor_FutureNextProgenitorMaxMass_stars FLOAT,
    Minor_FutureNextProgenitorMaxMass_gas FLOAT,
    Minor_FutureNextProgenitorMaxMassInRad_stars FLOAT,
    Minor_FutureNextProgenitorMaxMassInRad_gas FLOAT,
    Minor_TreeFlag INT(16),

    Tiny_CountSinceMainLeafProgenitor INT(32),
    Tiny_CountSinceHalfScaleFactor INT(32),
    Tiny_CountSince250Myr INT(32),
    Tiny_CountSince500Myr INT(32),
    Tiny_CountSince1Gyr INT(32),
    Tiny_CountSince2Gyr INT(32),
    Tiny_CountSince3Gyr INT(32),
    Tiny_FreqencySinceMainLeafProgenitor FLOAT,
    Tiny_PastSnapNum INT(16),
    Tiny_PastSubfindID INT(32),
    Tiny_PastMassRatio FLOAT,
    Tiny_TimeSinceMerger FLOAT,
    Tiny_PastMainProgenitorSnapNum INT(16),
    Tiny_PastMainProgenitorSubfindID INT(32),
    Tiny_PastMainProgenitorMaxMass_stars FLOAT,
    Tiny_PastMainProgenitorMaxMass_gas FLOAT,
    Tiny_PastMainProgenitorMaxMassInRad_stars FLOAT,
    Tiny_PastMainProgenitorMaxMassInRad_gas FLOAT,
    Tiny_PastNextProgenitorSnapNum INT(16),
    Tiny_PastNextProgenitorSubfindID INT(32),
    Tiny_PastNextProgenitorMaxMass_stars FLOAT,
    Tiny_PastNextProgenitorMaxMass_gas FLOAT,
    Tiny_PastNextProgenitorMaxMassInRad_stars FLOAT,
    Tiny_PastNextProgenitorMaxMassInRad_gas FLOAT,
    Tiny_CountUntilRootDescendent INT(32),
    Tiny_CountUntil250Myr INT(32),
    Tiny_CountUntil500Myr INT(32),
    Tiny_CountUntil1Gyr INT(32),
    Tiny_CountUntil2Gyr INT(32),
    Tiny_CountUntil3Gyr INT(32),
    Tiny_FutureSnapNum INT(16),
    Tiny_FutureSubfindID INT(32),
    Tiny_FutureMassRatio FLOAT,
    Tiny_TimeUntilMerger FLOAT,
    Tiny_FutureMainProgenitorSnapNum INT(16),
    Tiny_FutureMainProgenitorSubfindID INT(32),
    Tiny_FutureMainProgenitorMaxMass_stars FLOAT,
    Tiny_FutureMainProgenitorMaxMass_gas FLOAT,
    Tiny_FutureMainProgenitorMaxMassInRad_stars FLOAT,
    Tiny_FutureMainProgenitorMaxMassInRad_gas FLOAT,
    Tiny_FutureNextProgenitorSnapNum INT(16),
    Tiny_FutureNextProgenitorSubfindID INT(32),
    Tiny_FutureNextProgenitorMaxMass_stars FLOAT,
    Tiny_FutureNextProgenitorMaxMass_gas FLOAT,
    Tiny_FutureNextProgenitorMaxMassInRad_stars FLOAT,
    Tiny_FutureNextProgenitorMaxMassInRad_gas FLOAT,
    Tiny_TreeFlag INT(16),
    
    PRIMARY KEY (dbID)
);
