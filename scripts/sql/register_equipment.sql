-- Register equipment records for ACM SQL integration
-- Equipment IDs must match those used in config_table.csv

USE ACM;
GO

-- Check if Equipment table exists
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Equipment')
BEGIN
    PRINT 'ERROR: Equipment table does not exist. Run schema creation scripts first.';
    RETURN;
END
GO

-- Insert FD_FAN (EquipID = 1)
IF NOT EXISTS (SELECT 1 FROM dbo.Equipment WHERE EquipID = 1)
BEGIN
    SET IDENTITY_INSERT dbo.Equipment ON;
    
    INSERT INTO dbo.Equipment (EquipID, EquipCode, EquipName, Area, Unit, Status, CommissionDate, CreatedAtUTC)
    VALUES (
        1,
        'FD_FAN',
        'Forced Draft Fan',
        'Boiler Section',
        'Plant A',
        1,  -- Active
        CAST('2025-01-01' AS DATETIME2),
        SYSUTCDATETIME()
    );
    
    SET IDENTITY_INSERT dbo.Equipment OFF;
    PRINT 'Inserted equipment: FD_FAN (EquipID=1)';
END
ELSE
BEGIN
    PRINT 'Equipment FD_FAN (EquipID=1) already exists';
END
GO

-- Insert GAS_TURBINE (EquipID = 2621)
IF NOT EXISTS (SELECT 1 FROM dbo.Equipment WHERE EquipID = 2621)
BEGIN
    SET IDENTITY_INSERT dbo.Equipment ON;
    
    INSERT INTO dbo.Equipment (EquipID, EquipCode, EquipName, Area, Unit, Status, CommissionDate, CreatedAtUTC)
    VALUES (
        2621,
        'GAS_TURBINE',
        'Gas Turbine Generator',
        'Power Generation',
        'Plant A',
        1,  -- Active
        CAST('2025-01-01' AS DATETIME2),
        SYSUTCDATETIME()
    );
    
    SET IDENTITY_INSERT dbo.Equipment OFF;
    PRINT 'Inserted equipment: GAS_TURBINE (EquipID=2621)';
END
ELSE
BEGIN
    PRINT 'Equipment GAS_TURBINE (EquipID=2621) already exists';
END
GO

-- Verify insertions
SELECT 
    EquipID,
    EquipCode,
    EquipName,
    Area,
    Unit,
    Status,
    CommissionDate,
    CreatedAtUTC
FROM dbo.Equipment
WHERE EquipID IN (1, 2621)
ORDER BY EquipID;
GO

PRINT 'Equipment registration complete.';
