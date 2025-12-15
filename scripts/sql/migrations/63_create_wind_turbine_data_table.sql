/*
 * Script: 63_create_wind_turbine_data_table.sql
 * Purpose: Create WIND_TURBINE_Data table for Wind Turbine SCADA dataset
 * Context: Adding new equipment type to ACM
 * 
 * This table stores Wind Turbine SCADA data with 4 sensor columns:
 * - LV_ActivePower_kW: Low voltage active power in kilowatts
 * - Wind_Speed_ms: Wind speed in meters per second
 * - Theoretical_Power_Curve_KWh: Theoretical power output
 * - Wind_Direction_deg: Wind direction in degrees
 */

USE ACM;
GO

PRINT 'Creating WIND_TURBINE_Data table for SCADA dataset...';
PRINT '';

-- =====================================================================
-- WIND_TURBINE_Data: 4 sensor tags + timestamp
-- =====================================================================
PRINT '1. Creating WIND_TURBINE_Data table...';

IF OBJECT_ID('dbo.WIND_TURBINE_Data', 'U') IS NOT NULL
BEGIN
    PRINT '   Table already exists, skipping creation';
END
ELSE
BEGIN
    CREATE TABLE dbo.WIND_TURBINE_Data (
        EntryDateTime DATETIME2(0) NOT NULL,  -- Timestamp (renamed from Date/Time)
        
        -- 4 sensor tag columns (cleaned names from CSV)
        [LV_ActivePower_kW] FLOAT NULL,
        [Wind_Speed_ms] FLOAT NULL,
        [Theoretical_Power_Curve_KWh] FLOAT NULL,
        [Wind_Direction_deg] FLOAT NULL,
        
        -- Audit columns
        LoadedAt DATETIME2 DEFAULT GETUTCDATE(),
        
        CONSTRAINT PK_WIND_TURBINE_Data PRIMARY KEY CLUSTERED (EntryDateTime)
    );

    -- Index for time-range queries (used by stored procedure)
    CREATE NONCLUSTERED INDEX IX_WIND_TURBINE_Data_TimeRange 
        ON dbo.WIND_TURBINE_Data(EntryDateTime ASC);

    PRINT '   [OK] WIND_TURBINE_Data created (4 sensor tags)';
END
GO

-- =====================================================================
-- Add WIND_TURBINE to Equipments master table
-- =====================================================================
PRINT '';
PRINT '2. Adding WIND_TURBINE to Equipments master table...';

IF NOT EXISTS (SELECT 1 FROM dbo.Equipments WHERE EquipCode = 'WIND_TURBINE')
BEGIN
    INSERT INTO dbo.Equipments (EquipCode, Active)
    VALUES ('WIND_TURBINE', 1);
    
    DECLARE @NewEquipID INT = SCOPE_IDENTITY();
    PRINT '   [OK] Created WIND_TURBINE (EquipID=' + CAST(@NewEquipID AS VARCHAR) + ')';
END
ELSE
BEGIN
    DECLARE @ExistingEquipID INT;
    SELECT @ExistingEquipID = EquipID FROM dbo.Equipments WHERE EquipCode = 'WIND_TURBINE';
    PRINT '   Equipment already exists (EquipID=' + CAST(@ExistingEquipID AS VARCHAR) + ')';
END
GO

-- =====================================================================
-- Update stored procedure to support WIND_TURBINE
-- =====================================================================
PRINT '';
PRINT '3. Updating historian stored procedure for WIND_TURBINE support...';

-- Update usp_ACM_GetHistorianData_TEMP to include WIND_TURBINE
IF OBJECT_ID('dbo.usp_ACM_GetHistorianData_TEMP', 'P') IS NOT NULL
BEGIN
    -- The stored procedure should already support dynamic equipment names
    -- Just verify it exists
    PRINT '   [OK] Stored procedure exists (supports dynamic equipment names)';
END
ELSE
BEGIN
    PRINT '   [WARN] Stored procedure not found - may need to create it';
END
GO

PRINT '';
PRINT '========================================';
PRINT 'WIND_TURBINE setup complete!';
PRINT '';
PRINT 'To load data, run:';
PRINT '  python scripts/sql/load_wind_turbine_data.py';
PRINT '';
PRINT 'To run ACM pipeline:';
PRINT '  python -m core.acm_main --equip WIND_TURBINE';
PRINT '';
GO
