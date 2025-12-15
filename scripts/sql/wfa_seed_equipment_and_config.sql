-- Seed Wind Farm A equipment and config into SQL (Equipment, ACM_Config)
-- Deterministic EquipID mapping: 5000 + asset_id
-- Safe to re-run; skips existing EquipCode/ParamPath pairs

SET NOCOUNT ON;

DECLARE @targets TABLE (
    asset_id INT NOT NULL,
    EquipID INT NOT NULL,
    EquipCode NVARCHAR(100) NOT NULL,
    EquipName NVARCHAR(200) NOT NULL
);

INSERT INTO @targets(asset_id, EquipID, EquipCode, EquipName) VALUES
 (0, 5000, 'WFA_TURBINE_0', 'Wind Farm A Turbine 0'),
 (3, 5003, 'WFA_TURBINE_3', 'Wind Farm A Turbine 3'),
 (10, 5010, 'WFA_TURBINE_10', 'Wind Farm A Turbine 10'),
 (11, 5011, 'WFA_TURBINE_11', 'Wind Farm A Turbine 11'),
 (13, 5013, 'WFA_TURBINE_13', 'Wind Farm A Turbine 13'),
 (14, 5014, 'WFA_TURBINE_14', 'Wind Farm A Turbine 14'),
 (17, 5017, 'WFA_TURBINE_17', 'Wind Farm A Turbine 17'),
 (21, 5021, 'WFA_TURBINE_21', 'Wind Farm A Turbine 21'),
 (22, 5022, 'WFA_TURBINE_22', 'Wind Farm A Turbine 22'),
 (24, 5024, 'WFA_TURBINE_24', 'Wind Farm A Turbine 24'),
 (25, 5025, 'WFA_TURBINE_25', 'Wind Farm A Turbine 25'),
 (26, 5026, 'WFA_TURBINE_26', 'Wind Farm A Turbine 26'),
 (38, 5038, 'WFA_TURBINE_38', 'Wind Farm A Turbine 38'),
 (40, 5040, 'WFA_TURBINE_40', 'Wind Farm A Turbine 40'),
 (42, 5042, 'WFA_TURBINE_42', 'Wind Farm A Turbine 42'),
 (45, 5045, 'WFA_TURBINE_45', 'Wind Farm A Turbine 45'),
 (51, 5051, 'WFA_TURBINE_51', 'Wind Farm A Turbine 51'),
 (68, 5068, 'WFA_TURBINE_68', 'Wind Farm A Turbine 68'),
 (69, 5069, 'WFA_TURBINE_69', 'Wind Farm A Turbine 69'),
 (71, 5071, 'WFA_TURBINE_71', 'Wind Farm A Turbine 71'),
 (72, 5072, 'WFA_TURBINE_72', 'Wind Farm A Turbine 72'),
 (73, 5073, 'WFA_TURBINE_73', 'Wind Farm A Turbine 73'),
 (84, 5084, 'WFA_TURBINE_84', 'Wind Farm A Turbine 84'),
 (92, 5092, 'WFA_TURBINE_92', 'Wind Farm A Turbine 92');

-- Equipment inserts
DECLARE @equipIsIdentity BIT = COLUMNPROPERTY(OBJECT_ID('dbo.Equipment'), 'EquipID', 'IsIdentity');
IF @equipIsIdentity = 1 SET IDENTITY_INSERT dbo.Equipment ON;

INSERT INTO dbo.Equipment (EquipID, EquipCode, EquipName, Area, Unit, Status, CommissionDate)
SELECT t.EquipID, t.EquipCode, t.EquipName, N'Wind Farm A', N'Turbine', 1, '2025-01-01'
FROM @targets t
WHERE NOT EXISTS (SELECT 1 FROM dbo.Equipment e WHERE e.EquipCode = t.EquipCode);

IF @equipIsIdentity = 1 SET IDENTITY_INSERT dbo.Equipment OFF;

-- Shared tag list
DECLARE @tagColumns NVARCHAR(MAX) =
    '["sensor_0_avg","sensor_1_avg","sensor_2_avg","wind_speed_3_avg","wind_speed_4_avg","wind_speed_3_max","wind_speed_3_min","wind_speed_3_std","sensor_5_avg","sensor_5_max","sensor_5_min","sensor_5_std","sensor_6_avg","sensor_7_avg","sensor_8_avg","sensor_9_avg","sensor_10_avg","sensor_11_avg","sensor_12_avg","sensor_13_avg","sensor_14_avg","sensor_15_avg","sensor_16_avg","sensor_17_avg","sensor_18_avg","sensor_18_max","sensor_18_min","sensor_18_std","sensor_19_avg","sensor_20_avg","sensor_21_avg","sensor_22_avg","sensor_23_avg","sensor_24_avg","sensor_25_avg","sensor_26_avg","reactive_power_27_avg","reactive_power_27_max","reactive_power_27_min","reactive_power_27_std","reactive_power_28_avg","reactive_power_28_max","reactive_power_28_min","reactive_power_28_std","power_29_avg","power_29_max","power_29_min","power_29_std","power_30_avg","power_30_max","power_30_min","power_30_std","sensor_31_avg","sensor_31_max","sensor_31_min","sensor_31_std","sensor_32_avg","sensor_33_avg","sensor_34_avg","sensor_35_avg","sensor_36_avg","sensor_37_avg","sensor_38_avg","sensor_39_avg","sensor_40_avg","sensor_41_avg","sensor_42_avg","sensor_43_avg","sensor_44","sensor_45","sensor_46","sensor_47","sensor_48","sensor_49","sensor_50","sensor_51","sensor_52_avg","sensor_52_max","sensor_52_min","sensor_52_std","sensor_53_avg"]';

-- Config inserts (ACM_Config) with identity-safe handling
DECLARE @cfgIsIdentity BIT = COLUMNPROPERTY(OBJECT_ID('dbo.ACM_Config'), 'ConfigID', 'IsIdentity');
DECLARE @cfgBase INT = (SELECT ISNULL(MAX(ConfigID), 0) FROM dbo.ACM_Config);

IF @cfgIsIdentity = 1
BEGIN
    INSERT INTO dbo.ACM_Config (EquipID, ParamPath, ParamValue, ValueType)
    SELECT t.EquipID, v.ParamPath, v.ParamValue, v.ValueType
    FROM @targets t
    CROSS APPLY (VALUES
        ('data.timestamp_col', 'EntryDateTime', 'string'),
        ('data.sampling_secs', '600', 'int'),
        ('data.tag_columns', @tagColumns, 'list')
    ) v(ParamPath, ParamValue, ValueType)
    WHERE NOT EXISTS (
        SELECT 1 FROM dbo.ACM_Config c
        WHERE c.EquipID = t.EquipID AND c.ParamPath = v.ParamPath
    );
END
ELSE
BEGIN
    WITH to_add AS (
        SELECT t.EquipID, v.ParamPath, v.ParamValue, v.ValueType,
               ROW_NUMBER() OVER (ORDER BY t.EquipID, v.ParamPath) AS rn
        FROM @targets t
        CROSS APPLY (VALUES
            ('data.timestamp_col', 'EntryDateTime', 'string'),
            ('data.sampling_secs', '600', 'int'),
            ('data.tag_columns', @tagColumns, 'list')
        ) v(ParamPath, ParamValue, ValueType)
        WHERE NOT EXISTS (
            SELECT 1 FROM dbo.ACM_Config c
            WHERE c.EquipID = t.EquipID AND c.ParamPath = v.ParamPath
        )
    )
    INSERT INTO dbo.ACM_Config (ConfigID, EquipID, ParamPath, ParamValue, ValueType)
    SELECT @cfgBase + rn, EquipID, ParamPath, ParamValue, ValueType
    FROM to_add;
END

PRINT 'Wind Farm A equipment and config seeded.';
