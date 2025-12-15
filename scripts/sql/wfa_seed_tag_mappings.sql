-- Wind Farm A Tag Mappings - Populate ACM_TagEquipmentMap for all turbines
-- Maps CSV sensor columns to tags for each turbine
-- Run after equipment seeding to enable historian data retrieval

SET NOCOUNT ON;

-- Sensor tags from CSV header (excluding time_stamp, asset_id, id, train_test, status_type_id which are metadata)
DECLARE @sensorTags TABLE (TagName NVARCHAR(128), TagType NVARCHAR(50), TagUnit NVARCHAR(50));
INSERT INTO @sensorTags (TagName, TagType, TagUnit) VALUES
    ('sensor_0_avg', 'SENSOR', 'units'),
    ('sensor_1_avg', 'SENSOR', 'units'),
    ('sensor_2_avg', 'SENSOR', 'units'),
    ('wind_speed_3_avg', 'WIND', 'm/s'),
    ('wind_speed_4_avg', 'WIND', 'm/s'),
    ('wind_speed_3_max', 'WIND', 'm/s'),
    ('wind_speed_3_min', 'WIND', 'm/s'),
    ('wind_speed_3_std', 'WIND', 'm/s'),
    ('sensor_5_avg', 'SENSOR', 'units'),
    ('sensor_5_max', 'SENSOR', 'units'),
    ('sensor_5_min', 'SENSOR', 'units'),
    ('sensor_5_std', 'SENSOR', 'units'),
    ('sensor_6_avg', 'SENSOR', 'units'),
    ('sensor_7_avg', 'SENSOR', 'units'),
    ('sensor_8_avg', 'SENSOR', 'units'),
    ('sensor_9_avg', 'SENSOR', 'units'),
    ('sensor_10_avg', 'SENSOR', 'units'),
    ('sensor_11_avg', 'SENSOR', 'units'),
    ('sensor_12_avg', 'SENSOR', 'units'),
    ('sensor_13_avg', 'SENSOR', 'units'),
    ('sensor_14_avg', 'SENSOR', 'units'),
    ('sensor_15_avg', 'SENSOR', 'units'),
    ('sensor_16_avg', 'SENSOR', 'units'),
    ('sensor_17_avg', 'SENSOR', 'units'),
    ('sensor_18_avg', 'SENSOR', 'units'),
    ('sensor_18_max', 'SENSOR', 'units'),
    ('sensor_18_min', 'SENSOR', 'units'),
    ('sensor_18_std', 'SENSOR', 'units'),
    ('sensor_19_avg', 'SENSOR', 'units'),
    ('sensor_20_avg', 'SENSOR', 'units'),
    ('sensor_21_avg', 'SENSOR', 'units'),
    ('sensor_22_avg', 'SENSOR', 'units'),
    ('sensor_23_avg', 'SENSOR', 'units'),
    ('sensor_24_avg', 'SENSOR', 'units'),
    ('sensor_25_avg', 'SENSOR', 'units'),
    ('sensor_26_avg', 'SENSOR', 'units'),
    ('reactive_power_27_avg', 'POWER', 'kVAR'),
    ('reactive_power_27_max', 'POWER', 'kVAR'),
    ('reactive_power_27_min', 'POWER', 'kVAR'),
    ('reactive_power_27_std', 'POWER', 'kVAR'),
    ('reactive_power_28_avg', 'POWER', 'kVAR'),
    ('reactive_power_28_max', 'POWER', 'kVAR'),
    ('reactive_power_28_min', 'POWER', 'kVAR'),
    ('reactive_power_28_std', 'POWER', 'kVAR'),
    ('power_29_avg', 'POWER', 'kW'),
    ('power_29_max', 'POWER', 'kW'),
    ('power_29_min', 'POWER', 'kW'),
    ('power_29_std', 'POWER', 'kW'),
    ('power_30_avg', 'POWER', 'kW'),
    ('power_30_max', 'POWER', 'kW'),
    ('power_30_min', 'POWER', 'kW'),
    ('power_30_std', 'POWER', 'kW'),
    ('sensor_31_avg', 'SENSOR', 'units'),
    ('sensor_31_max', 'SENSOR', 'units'),
    ('sensor_31_min', 'SENSOR', 'units'),
    ('sensor_31_std', 'SENSOR', 'units'),
    ('sensor_32_avg', 'SENSOR', 'units'),
    ('sensor_33_avg', 'SENSOR', 'units'),
    ('sensor_34_avg', 'SENSOR', 'units'),
    ('sensor_35_avg', 'SENSOR', 'units'),
    ('sensor_36_avg', 'SENSOR', 'units'),
    ('sensor_37_avg', 'SENSOR', 'units'),
    ('sensor_38_avg', 'SENSOR', 'units'),
    ('sensor_39_avg', 'SENSOR', 'units'),
    ('sensor_40_avg', 'SENSOR', 'units'),
    ('sensor_41_avg', 'SENSOR', 'units'),
    ('sensor_42_avg', 'SENSOR', 'units'),
    ('sensor_43_avg', 'SENSOR', 'units'),
    ('sensor_44', 'SENSOR', 'units'),
    ('sensor_45', 'SENSOR', 'units'),
    ('sensor_46', 'SENSOR', 'units'),
    ('sensor_47', 'SENSOR', 'units'),
    ('sensor_48', 'SENSOR', 'units'),
    ('sensor_49', 'SENSOR', 'units'),
    ('sensor_50', 'SENSOR', 'units'),
    ('sensor_51', 'SENSOR', 'units'),
    ('sensor_52_avg', 'SENSOR', 'units'),
    ('sensor_52_max', 'SENSOR', 'units'),
    ('sensor_52_min', 'SENSOR', 'units'),
    ('sensor_52_std', 'SENSOR', 'units'),
    ('sensor_53_avg', 'SENSOR', 'units');

-- Insert tag mappings for each Wind Farm A turbine
INSERT INTO dbo.ACM_TagEquipmentMap (TagName, EquipmentName, EquipID, TagDescription, TagUnit, TagType, IsActive, CreatedAt, UpdatedAt)
SELECT 
    t.TagName,
    e.EquipName,
    e.EquipID,
    CONCAT(t.TagName, ' for ', e.EquipCode) AS TagDescription,
    t.TagUnit,
    t.TagType,
    1 AS IsActive,
    GETDATE() AS CreatedAt,
    GETDATE() AS UpdatedAt
FROM @sensorTags t
CROSS JOIN dbo.Equipment e
WHERE e.EquipCode LIKE 'WFA_TURBINE_%'
  AND NOT EXISTS (
      SELECT 1 FROM dbo.ACM_TagEquipmentMap m 
      WHERE m.EquipID = e.EquipID AND m.TagName = t.TagName
  );

DECLARE @inserted INT = @@ROWCOUNT;
PRINT CONCAT('Inserted ', @inserted, ' tag mappings for Wind Farm A turbines.');

-- Verify counts
SELECT e.EquipCode, COUNT(m.TagID) AS TagCount
FROM dbo.Equipment e
LEFT JOIN dbo.ACM_TagEquipmentMap m ON e.EquipID = m.EquipID
WHERE e.EquipCode LIKE 'WFA_TURBINE_%'
GROUP BY e.EquipCode
ORDER BY e.EquipCode;
