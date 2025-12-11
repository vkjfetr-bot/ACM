USE [ACM];
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'UQ_ACM_AdaptiveConfig_EquipKey' AND parent_object_id = OBJECT_ID('dbo.[ACM_AdaptiveConfig]'))
BEGIN
    ALTER TABLE dbo.[ACM_AdaptiveConfig] ADD CONSTRAINT [UQ_ACM_AdaptiveConfig_EquipKey] UNIQUE ([EquipID], [ConfigKey]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'UQ_ACM_Config_Path' AND parent_object_id = OBJECT_ID('dbo.[ACM_Config]'))
BEGIN
    ALTER TABLE dbo.[ACM_Config] ADD CONSTRAINT [UQ_ACM_Config_Path] UNIQUE ([EquipID], [ParamPath]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'UQ_TagEquipmentMap_TagEquip' AND parent_object_id = OBJECT_ID('dbo.[ACM_TagEquipmentMap]'))
BEGIN
    ALTER TABLE dbo.[ACM_TagEquipmentMap] ADD CONSTRAINT [UQ_TagEquipmentMap_TagEquip] UNIQUE ([TagName], [EquipID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'UQ__Equipmen__50592EBA17802490' AND parent_object_id = OBJECT_ID('dbo.[Equipment]'))
BEGIN
    ALTER TABLE dbo.[Equipment] ADD CONSTRAINT [UQ__Equipmen__50592EBA17802490] UNIQUE ([EquipCode]);
END
GO
