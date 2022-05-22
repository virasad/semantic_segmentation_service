from flash.image import SemanticSegmentationData


def get_dataset_for_flash(train_folder, train_target_folder, val_folder, val_target_folder, batch_size, num_workers=8, num_classes=2):
    datamodule = SemanticSegmentationData.from_folders(
        train_folder=train_folder,
        train_target_folder=train_target_folder,
        val_folder=val_folder,
        val_target_folder=val_target_folder,
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return datamodule
