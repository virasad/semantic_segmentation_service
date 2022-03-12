from flash.image import SemanticSegmentation, SemanticSegmentationData

def get_dataset_for_flash(data_path, num_classes ,labels_path, batch_size, num_workers = 8):
    datamodule = SemanticSegmentationData.from_folders(
        train_folder=data_path,
        train_target_folder=labels_path,
        val_split=0.2,
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return datamodule