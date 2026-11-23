# NASA ISS-LIS HDF Safe Cleanup Report

- **Execution Mode**: `DRY RUN (No files modified or deleted)`
- **Total HDF Files Analyzed**: 4407
- **Eligible for Safe Deletion**: 4407 (100% paired with verified NetCDF granules)
- **Ineligible / Skipped**: 0
- **Reclaimable Storage Space**: 9004.37 MB (8.79 GB)

> [!NOTE]
> Dry-run mode completed. No `.hdf` or `.nc` files were deleted. To perform the actual deletion, run:
> ```powershell
> python main.py cleanup-hdf --delete-hdf
> ```
