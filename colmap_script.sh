DATASET_PATH=.

colmap feature_extractor --database_path database.db --image_path reduced_fps_one_cam --ImageReader.camera_model RADIAL_FISHEYE
colmap sequential_matcher    --database_path database.db
mkdir $DATASET_PATH/sparse
colmap mapper     --database_path database.db     --image_path reduced_fps_one_cam --output_path sparse
colmap image_undistorter     --image_path reduced_fps_one_cam     --input_path sparse/0     --output_path dense     --output_type COLMAP     --max_image_size 2000
colmap patch_match_stereo     --workspace_path dense     --workspace_format COLMAP      --PatchMatchStereo.geom_consistency true
colmap stereo_fusion     --workspace_path dense     --workspace_format COLMAP     --input_type geometric     --output_path dense/fused.ply
colmap poisson_mesher     --input_path dense/fused.ply     --output_path dense/meshed-poisson.ply
colmap delaunay_mesher     --input_path dense     --output_path dense/meshed-delaunay.ply
