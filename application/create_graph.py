import os
import hydra
from numpy.ma.core import max_filler
from omegaconf import DictConfig
from hovsg.graph.graph import Graph

# pylint: disable=all


@hydra.main(version_base=None, config_path="../config", config_name="create_graph")
def main(params: DictConfig):
    # create logging directory
    save_dir = os.path.join(params.main.save_path, params.main.dataset, params.main.scene_id)
    params.main.save_path = save_dir
    params.main.dataset_path = os.path.join(params.main.dataset_path, params.main.split, params.main.scene_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # create graph object (loads models and dataset once)
    hovsg = Graph(params)

    incremental = getattr(params.pipeline, "incremental", False)

    if not incremental:
        # --- Original single-graph mode ---
        hovsg.create_feature_map()

        hovsg.save_masked_pcds(path=save_dir, state="both")
        hovsg.save_full_pcd(path=save_dir)
        hovsg.save_full_pcd_feats(path=save_dir)

        print(params.main.dataset)
        if params.main.dataset != "replica" and params.main.dataset != "scannet" and params.pipeline.create_graph:
            hovsg.build_graph(save_path=save_dir)
        else:
            print("Skipping hierarchical scene graph creation for Replica and ScanNet datasets.")
    else:
        # --- Incremental mode: build a new graph at every frame step ---
        skip = params.pipeline.skip_frames
        dataset_len = len(hovsg.dataset)
        # frame indices that will be processed: 10, 10+skip, 10+2*skip, ...
        frame_indices = list(range(10, dataset_len, skip))
        # Start from the third idx so the creation graph logic works
        # (problems with floors [at least has to be one] and rooms detections [at least has to be two o them])
        frame_indices = frame_indices[4:]
        if frame_indices[-1] < dataset_len - 1:
            frame_indices.append(dataset_len - 1)

        for step_idx, max_frame_idx in enumerate(frame_indices):

            step_label = f"frame_{max_frame_idx:06d}"
            step_dir = os.path.join(save_dir, step_label)
            os.makedirs(step_dir, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Incremental step {step_idx + 1}/{len(frame_indices)} "
                  f"— processing frames 0..{max_frame_idx} "
                  f"(max_frame={max_frame_idx})")
            print(f"{'='*60}")

            # reset accumulated state from previous step
            hovsg.reset()

            # build feature map with only the frames seen so far
            hovsg.create_feature_map(max_frame=max_frame_idx)

            # save outputs for this step
            hovsg.save_masked_pcds(path=step_dir, state="both")
            hovsg.save_full_pcd(path=step_dir)
            hovsg.save_full_pcd_feats(path=step_dir)

            # build hierarchical graph if applicable
            print(params.main.dataset)
            if params.main.dataset != "replica" and params.main.dataset != "scannet" and params.pipeline.create_graph:
                hovsg.build_graph(save_path=step_dir, max_frame=max_frame_idx)
            else:
                print("Skipping hierarchical scene graph creation for Replica and ScanNet datasets.")

        print(f"\nIncremental graph creation complete — {len(frame_indices)} steps saved under {save_dir}")

if __name__ == "__main__":
    main()