import random
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class RNNDataset:
    def __init__(self, config):
        self.config = config

        self.train_video_ids, self.test_video_ids = train_test_split(
            sorted(config.all_video_ids()),
            test_size=0.05,
            random_state=12)
        self.train_video_ids = set(self.train_video_ids)
        self.test_video_ids = set(self.test_video_ids)

        self.gt = pd.read_csv(config.length_path())
        self.gt.dropna(axis=0, inplace=True)
        self.gt['have_frame'] = 1.0

        self.video_frames_count = {}
        self.video_data = {}
        self.video_data_gt = {}

        print('load video data...')
        self.video_frames_count, self.video_data, self.video_data_gt, self.columns = self.load()
        print('loaded')

    def load(self):
        print('generate video data...')
        video_frames_count = {}
        video_data = {}
        video_data_gt = {}
        num_frames = pd.read_csv(self.config.num_frames_path())
        detect_out = pd.read_csv(self.config.detect_inference_path())
        classify_out = pd.read_csv(self.config.classify_inference_path())
        species_cols = ['species_' + s for s in self.config.species()]
        columns = ['species__',]
        columns += species_cols
        columns += ['no_fish', 'covered', 'clear', 'x', 'y', 'w', 'h',
                    'det_conf', 'det_species']

        for _, cnt in num_frames.iterrows():
            video_id = cnt.video_id
            nb_frames = cnt.num_frames
            video_frames_count[video_id] = nb_frames
            ds_detection = detect_out.loc[detect_out['video_id'] == video_id]
            ds_classification = classify_out.loc[classify_out['video_id'] == video_id]
            ds_combined = ds_classification.join(ds_detection, on='frame', how='left', rsuffix='_det')
            ds_combined.x /= self.config.detect_width()
            ds_combined.y /= self.config.detect_height()
            ds_combined.w /= self.config.detect_width()
            ds_combined.h /= self.config.detect_height()
            all_frames = pd.DataFrame({'frame': list(range(nb_frames))})
            ds_combined = all_frames.merge(ds_combined, on='frame', how='left').fillna(0.0)
            ds_combined['species__'] = 1.0 - sum([ds_combined[scol] for scol in species_cols])
            video_data[video_id] = ds_combined.as_matrix(columns=columns)

            all_frames = pd.DataFrame({'frame': list(range(nb_frames))})
            gt_combined = all_frames.merge(self.gt.loc[self.gt.video_id == video_id], on='frame', how='left').fillna(
                0.0)
            video_data_gt[video_id] = gt_combined.as_matrix(columns=['have_frame'])

        return video_frames_count, video_data, video_data_gt, columns

    def generate_x(self, video_id, offset):
        nb_steps = self.config.count_num_steps()
        res = np.zeros((nb_steps, self.config.count_num_features()))
        nb_res_steps = nb_steps - self.config.count_num_steps_crop() * 2

        nb_frames = self.video_frames_count[video_id]
        steps_before = min(self.config.count_num_steps_crop(), offset)
        steps_after = min(self.config.count_num_steps_crop(), nb_frames - offset - nb_res_steps)

        res[self.config.count_num_steps_crop() - steps_before:nb_steps - self.config.count_num_steps_crop() + steps_after, :] = \
            self.video_data[video_id][offset - steps_before:offset + nb_res_steps + steps_after, :]
        return res

    def generate_y(self, video_id, offset):
        res = np.zeros((self.config.count_num_res_steps(),))
        nb_frames = self.video_frames_count[video_id]
        frames_used = min(self.config.count_num_res_steps(), nb_frames - offset)
        res[0:frames_used] = self.video_data_gt[video_id][offset:offset + frames_used, 0]
        return res

    def generate(self, batch_size, use_cumsum=True):
        valid_video_ids = list(self.train_video_ids.intersection(self.video_data.keys()))
        shape = (self.config.count_num_steps(), self.config.count_num_features())
        batch_x = np.zeros((batch_size,) + shape, dtype=np.float32)
        batch_y = np.zeros((batch_size, self.config.count_num_res_steps()), dtype=np.float32)
        while True:
            for batch_idx in range(batch_size):
                video_id = random.choice(valid_video_ids)

                if self.video_frames_count[video_id] < self.config.count_num_res_steps():
                    offset = 0
                else:
                    offset = random.randrange(0, self.video_frames_count[video_id] - self.config.count_num_res_steps())

                batch_x[batch_idx] = self.generate_x(video_id, offset)
                batch_y[batch_idx] = self.generate_y(video_id, offset)
                """
                if np.any(np.isnan(batch_x[batch_idx])):
                    print("INPUT TO NETWORK CONTAINS NAN! {}".format(batch_x[batch_idx]))
                if np.any(np.isnan(batch_y[batch_idx])):
                    print("OUTPUT TO NETWORK CONTAINS NAN!")
                if not np.any(np.nonzero(batch_x[batch_idx])):
                    print("INPUT TO NETWORK IS ALL ZEROS!")
                if not np.any(np.nonzero(batch_y[batch_idx])):
                    print("OUTPUT TO NETWORK IS ALL ZEROS!")
                """


            if use_cumsum:
                yield (batch_x, {'current_values': batch_y, 'cumsum_values': np.cumsum(batch_y, axis=1)})
            else:
                yield (batch_x, batch_y)

    def test_batches(self, batch_size):
        valid_video_ids = sorted(self.test_video_ids.intersection(self.video_data.keys()))
        batch_idx = 0
        batches_count = 0
        for video_id in valid_video_ids:
            for offset in range(0, self.video_frames_count[video_id], self.config.count_num_res_steps()):
                batch_idx += 1

                if batch_idx == batch_size:
                    batch_idx = 0
                    batches_count += 1
        print('val batches count:', batches_count)
        return batches_count

    def generate_test(self, batch_size, verbose=False, use_cumsum=True):
        valid_video_ids = sorted(self.test_video_ids.intersection(self.video_data.keys()))
        print('test valid_video_ids:', len(valid_video_ids))

        shape = (self.config.count_num_steps(), self.config.count_num_features())
        batch_x = np.zeros((batch_size,) + shape, dtype=np.float32)
        batch_y = np.zeros((batch_size, self.config.count_num_res_steps()), dtype=np.float32)
        while True:
            batch_idx = 0
            for video_id in valid_video_ids:
                for offset in range(0, self.video_frames_count[video_id], self.config.count_num_res_steps()):
                    if verbose:
                        print(video_id, offset)
                    batch_x[batch_idx] = self.generate_x(video_id, offset)
                    batch_y[batch_idx] = self.generate_y(video_id, offset)
                    batch_idx += 1

                    self.last_offset = offset

                    if batch_idx == batch_size:
                        batch_idx = 0
                        if use_cumsum:
                            yield (batch_x, {'current_values': batch_y, 'cumsum_values': np.cumsum(batch_y, axis=1)})
                        else:
                            yield (batch_x, batch_y)


