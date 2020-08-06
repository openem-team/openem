""" Various methods to compute weights """

import numpy as np
import math
import progressbar

class HybridWeights:
    def __init__(self,
                 det_comparator,
                 track_comparator,
                 app_normalizer,
                 vid_dims,
                 fps,
                 single_frame_bias,
                 batch_size):
        self.det_comparator = det_comparator
        self.track_comparator = track_comparator
        self.app_normalizer = app_normalizer
        self.vid_dims = vid_dims
        self.fps = fps
        self.single_frame_bias = single_frame_bias
        self.batch_size = batch_size

    def compute(self, tracklets,pairs):
        """ CNN/RNN hybrid strategy """
        timesteps = 24
        weights = [0.0 for _ in pairs]
        bar=progressbar.ProgressBar(max_value=len(pairs), prefix="weights")
        detection_w_idx=[]
        detection_w_d0=[]
        detection_w_d1=[]

        detection_y_idx=[]
        detection_y_d0_app=[]
        detection_y_d0_st=[]
        detection_y_d1_app=[]
        detection_y_d1_st=[]

        vid_w = self.vid_dims[1]
        vid_h = self.vid_dims[0]

        for i, (t0, t1) in bar(enumerate(pairs)):
            num_dets = min(len(tracklets[t0]), len(tracklets[t1]))
            num_dets = min(num_dets, timesteps)
            if num_dets == 1:
                # Single pair run can use rgb lookup for now
                # TODO: Switch this to appearence extraction to be
                # consistent
                d0 = tracklets[t0][-1]
                d1 = tracklets[t1][0]
                detection_w_idx.append(i)
                detection_w_d0.append(d0)
                detection_w_d1.append(d1)
            else:
                d0_app = np.array([self.app_normalizer(tracklets[t0][k]) for k in range(-num_dets, 0)])
                d1_app = np.array([self.app_normalizer(tracklets[t1][k]) for k in range(0, num_dets)])
                d0_st = np.array([[
                    float(tracklets[t0][k]['x']) / float(vid_w),
                    float(tracklets[t0][k]['y']) / float(vid_h),
                    float(tracklets[t0][k]['w']) / float(vid_w),
                    float(tracklets[t0][k]['h']) / float(vid_h),
                    float(tracklets[t0][k]['frame']) / float(self.fps)
                ] for k in range(-num_dets, 0)])
                d1_st = np.array([[
                    float(tracklets[t1][k]['x']) / float(vid_w),
                    float(tracklets[t1][k]['y']) / float(vid_h),
                    float(tracklets[t1][k]['w']) / float(vid_w),
                    float(tracklets[t1][k]['h']) / float(vid_h),
                    float(tracklets[t1][k]['frame']) / float(self.fps)
                ] for k in range(0, num_dets)])
                assert(d0_st[-1, -1] < d1_st[0, -1])
                min_frame = np.min(d0_st[:, -1])
                d0_st[:, -1] -= min_frame
                d1_st[:, -1] -= min_frame
                d0_app = constant_length(d0_app, timesteps, d0_app.shape[-1])
                d1_app = constant_length(d1_app, timesteps, d1_app.shape[-1])
                d0_st = constant_length(d0_st, timesteps, d0_st.shape[-1])
                d1_st = constant_length(d1_st, timesteps, d1_st.shape[-1])
                detection_y_idx.append(i)
                detection_y_d0_app.append(np.expand_dims(d0_app, axis=0))
                detection_y_d0_st.append(np.expand_dims(d0_st,axis=0))
                detection_y_d1_app.append(np.expand_dims(d1_app,axis=0))
                detection_y_d1_st.append(np.expand_dims(d1_st, axis=0))

        num_batches = math.ceil(len(detection_w_idx)/self.batch_size)
        bar=progressbar.ProgressBar(max_value=num_batches,prefix="batches")
        for bn in bar(range(num_batches)):
            start=bn*self.batch_size
            end=start+self.batch_size
            indices=detection_w_idx[start:end]
            for idx in indices:
                self.det_comparator.addPair(detection_w_d0[idx]['bgr'],
                                       detection_w_d1[idx]['bgr'])
            res = self.det_comparator.process()
            for res_idx, res in enumerate(res):
                # get global idx from result idx
                g_idx=int(indices[res_idx])
                res = res[0]
                orig_res = res
                res += self.single_frame_bias
                # bound between 0 and 1
                res = min(max(res, 0.0), 1.0)
                if res != 0:
                    res = np.log((1.0 - res) / res)
                else:
                    res = 1000000
                res= min(1000000, res)
                res = max(-1000000, res)
                weights[g_idx] = res

        #batch_size=track_comparator.batch
        num_batches = math.ceil(len(detection_y_idx)/self.batch_size)
        bar=progressbar.ProgressBar(max_value=num_batches,prefix="batches")
        for bn in bar(range(num_batches)):
            start=bn*self.batch_size
            end=start+self.batch_size
            indices=detection_y_idx[start:end]
            d0_app=np.vstack(detection_y_d0_app[start:end])
            d0_st=np.vstack(detection_y_d0_st[start:end])
            d1_app=np.vstack(detection_y_d1_app[start:end])
            d1_st=np.vstack(detection_y_d1_st[start:end])
            feed_dict={self.track_comparator.d0_app: d0_app,
                       self.track_comparator.d0_st: d0_st,
                       self.track_comparator.d1_app: d1_app,
                       self.track_comparator.d1_st: d1_st}
            res = self.track_comparator.session.run([self.track_comparator.output],feed_dict=feed_dict)[0][0]

            # Batch logic here would be faster...
            for res_idx, res in enumerate(res):
                # get global idx from result idx
                g_idx=int(indices[res_idx])
                res = np.log((1.0 - res) / res)
                res = min(1000000, res)
                res = max(-1000000, res)
                weights[g_idx] = res

        return weights
