"""Bounded episode cache preserves frames and repeat-last action chunks."""
from collections import OrderedDict
import unittest
import numpy as np
from rulervla.training.main_data import SpatialDataset

class CacheTests(unittest.TestCase):
    def dataset(self,limit):
        d=SpatialDataset.__new__(SpatialDataset)
        d.rows=[dict(episode_index=i,frames=3,task='test') for i in range(3)]
        d.geometry={i:{} for i in range(3)}
        d.ends=np.array([3,6,9]);d.cache=OrderedDict();d.cache_bytes=0;d.max_cache_bytes=limit
        d._load_episode=lambda row,geo:dict(actions=np.arange(21,dtype=np.float32).reshape(3,7)+row['episode_index'],state8=np.ones((3,8),np.float32)*row['episode_index'])
        return d
    def test_eviction_does_not_change_any_sample_or_chunk(self):
        unlimited=self.dataset(None);bounded=self.dataset(180)
        for i in [0,3,6,1,4,7,2,5,8,0]:
            a,b=unlimited[i],bounded[i]
            self.assertEqual(a.keys(),b.keys())
            for k in a:
                if k=='task':self.assertEqual(a[k],b[k])
                else:np.testing.assert_array_equal(a[k],b[k])
            self.assertLessEqual(bounded.cache_bytes,180)
            self.assertEqual(len(bounded.cache),1)
        self.assertEqual(len(unlimited.cache),3)
if __name__=='__main__':unittest.main()
