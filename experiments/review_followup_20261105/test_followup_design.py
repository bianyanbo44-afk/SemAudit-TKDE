"""Pre-scoring structural tests, independent of scientific test outcomes."""
from dataclasses import replace
import unittest
import numpy as np
import torch
import common as c
import neural_calibration as n
from strict_history import transform

class DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=c.source('EPA-Air','selective_ssm')[2]['config']
        # Generated fixture only; no real targets or fitted checkpoint scores.
        cls.records={s:[dict(sample_id=k*10+i,entity=f'{s}_{i}',start_time=0.,schedule=0,
              events=[dict(event_time=.2,arrival_time=.2,modality=m,value=torch.tensor([.1*(m+1)]),quality=1.,event_id=m)for m in range(4)]+
                     ([dict(event_time=float(j+1),arrival_time=float(j+1),modality=4,value=torch.arange(384,dtype=torch.float32)*(j+1)/384,
                            quality=1.,event_id=4+j)for j in range(3)]if i else []),
              baseline=[0.]*4,signal=1.,noise=[0.]*4,text_count=3 if i else 0)for i in range(2)]for k,s in enumerate(['train','val','test'])}
    def test_four_outputs_and_independent_entities(self):
        d=n.make_data(self.records,.1,'selective_ssm')
        self.assertEqual(d.input_dims,(1,1,1,1,384))
        self.assertEqual(d.train_windows[0].targets.shape,(1,4))
        self.assertFalse({d.window_info[w.sample_id].entity for w in d.train_windows}&{d.window_info[w.sample_id].entity for w in d.test_windows})
    def test_strength_does_not_change_inputs_or_noise(self):
        a=n.make_data(self.records,0.,'selective_ssm');b=n.make_data(self.records,1.,'selective_ssm')
        for x,y in zip(a.test_windows,b.test_windows):
            self.assertTrue(all(torch.equal(e.value,f.value)for e,f in zip(x.events,y.events)))
            torch.testing.assert_close(y.targets-x.targets,torch.tensor([[1.,-1.,.5,-.5]]))
    def test_historical_controls_preserve_units_multisets_and_empty_histories(self):
        d=n.make_data(self.records,.1,'selective_ssm')
        for name in ['history_rotation','history_nearest_swap']:
            w,m,diag=transform(d.test_windows,d.window_info,d.text_modality,name)
            self.assertEqual(diag['donor_after_cutoff'],0)
            self.assertEqual(len(w[0].events),4)
            self.assertTrue(all(r['donor_arrival']<=r['recipient_cutoff']for r in m))
            c.audit.verify_invariants(d.test_windows,w,d.text_modality)
    def test_original_family_forward_backward_with_empty_text_and_new_width(self):
        for family in c.FAMILIES:
            d=n.make_data(self.records,.1,family);model=c.build(family,d,self.config,torch.device('cpu'))
            split=c.prepare_timeimm_split(model,d.train_windows,d.input_dims,4)
            pred=model.forward_packed_queries(split.packed,split.query_times)
            self.assertTrue(torch.isfinite(pred).all(),family)
            pred.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all()for p in model.parameters()if p.grad is not None),family)
    def test_nonempty_original_width_weights_and_outputs_unchanged(self):
        data=n.make_data(self.records,.1,'immtsf_recavg_gradd')
        # Expand only the test fixture to the original width.
        def wide(w):return replace(w,events=tuple(replace(e,value=torch.cat([e.value,e.value]))if e.modality==4 else e for e in w.events))
        data=replace(data,text_embedding_dim=768,train_windows=tuple(wide(w)for w in data.train_windows))
        for family in ['immtsf_recavg_gradd','immtsf_t2v_xattn']:
            c.set_deterministic_seed(53);a=c.audit.build_model(family,data,self.config,torch.device('cpu')).eval()
            c.set_deterministic_seed(53);b=c.build(family,data,self.config,torch.device('cpu')).eval()
            self.assertEqual(c.audit.state_digest(a),c.audit.state_digest(b))
            split=c.prepare_timeimm_split(a,[data.train_windows[1]],data.input_dims,4)
            torch.testing.assert_close(a.forward_packed_queries(split.packed,split.query_times),b.forward_packed_queries(split.packed,split.query_times),rtol=0,atol=0)

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main(verbosity=2)
