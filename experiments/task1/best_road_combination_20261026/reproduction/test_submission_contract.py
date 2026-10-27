"""无需原始数据/GPU的提交契约测试：有效三任务及错误键/值必须拒绝。"""
from pathlib import Path
import tempfile,unittest
import pandas as pd
from merge_submission import merge
class ContractTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.r=Path(self.temp.name);(self.r/'parts/validation').mkdir(parents=True)
  self.state=dict(panel='R',timestamp='2031-03-01T00:00:00Z',station_id='S',link_id='L',mask_regime='R1',speed_kmh=60.,flow_vph=1200.)
  pd.DataFrame([self.state]).to_csv(self.r/'parts/validation/R.csv',index=False)
  self.queue=dict(panel='R',window_id='W',timestamp=self.state['timestamp'],link_id='L',queue_pred=1)
  pd.DataFrame([self.queue]).to_csv(self.r/'queue.csv',index=False)
  pd.DataFrame([dict(panel='R',departure_time='AM_TOKEN',path_id='P',path_flow=100)]).to_csv(self.r/'odme.csv',index=False)
  base=dict(split='validation',panel='R',timestamp=self.state['timestamp'],station_id='S',link_id='L',mask_regime='R1',window_id='W',departure_time='AM_TOKEN',path_id='P');pd.DataFrame([dict(base,submission_id=i+1,task=t) for i,t in enumerate(['state','queue','odme'])]).to_csv(self.r/'submission_key.csv',index=False)
 def tearDown(self):self.temp.cleanup()
 def run_merge(self):return merge(self.r,self.r/'parts',[self.r/'queue.csv'],[self.r/'odme.csv'],self.r/'submission.csv')
 def test_three_tasks_and_period_token(self):
  result=self.run_merge();self.assertEqual(result['rows'],3);f=pd.read_csv(self.r/'submission.csv');self.assertEqual(f.submission_id.tolist(),[1,2,3]);self.assertEqual(f.loc[0,'speed_kmh'],60);self.assertEqual(f.loc[1,'queue_pred'],1);self.assertEqual(f.loc[2,'path_flow'],100)
 def test_missing_state_key_rejected(self):
  pd.DataFrame([dict(self.state,station_id='WRONG')]).to_csv(self.r/'parts/validation/R.csv',index=False)
  with self.assertRaises(ValueError):self.run_merge()
 def test_duplicate_queue_key_rejected(self):
  pd.DataFrame([self.queue,self.queue]).to_csv(self.r/'queue.csv',index=False)
  with self.assertRaises(ValueError):self.run_merge()
 def test_nonbinary_queue_rejected(self):
  pd.DataFrame([dict(self.queue,queue_pred=.4)]).to_csv(self.r/'queue.csv',index=False)
  with self.assertRaises(AssertionError):self.run_merge()
 def test_extra_state_panel_rejected(self):
  pd.DataFrame([dict(self.state,panel='EXTRA')]).to_csv(self.r/'parts/validation/EXTRA.csv',index=False)
  with self.assertRaises(AssertionError):self.run_merge()
if __name__=='__main__':unittest.main()
