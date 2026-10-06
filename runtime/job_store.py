from __future__ import annotations
import hashlib, json, threading
from pathlib import Path

class JobStore:
    def __init__(self, output_root: Path):
        self.lock=threading.RLock(); self.jobs={}; self.output_root=Path(output_root)
    @staticmethod
    def digest(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    def _dir(self,obj): return self.output_root/obj['experiment']['experiment_id']/obj['experiment']['hole_uid']/f"k{int(obj['step']['completed_control_step']):02d}"
    def get_or_reserve(self,rid,obj):
        h=self.digest(obj)
        with self.lock:
            j=self.jobs.get(rid)
            if j:
                if j['hash']!=h: raise ValueError('REQUEST_ID_CONFLICT')
                return j,False
            d=self._dir(obj); rq=d/'request.json'; rp=d/'response.json'
            if rq.exists():
                previous=json.loads(rq.read_text(encoding='utf-8'))
                if self.digest(previous)!=h: raise ValueError('REQUEST_ID_CONFLICT')
                if rp.exists():
                    response=json.loads(rp.read_text(encoding='utf-8'))
                    j={'hash':h,'status':'done','response':response,'acked':(d/'ack.json').exists(),'request':obj,'dir':d}; self.jobs[rid]=j; return j,False
            j={'hash':h,'status':'processing','response':None,'acked':False,'request':obj,'dir':d}; self.jobs[rid]=j; return j,True
    def complete(self,rid,response):
        with self.lock: self.jobs[rid].update(status='done',response=response)
    def fail(self,rid,response):
        with self.lock: self.jobs[rid].update(status='error',response=response)
    def ack(self,rid,accepted):
        with self.lock:
            if rid not in self.jobs: raise KeyError(rid)
            j=self.jobs[rid]; j['acked']=bool(accepted)
            d=Path(j['dir']); d.mkdir(parents=True,exist_ok=True)
            (d/'ack.json').write_text(json.dumps({'request_id':rid,'accepted':bool(accepted)},indent=2),encoding='utf-8')
            return dict(j)
