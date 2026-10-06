from __future__ import annotations
import argparse,json,traceback,shutil
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from runtime.config import load_runtime_config
from runtime.engine import AdaptiveEngine
from runtime.job_store import JobStore
from runtime.storage import resolve_inbox_transfer_dir

class App:
    def __init__(self,cfg): self.cfg=cfg; self.engine=AdaptiveEngine(cfg); self.jobs=JobStore(Path(cfg['paths']['output_root']))

def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def send(self,code,obj):
            b=json.dumps(obj,ensure_ascii=False).encode(); self.send_response(code); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
        def body(self): return json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))).decode())
        def do_GET(self):
            if self.path!='/v1/health': return self.send(404,{'status':'not_found'})
            self.send(200,{
                'status':'ok',
                'device':app.cfg.get('device','auto'),
                'modes':list(app.cfg['models']),
                'session_root':app.cfg['storage']['session_root'],
                'transfer_root':app.cfg['storage']['transfer_root'],
                'action_set_id':app.cfg['resources']['action_set_id'],
                'target_set_id':app.cfg['resources']['target_set_id'],
                'inbox_root':app.cfg['paths']['inbox_root'],
                'output_root':app.cfg['paths']['output_root'],
            })
        def do_POST(self):
            try: obj=self.body()
            except Exception as e: return self.send(400,{'status':'error','error':{'code':'BAD_JSON','message':str(e)}})
            if self.path=='/v1/decision':
                rid=str(obj.get('request_id',''))
                try:
                    job,is_new=app.jobs.get_or_reserve(rid,obj)
                    if not is_new:
                        if job['response'] is not None: return self.send(200,job['response'])
                        return self.send(202,{'protocol_version':'1.0','request_id':rid,'status':'processing','decision':None})
                    try: resp=app.engine.process(obj); app.jobs.complete(rid,resp); return self.send(200,resp)
                    except Exception as e:
                        resp={'protocol_version':'1.0','request_id':rid,'status':'error','decision':None,'error':{'stage':'runtime','code':type(e).__name__,'message':str(e)}}; app.jobs.fail(rid,resp); traceback.print_exc(); return self.send(500,resp)
                except ValueError as e: return self.send(409,{'protocol_version':'1.0','request_id':rid,'status':'error','decision':None,'error':{'code':str(e),'message':str(e)}})
            if self.path=='/v1/ack':
                try:
                    job=app.jobs.ack(str(obj['request_id']),bool(obj.get('accepted',False)))
                    if bool(obj.get('accepted',False)):
                        req=job.get('request') or {}; orca=(req.get('acquisition') or {}).get('orca') or {}; rel=orca.get('transfer_rel_dir')
                        if rel:
                            temp=resolve_inbox_transfer_dir(Path(app.cfg['paths']['inbox_root']),str(rel))
                            if temp.exists(): shutil.rmtree(temp)
                    return self.send(200,{'status':'ok','request_id':obj['request_id']})
                except Exception as e: return self.send(404,{'status':'error','error':{'code':type(e).__name__,'message':str(e)}})
            return self.send(404,{'status':'not_found'})
        def log_message(self,fmt,*args): print('[HTTP]',fmt%args)
    return H

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='adaptive_runtime_config.yaml'); args=ap.parse_args(); cfg=load_runtime_config(Path(args.config)); app=App(cfg); host=cfg.get('server',{}).get('host','0.0.0.0'); port=int(cfg.get('server',{}).get('port',8765)); print(f'Adaptive runtime listening on {host}:{port}'); print(f"Session root: {cfg['storage']['session_root']}"); print(f"Transfer root: {cfg['storage']['transfer_root']}"); print(f"Inbox root: {cfg['paths']['inbox_root']}"); ThreadingHTTPServer((host,port),make_handler(app)).serve_forever()
if __name__=='__main__': main()
