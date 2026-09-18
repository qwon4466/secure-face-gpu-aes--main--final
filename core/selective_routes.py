"""Loopback-only UI; bounded jobs; browser keys never persist to files."""
import asyncio
import hmac
import json
from pathlib import Path
import secrets
import subprocess
import sys
import time
import uuid
import cv2
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from .selective_crypto import ROOT, ID, RestoreError, inspect_chunk, private_dir, read_bounded, unb64, section_bytes
from .selective_index import completed, find, summary
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap, InvalidUnwrap
import config


MAX_JOBS=24

def create_router(integrated, require_admin, audit=None):
    root=private_dir(integrated.root); chunks=integrated.chunks; results=private_dir(root/('.results' if integrated.operational else 'results'))
    app=APIRouter(prefix="/selective", dependencies=[Depends(require_admin)])
    sessions={}; jobs={}; active=asyncio.Lock(); tasks=set(); failures={}
    state={'recording':False,'preview':None,'faces':[],'stats':{},'error':None}

    def user(req):
        sid=req.cookies.get('srx_session'); s=sessions.get(sid)
        if not s or time.monotonic()-s['created']>3600:
            raise HTTPException(401,'화면을 새로 열어 세션을 생성하세요')
        return sid,s

    def owner(req,jid):
        sid,_=user(req)
        j=jobs.get(jid)
        if not j or j['owner']!=sid:
            raise HTTPException(404,'작업 없음')
        return j

    def chunk_path(cid):
        if not ID.fullmatch(cid):raise HTTPException(400,'청크 ID 오류')
        try:return find(chunks,cid)
        except RestoreError:raise HTTPException(404,'완료 청크 없음')

    async def secure_post(req:Request):
        _,session=user(req)
        if not secrets.compare_digest(req.headers.get('x-srx-csrf',''),session['csrf']):
            raise HTTPException(403,'요청 검증 실패')
        try:
            length=int(req.headers.get('content-length','0'))
        except ValueError:
            raise HTTPException(400,'요청 크기 오류')
        if length>32 or length<0:
            raise HTTPException(413,'요청은 최대 32바이트입니다')

    @app.get('/')
    async def index(req:Request):
        sid=req.cookies.get('srx_session')
        if sid not in sessions or time.monotonic()-sessions[sid]['created']>3600:
            if len(sessions)>=64:
                raise HTTPException(503,'세션 상한; 서버를 재시작하세요')
            sid=secrets.token_hex(32)
            sessions[sid]={'csrf':secrets.token_hex(32),'created':time.monotonic()}
        html=(ROOT/'templates/selective_restore.html').read_text().replace('__CSRF__',sessions[sid]['csrf'])
        response=HTMLResponse(html)
        response.set_cookie('srx_session',sid,httponly=True,samesite='strict',max_age=3600)
        return response

    @app.get('/api/status')
    async def status(req:Request):
        sid,_=user(req)
        if integrated is not None:state.update(integrated.web_state())
        return {k:v for k,v in state.items() if k!='preview'} | {'jobs':[{k:v for k,v in j.items() if k not in ('owner','path')} for j in jobs.values() if j['owner']==sid]}

    @app.get('/api/preview')
    async def preview(req:Request):
        user(req)
        if integrated is not None:state['preview']=integrated.get_jpeg()
        if state['preview'] is None:
            raise HTTPException(404,'아직 보호 프레임 없음')
        return Response(state['preview'],media_type='image/jpeg')

    @app.get('/api/chunks')
    async def listing(req:Request):
        user(req)
        return [summary(p,m,chunks) for p,m in completed(chunks)]

    @app.get('/api/chunks/{cid}/frame/{index}')
    async def protected_frame(req:Request,cid:str,index:int):
        user(req);p=chunk_path(cid)
        row=next((m for cp,m in completed(chunks) if cp==p),None)
        if row is None:raise HTTPException(404,'완료 청크 없음')
        count=len(row['frames'])
        if not 0<=index<count:raise HTTPException(400,'프레임 범위 오류')
        def decode():
            import os
            fd=os.memfd_create('srx-preview',os.MFD_CLOEXEC)
            with os.fdopen(fd,'w+b') as source:
                source.write(section_bytes(p,'protected'))
                source.flush()
                cap=cv2.VideoCapture('/proc/self/fd/'+str(source.fileno()))
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
                    if not ok:raise ValueError('보호 영상 프레임 읽기 실패')
                    ok,jpg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,88])
                    if not ok:raise ValueError('미리보기 인코딩 실패')
                    return jpg.tobytes()
                finally:cap.release()
        try:data=await asyncio.to_thread(decode)
        except ValueError:raise HTTPException(500,'보호 영상 미리보기 실패')
        return Response(data,media_type='image/jpeg',headers={'Cache-Control':'no-store'})

    def new_job(req,kind):
        sid,_=user(req)
        if len(jobs)>=MAX_JOBS:
            raise HTTPException(429,'작업 상한; 결과 보관 후 서버를 재시작하세요')
        if active.locked():
            raise HTTPException(409,'처리 중입니다')
        jid=uuid.uuid4().hex
        j={'id':jid,'owner':sid,'kind':kind,'status':'running','progress':0,'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        jobs[jid]=j
        return j

    def launch(coro):
        task=asyncio.create_task(coro); tasks.add(task); task.add_done_callback(tasks.discard)

    @app.post('/api/restore/{cid}', dependencies=[Depends(secure_post)])
    async def restore(req:Request,cid:str):
        p=chunk_path(cid)
        sid,_=user(req)
        actor=(sid,req.client.host if req.client else '')
        now=time.monotonic();count,until=failures.get(actor,(0,0))
        if now<until:
            raise HTTPException(429,'잠시 후 다시 시도하세요.')
        password=bytearray()
        async for block in req.stream():
            password.extend(block)
            if len(password)>32:raise HTTPException(413,'요청 크기 초과')
        supplied=bytes(password)
        role=None
        if hmac.compare_digest(supplied,config.SELECTIVE_EXTERNAL_PASSWORD.encode()):role='external'
        elif hmac.compare_digest(supplied,config.SELECTIVE_INTERNAL_PASSWORD.encode()):role='internal'
        for i in range(len(password)):password[i]=0
        if role is None:
            count+=1;failures[actor]=(count,time.monotonic()+(30 if count>=5 else 1))
            if audit:audit(req,cid,False,'invalid')
            await asyncio.sleep(.4)
            raise HTTPException(403,'선택적 복원 비밀번호가 올바르지 않습니다.')
        failures.pop(actor,None)
        key=bytearray(integrated.keys[role])
        j=new_job(req,'restore'); await active.acquire()
        j['chunk']=cid;j['role']=role
        output=results/j['id']
        async def work():
            try:
                j['progress']=25
                # exec creates fresh process memory; no recording DEKs or other keys passed.
                run=await asyncio.to_thread(subprocess.run,
                    [sys.executable,'-B','-m','core.selective_worker',str(p),str(output)],
                    cwd=ROOT,input=bytes(key),capture_output=True,timeout=180)
                if run.returncode:
                    raise ValueError('인증 실패')
                report=json.loads(run.stdout)
                j['progress']=75
                def convert():
                    return subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(output/'restored.avi'),
                        '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'restored.mp4')],
                        capture_output=True,timeout=180)
                converted=await asyncio.to_thread(convert)
                if converted.returncode:raise ValueError('결과 영상 변환 실패')
                if audit:audit(req,cid,True,role)
                j.update(status='complete',progress=100,completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),path=str(output),report={k:v for k,v in report.items() if k!='output'})
            except Exception:
                if audit:audit(req,cid,False,role)
                j.update(status='failed',error='청크 인증 또는 복원 오류')
            finally:
                for i in range(len(key)):
                    key[i]=0
                active.release()
        launch(work())
        return {'job':j['id']}

    @app.get('/api/jobs/{jid}')
    async def job(req:Request,jid:str):
        return {k:v for k,v in owner(req,jid).items() if k not in ('owner','path')}

    @app.get('/api/jobs/{jid}/download')
    async def download(req:Request,jid:str):
        j=owner(req,jid)
        if j['status']!='complete' or j['kind']!='restore':
            raise HTTPException(409,'복원 완료 결과가 없습니다')
        row=next((m for p,m in completed(chunks) if m['chunk']==j['chunk']),None)
        stamp=row['started_at'][:10]+'_'+row['started_at'][11:19].replace(':','-') if row else j['chunk']
        return FileResponse(Path(j['path'])/'restored.mp4',media_type='video/mp4',filename=f'selective_{j["role"]}_{stamp}.mp4',
                            content_disposition_type='inline' if req.query_params.get('inline')=='1' else 'attachment')

    @app.get('/api/jobs/{jid}/frame/{index}')
    async def frame(req:Request,jid:str,index:int):
        j=owner(req,jid)
        if j['status']!='complete' or j['kind']!='restore' or not 0<=index<j['report']['frames']:
            raise HTTPException(400,'프레임 범위 오류')
        return Response(read_bounded(Path(j['path'])/f'{index:06d}.png'),media_type='image/png')

    return app
