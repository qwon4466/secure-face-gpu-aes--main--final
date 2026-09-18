"""Read completed selective chunks under the active raw_data tree."""
import json
from pathlib import Path
import re
from .selective_crypto import ID, RestoreError, read_bounded, RAW_NAME, RAW_META_NAME
DATE=re.compile(r'^\d{4}-\d{2}-\d{2}$')
HOUR=re.compile(r'^\d{2}$')
STAMP=re.compile(r'^\d{2}-\d{2}-\d{2}$')

def completed(root):
    root=Path(root)
    if not root.exists():return []
    paths=[]
    # Test/file fixtures keep the old UUID layout; operational data uses date/hour/time.
    for p in root.iterdir():
        if p.is_symlink():continue
        if ID.fullmatch(p.name) and p.is_dir():paths.append(p)
        elif DATE.fullmatch(p.name) and p.is_dir():
            for hour in p.iterdir():
                if hour.is_symlink() or not HOUR.fullmatch(hour.name) or not hour.is_dir():continue
                for chunk in hour.iterdir():
                    if not chunk.is_symlink() and STAMP.fullmatch(chunk.name) and chunk.is_dir():paths.append(chunk)
    result=[]
    for p in paths:
        try:
            if (p/'manifest.json').is_file():
                doc=json.loads(read_bounded(p/'manifest.json',16*1024*1024))
                files=('manifest.json','protected.avi',RAW_NAME,RAW_META_NAME)
            else:
                index=json.loads(read_bounded(p/RAW_META_NAME,16*1024*1024))
                if index.get('format')!='srx-container-v2' or index.get('status')!='completed':continue
                doc=index['document'];files=(RAW_NAME,RAW_META_NAME)
            meta=doc['manifest']['meta']
            if (meta.get('complete') is True and ID.fullmatch(meta['chunk'])
                    and all((p/name).is_file() and not (p/name).is_symlink()
                            for name in files)):
                result.append((p,meta))
        except (OSError,KeyError,ValueError,TypeError):continue
    return sorted(result,key=lambda entry:(entry[1].get('started_at',''),str(entry[0])),reverse=True)

def find(root,chunk_id):
    if not ID.fullmatch(chunk_id):raise RestoreError('청크 ID 오류')
    for path,meta in completed(root):
        if meta['chunk']==chunk_id:
            if not path.resolve().is_relative_to(Path(root).resolve()):raise RestoreError('청크 경로 오류')
            return path
    raise RestoreError('완료 청크 없음')

def summary(path,meta,root):
    frames=meta['frames'];fps=meta['fps']
    rel=path.relative_to(root).as_posix()
    started=meta.get('started_at') or (path.parent.parent.name+' '+path.name.replace('-',':') if STAMP.fullmatch(path.name) else '')
    counts={g:sum(f.get('group')==g for row in frames for f in row['faces']) for g in ('external','internal','master')}
    return {'id':meta['chunk'],'name':rel,'date':started[:10],'started_at':started,'ended_at':meta.get('ended_at',''),
            'duration_seconds':round(meta.get('duration_seconds',len(frames)/fps),3),
            'frames':len(frames),'fps':fps,'shape':meta['shape'],'groups':counts,
            'path':rel,'complete':True,'drops':meta.get('dropped_frames',0)}
