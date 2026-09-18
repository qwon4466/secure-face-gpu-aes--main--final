"""Restorer process: only explicit chunk, output and stdin key. No recording imports."""
import json
import os
import sys
from .selective_crypto import restore_to

def main():
    os.umask(0o077)
    try:
        key=sys.stdin.buffer.read(33)
        if len(key) not in (0,32):
            raise ValueError('키 크기 오류')
        result=restore_to(sys.argv[1],sys.argv[2],key or None)
        print(json.dumps(result))
    except Exception:
        print(json.dumps({'error':'잘못된 키 또는 청크 인증/저장 오류'}))
        sys.exit(2)
if __name__=='__main__':
    main()
