from pathlib import Path,PurePosixPath
import hashlib,json,tarfile,posixpath
root=Path('/cann92');lock=json.loads((root/'official-image-lock.json').read_text())
assert lock['manifest_digest']=='sha256:0523f773498c81cf87b288b38c3c7626109463c798631d17d0ed9321c3c9fad0'
expected=[('9735e4842a98fd25d906d3bd349b3461fe5c3e631d485101861295ee325a98da',4691731401),('5f0fafc82f4b5f2deb0c1aeb4f3da674af66f014e8479f5f9961cdfc88516ae2',222),('f7145efeeea1c8e065d9d8a298b3827ee6371c0059358170590154141f28cbe1',1407)]
assert [(r['digest'],r['size']) for r in lock['selected_payload_layers']]==[('sha256:'+s,n) for s,n in expected]
assert lock['manifest']['layers'][5:8]==lock['selected_payload_layers']
for index,row in enumerate(lock['selected_payload_layers']):
 sha=row['digest'].removeprefix('sha256:');p=root/(sha+'.tar.gz');h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 assert p.stat().st_size==row['size'] and h.hexdigest()==sha,p
 if index==2:continue # Profile/OS layer retained only as evidence, never extracted.
 allowed='usr/local/Ascend' if index==0 else 'etc/Ascend'
 with tarfile.open(p,'r|gz') as tf:
  for member in tf:
   q=PurePosixPath(member.name)
   assert not q.is_absolute() and '..' not in q.parts,member.name
   name=str(q)
   if member.isdir() and name in ('usr','usr/local','etc'):continue
   assert name==allowed or name.startswith(allowed+'/'),name
   assert member.isdir() or member.isfile() or member.issym() or member.islnk(),name
   if member.issym() or member.islnk():
    target=member.linkname
    if member.issym() and not target.startswith('/'):
     target=posixpath.join(posixpath.dirname(name),target)
    target=posixpath.normpath(target).lstrip('/')
    assert target==allowed or target.startswith(allowed+'/'),(name,member.linkname,target)
 print('OFFICIAL_CANN_LAYER_VERIFIED',sha,flush=True)
