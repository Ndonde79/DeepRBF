# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
# Founder and CEO: Ramadhani W. Ndonde.
"""Restartable joint SIS ensembles with audited state and sampling uncertainty."""
import hashlib
import json
import os
from functools import wraps
from pathlib import Path
import numpy as np
from .lithology import simulate_lithology_sis


def _fingerprint(arrays,settings):
    digest=hashlib.sha256(json.dumps(settings,sort_keys=True,allow_nan=False).encode())
    for array in arrays:
        array=np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def _exclusive_checkpoint(function):
    @wraps(function)
    def locked(*args,**kwargs):
        path=Path(kwargs['checkpoint_path']);path.parent.mkdir(parents=True,exist_ok=True)
        # OS locks are released on process death, so interruption does not leave
        # an unrecoverable sentinel. The one-byte lock file may remain safely.
        with path.with_name(path.name+'.lock').open('a+b') as stream:
            stream.seek(0,2)
            if stream.tell()==0:stream.write(b'0');stream.flush()
            stream.seek(0)
            if os.name=='nt':
                import msvcrt
                try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                except OSError as exc:raise RuntimeError('Checkpoint is already in use') from exc
            else:
                import fcntl
                try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except OSError as exc:raise RuntimeError('Checkpoint is already in use') from exc
            try:return function(*args,**kwargs)
            finally:
                stream.seek(0)
                if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
                else:fcntl.flock(stream,fcntl.LOCK_UN)
    return locked


@_exclusive_checkpoint
def simulate_sis_ensemble(sample_xyz,sample_categories,target_xyz,*,n_realizations,
                          checkpoint_path,batch_size=10,seed=42,**settings):
    """Checkpoint across realizations, never split the jointly simulated targets.

    Restart requires identical inputs, settings and native binary. Each realization
    uses seed + 104729 * absolute_realization_index. Counts, progress, fingerprint
    and checksum are committed in one atomic NPZ replacement. Existing completed
    states may be extended to a larger ensemble. Returned probabilities summarize
    draws; they do not measure geological model uncertainty beyond this hypothesis.
    """
    for value,name in [(n_realizations,'n_realizations'),(batch_size,'batch_size'),(seed,'seed')]:
        if not isinstance(value,int) or isinstance(value,bool) or value<(0 if name=='seed' else 1):
            raise ValueError(name+' must be a valid nonnegative seed or positive count')
    if settings.get('backend','native')!='native':raise ValueError('Checkpointed ensembles require the native deterministic generator')
    if seed+104729*(n_realizations-1)>2**64-1:raise ValueError('Ensemble seed sequence exceeds uint64')
    import deeprbf_recovered as native
    settings=dict(settings,backend='native')
    # Stable JSON representations for rotation/ranges and explicit category priors.
    canonical={key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in settings.items()}
    if 'proportions' in canonical and canonical['proportions'] is not None:
        canonical['proportions']={str(int(k)):float(v) for k,v in canonical['proportions'].items()}
    canonical.update(seed=seed,native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest())
    canonical['implementation']=_implementation_identity()
    fingerprint=_fingerprint([np.asarray(sample_xyz,float),np.asarray(sample_categories),np.asarray(target_xyz,float)],canonical)
    path=Path(checkpoint_path);path.parent.mkdir(parents=True,exist_ok=True)
    categories=np.unique(sample_categories);counts=np.zeros((len(target_xyz),len(categories)),dtype=np.int64)
    completed=0
    if path.exists():
        with np.load(path,allow_pickle=False) as state:
            metadata=json.loads(str(state['metadata']))
            if metadata['fingerprint']!=fingerprint:raise ValueError('Checkpoint input/settings/native binary fingerprint mismatch')
            saved=state['counts'];completed=metadata['completed']
            if type(completed) is not int or completed<0:raise ValueError('Invalid checkpoint progress')
            if saved.shape!=counts.shape or saved.dtype!=counts.dtype or (saved<0).any() or not np.all(saved.sum(1)==completed):
                raise ValueError('Checkpoint count/progress invariant failed')
            if hashlib.sha256(saved.tobytes()).hexdigest()!=metadata['counts_sha256']:raise ValueError('Checkpoint checksum mismatch')
            counts=saved.copy()
        if completed>n_realizations:raise ValueError('Requested ensemble is smaller than its checkpoint')
    while completed<n_realizations:
        size=min(batch_size,n_realizations-completed)
        result=simulate_lithology_sis(sample_xyz,sample_categories,target_xyz,n_realizations=size,
                                      seed=seed+104729*completed,**settings)
        np.testing.assert_array_equal(result.categories,categories)
        for index,category in enumerate(categories):counts[:,index]+=np.sum(result.realizations==category,axis=0)
        completed+=size
        metadata=dict(fingerprint=fingerprint,completed=completed,settings=canonical,
                      counts_sha256=hashlib.sha256(counts.tobytes()).hexdigest())
        temporary=path.with_name(path.name+'.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,counts=counts,metadata=json.dumps(metadata,sort_keys=True))
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    probabilities=counts/completed
    # Wilson intervals avoid the false certainty of plug-in SE=0 at zero counts.
    z=1.959963984540054;denominator=1+z*z/completed
    halfwidth=z*np.sqrt(probabilities*(1-probabilities)/completed+z*z/(4*completed**2))/denominator
    return dict(categories=categories,probabilities=probabilities,realizations=completed,
                maximum_probability_mcse=float(np.sqrt(probabilities*(1-probabilities)/completed).max()),
                maximum_wilson_95_halfwidth=float(halfwidth.max()),checkpoint=str(path),fingerprint=fingerprint)


def _canonical(value):
    """Hash nested numerical settings without lossy JSON array conversion."""
    from dataclasses import asdict, is_dataclass
    if is_dataclass(value):return _canonical(asdict(value))
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject:raise ValueError('Object arrays cannot be checkpointed')
        return dict(dtype=str(value.dtype),shape=list(value.shape),
                    sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
    if isinstance(value,np.generic):return _canonical(value.item())
    if isinstance(value,dict):return {str(k):_canonical(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_canonical(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return dict(nonfinite=repr(value))
    return value


def _implementation_identity():
    import deeprbf_recovered as native
    import scipy
    paths=[Path(native.__file__),*sorted(Path(__file__).parent.glob('*.py'))]
    return dict(numpy=np.__version__,scipy=scipy.__version__,
                files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def _restart_counts(draw,categories,shape,active,identity,*,n_realizations,
                    checkpoint_path,batch_size,seed,retain_realizations=False):
    for value,name in ((n_realizations,'n_realizations'),(batch_size,'batch_size'),(seed,'seed')):
        if type(value) is not int or value<(0 if name=='seed' else 1):
            raise ValueError(name+' must be a nonnegative seed or positive count')
    if seed+104729*(n_realizations-1)>2**32-1:
        raise ValueError('Ensemble seed sequence exceeds uint32')
    cats=np.asarray(categories)
    if cats.ndim!=1 or not len(cats) or len(np.unique(cats))!=len(cats):
        raise ValueError('Unique nonempty category catalogue required')
    active=np.asarray(active,bool)
    if active.shape!=shape or not active.any():raise ValueError('Nonempty matching simulation mask required')
    if type(retain_realizations) is not bool:raise ValueError('retain_realizations must be boolean')
    identity=_canonical(dict(schema=2,inputs=identity,seed=seed,retain_realizations=retain_realizations,implementation=_implementation_identity()))
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()
    path=Path(checkpoint_path)
    draw_dir=path.with_name(path.name+'.draws');draw_hashes=[]
    if retain_realizations:draw_dir.mkdir(exist_ok=True)
    counts=np.zeros(shape+(len(cats),),dtype=np.int64);completed=0
    if path.exists():
        with np.load(path,allow_pickle=False) as state:
            metadata=json.loads(str(state['metadata']))
            if metadata['fingerprint']!=fingerprint:raise ValueError('Checkpoint fingerprint mismatch')
            completed=metadata['completed'];saved=state['counts']
            if type(completed) is not int or completed<0 or saved.shape!=counts.shape or saved.dtype!=counts.dtype:
                raise ValueError('Checkpoint count/progress invariant failed')
            if (saved<0).any() or not np.all(saved[active].sum(-1)==completed) or np.any(saved[~active]):
                raise ValueError('Checkpoint count/progress invariant failed')
            if hashlib.sha256(saved.tobytes()).hexdigest()!=metadata['counts_sha256']:
                raise ValueError('Checkpoint checksum mismatch')
            counts=saved.copy()
            if retain_realizations:
                draw_hashes=metadata.get('draw_sha256',[])
                if len(draw_hashes)!=completed:raise ValueError('Checkpoint realization inventory mismatch')
                for index,digest in enumerate(draw_hashes):
                    file=draw_dir/f'{index:08d}.npy'
                    if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
                        raise ValueError('Retained realization missing or checksum mismatch')
        if completed>n_realizations:raise ValueError('Requested ensemble is smaller than its checkpoint')
    while completed<n_realizations:
        end=min(completed+batch_size,n_realizations)
        for index in range(completed,end):
            values=np.asarray(draw(seed+104729*index))
            if values.shape!=shape or not np.isin(values[active],cats).all():
                raise RuntimeError('Simulation shape/category audit failed')
            if retain_realizations:
                file=draw_dir/f'{index:08d}.npy';temporary=file.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.save(stream,values,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
                os.replace(temporary,file);draw_hashes.append(hashlib.sha256(file.read_bytes()).hexdigest())
            for column,category in enumerate(cats):counts[...,column]+=(values==category)&active
        metadata=dict(fingerprint=fingerprint,completed=end,identity=identity,
                      draw_sha256=draw_hashes,
                      counts_sha256=hashlib.sha256(counts.tobytes()).hexdigest())
        temporary=path.with_name(path.name+'.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,counts=counts,metadata=json.dumps(metadata,sort_keys=True,allow_nan=False))
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path);completed=end
    probabilities=counts/completed;probabilities[~active]=np.nan
    p=probabilities[active];z=1.959963984540054
    halfwidth=z*np.sqrt(p*(1-p)/completed+z*z/(4*completed**2))/(1+z*z/completed)
    return dict(categories=cats,probabilities=probabilities,realizations=completed,
                maximum_probability_mcse=float(np.sqrt(p*(1-p)/completed).max()),
                maximum_wilson_95_halfwidth=float(halfwidth.max()),checkpoint=str(path),fingerprint=fingerprint,
                realization_directory=str(draw_dir) if retain_realizations else None)


@_exclusive_checkpoint
def simulate_pgs_ensemble(sample_xyz,sample_categories,target_xyz,*,n_realizations,
                          checkpoint_path,batch_size=10,seed=42,conditioning_backend='dense_spectral',retain_realizations=False,**settings):
    """Restartable dense-spectral or sparse-Vecchia PGS draws.

    The absolute draw seed is seed + 104729*i, including its spectrum. This is
    batch-independent, but differs from simulate_plurigaussian's shared-spectrum
    multi-draw call. retain_realizations=True additionally stores checksummed
    geometry files beside the checkpoint. Sparse Vecchia is an approximation.
    """
    from .lithology import simulate_plurigaussian
    if conditioning_backend not in ('dense_spectral','sparse_vecchia'):raise ValueError('Unknown PGS conditioning backend')
    if conditioning_backend=='sparse_vecchia':
        from .sparse_pgs import simulate_plurigaussian_sparse as simulate_plurigaussian
    xyz=np.asarray(sample_xyz,float);labels=np.asarray(sample_categories);q=np.asarray(target_xyz,float)
    def draw(draw_seed):
        return simulate_plurigaussian(xyz,labels,q,n_realizations=1,seed=draw_seed,**settings).realizations[0]
    catalogue=settings.get('category_catalogue')
    if catalogue is None:catalogue=np.unique(labels)
    return _restart_counts(draw,catalogue,(len(q),),np.ones(len(q),bool),
                           dict(method='pgs_independent_spectra',conditioning_backend=conditioning_backend,xyz=xyz,labels=labels,targets=q,settings=settings),
                           n_realizations=n_realizations,checkpoint_path=checkpoint_path,batch_size=batch_size,seed=seed,retain_realizations=retain_realizations)


@_exclusive_checkpoint
def simulate_snesim_ensemble(training_image,simulation_shape,*,n_realizations,
                             checkpoint_path,batch_size=10,seed=42,retain_realizations=False,**settings):
    """Restart native or Python SNESIM, including masks and advanced Python maps.

    Inactive probabilities are NaN. TI, conditioning, maps, configuration and
    implementation are fingerprinted. Counts summarize one TI hypothesis only.
    """
    from .snesim import simulate_snesim,UNKNOWN
    ti=np.asarray(training_image);shape=tuple(simulation_shape)
    if 'return_diagnostics' in settings:raise ValueError('Ensemble controls return_diagnostics')
    cats=settings.get('categories')
    if cats is None:cats=np.unique(ti[ti!=UNKNOWN])
    active=settings.get('simulation_mask')
    if active is None:active=np.ones(shape,bool)
    def draw(draw_seed):
        return simulate_snesim(ti,shape,n_realizations=1,seed=draw_seed,
                               return_diagnostics=False,**settings)[0]
    return _restart_counts(draw,cats,shape,active,dict(method='snesim',ti=ti,shape=shape,settings=settings),
                           n_realizations=n_realizations,checkpoint_path=checkpoint_path,batch_size=batch_size,seed=seed,retain_realizations=retain_realizations)
