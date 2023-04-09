# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Event managing and triggering."""
import inspect
from functools import wraps
from . import singleton, make_decorator
from modnas.utils.logging import get_logger
from modnas.utils import merge_config


logger = get_logger(__name__)


@singleton
class EventManager():
    """Event manager class."""

    def __init__(self):
        self.handlers = {}
        self.event_queue = []

    def reset(self):
        """Reset event states."""
        self.handlers.clear()
        self.event_queue.clear()

    def get_handlers(self, ev):
        """Return generator over handlers of event."""
        ev_handlers = self.handlers.get(ev, [])
        for p, h in ev_handlers:
            yield h

    def on(self, ev, handler, priority=0):
        """Bind handler on event with priority."""
        logger.debug('on: %s %s %s' % (ev, handler, priority))
        ev_handlers = self.handlers.get(ev, [])
        ev_handlers.append((priority, handler))
        ev_handlers.sort(key=lambda s: -s[0])
        self.handlers[ev] = ev_handlers

    def emit(self, ev, *args, e_cb=None, e_delayed=False, e_merge_ret=False,
             e_chain_ret=True, e_fret=None, e_is_ret=False, **kwargs):
        """Trigger event with arguments."""
        logger.debug('emit: %s a: %s kw: %s d: %s' % (ev, len(args), len(kwargs), e_delayed))
        if ev not in self.handlers:
            return
        self.event_queue.append((ev, args, kwargs, e_cb))
        if e_delayed:
            return
        return self.dispatch_all(merge_ret=e_merge_ret, chain_ret=e_chain_ret, fret=e_fret, is_ret=e_is_ret)

    def off(self, ev, handler=None):
        """Un-bind handler on event."""
        logger.debug('off: %s %s' % (ev, handler))
        ev_handlers = self.handlers.get(ev, None)
        if ev_handlers is None:
            return
        if handler is None:
            del self.handlers[ev]
            return
        for i, (p, h) in enumerate(ev_handlers):
            if h == handler:
                ev_handlers.pop(i)
                break
        if not ev_handlers:
            del self.handlers[ev]

    def dispatch_all(self, merge_ret=False, chain_ret=True, fret=None, is_ret=False):
        """Trigger all delayed event handlers."""
        rets = {}
        self.event_queue, ev_queue = [], self.event_queue
        for ev_spec in ev_queue:
            ev, args, kwargs, callback = ev_spec
            ret = None
            for handler in self.get_handlers(ev):
                hret = handler(*args, **kwargs)
                logger.debug('handler: %s %s' % (handler, hret))
                if chain_ret and is_ret:
                    args = (fret if hret is None else hret, ) + args[1:]
                ret = merge_config(ret, hret) if merge_ret and hret is not None else hret
            if callback is not None:
                callback(ret)
            if len(ev_queue) == 1:
                return ret
            rets[ev] = ret
        return rets


@make_decorator
def event_hooked(func, name=None, before=True, after=True, pass_ret=True, qual=True, module=False, **emit_args):
    """Return wrapped function with hooked event triggers."""
    qual = func.__qualname__.split('.')[0] if qual is True else (None if qual is False else qual)
    module = func.__module__ if module is True else (None if module is False else module)
    name = func.__name__ if name is None else (None if name is False else name)
    ev = (module + '.' if module else '') + (qual + '.' if qual else '') + name
    ev_before = None if before is False else (('before' if before is True else before) + ':' + ev)
    ev_after = None if after is False else (('after' if after is True else after) + ':' + ev)

    @wraps(func)
    def wrapped(*args, **kwargs):
        ev_before = wrapped.ev_before
        ev_after = wrapped.ev_after
        if ev_before:
            hret = EventManager().emit(ev_before, *args, **kwargs, **emit_args)
            if hret is not None:
                args, kwargs = args if hret[0] is None else hret[0], kwargs if hret[1] is None else hret[1]
        fret = func(*args, **kwargs)
        if ev_after:
            if wrapped.pass_ret:
                args = (fret,) + args
            hret = EventManager().emit(ev_after, *args, **kwargs, e_fret=fret, e_is_ret=True, **emit_args)
            if hret is not None:
                return hret
        return fret
    wrapped._event_unhooked = func
    wrapped.ev_before = ev_before
    wrapped.ev_after = ev_after
    wrapped.pass_ret = pass_ret
    return wrapped


@make_decorator
def event_unhooked(func, remove_all=False, before=False, after=False):
    """Return function with event hooks removed."""
    func.ev_before = None if before is False else func.ev_before
    func.ev_after = None if after is False else func.ev_after
    if remove_all:
        return func._event_unhooked
    return func


@make_decorator
def event_hooked_method(obj, attr=None, method=None, *args, base_qual=True, **kwargs):
    """Return object with event hooked for given method."""
    if attr is None and inspect.ismethod(obj):
        attr = obj.__name__
        method = obj if method is None else method
        obj = obj.__self__
    if attr is None:
        attr = obj.__name__
    if method is None:
        method = getattr(obj, attr)
    if base_qual and 'qual' not in kwargs:
        cls = obj if inspect.isclass(obj) else obj.__class__
        bases = (cls,) + inspect.getmro(cls)
        for base in bases:
            if attr not in base.__dict__:
                continue
            cls = base
        kwargs['qual'] = cls.__name__
    setattr(obj, attr, event_hooked(method, *args, **kwargs))
    return obj


@make_decorator
def event_hooked_members(obj, *args, methods=None, is_method=False, is_function=False, **kwargs):
    """Return object with event hooked for member methods."""
    for attr, mem in inspect.getmembers(obj):
        if methods is not None and attr not in methods:
            continue
        if is_method and not inspect.ismethod(mem):
            continue
        if is_function and not inspect.isfunction(mem):
            continue
        event_hooked_method(obj, attr=attr, method=mem, *args, **kwargs)
    return obj


@make_decorator
def event_hooked_inst(cls, *args, **kwargs):
    """Return object factory with event hooked for member methods in instances."""
    @wraps(cls)
    def wrapped(*cls_args, **cls_kwargs):
        inst = cls(*cls_args, **cls_kwargs)
        event_hooked_members(inst, *args, is_method=True, **kwargs)
        return inst
    return wrapped


@make_decorator
def event_hooked_class(cls, *args, **kwargs):
    """Return class with event hooked for member methods."""
    event_hooked_members(cls, *args, is_function=True, **kwargs)
    return cls


@make_decorator
def event_hooked_subclass(cls, *args, **kwargs):
    """Return class with event hooked for member methods in all subclasses."""
    ori_new = cls.__new__

    def mod_new(cls, *fn_args, **fn_kwargs):
        if not mod_new.hooked.get(cls, False):
            event_hooked_class(cls, *args, **kwargs)
            mod_new.hooked[cls] = True
        return ori_new(cls) if ori_new == object.__new__ else ori_new(cls, *fn_args, **fn_kwargs)
    mod_new.hooked = {}
    setattr(cls, '__new__', mod_new)
    return cls


@make_decorator
def event_hooked_subclass_inst(cls, *args, **kwargs):
    """Return class with event hooked for member methods in all subclass instances."""
    ori_init = cls.__init__

    def mod_init(self, *fn_args, **fn_kwargs):
        ori_init(self, *fn_args, **fn_kwargs)
        event_hooked_members(self, *args, is_method=True, **kwargs)
    setattr(cls, '__init__', mod_init)
    return cls


event_on = EventManager().on
event_off = EventManager().off
event_emit = EventManager().emit
