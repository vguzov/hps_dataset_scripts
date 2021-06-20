#!/usr/bin/env python

# BSD 3-Clause License
#
# Copyright (c) 2018, Centre National de la Recherche Scientifique
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os, sys
# if OpenGL was already loaded, we have to reload it after the
# PYOPENGL_PLATFORM variable is set...
ogl_module_names = list(k for k in sys.modules.keys() if k.startswith('OpenGL'))
for mod_name in ogl_module_names:
    del sys.modules[mod_name]
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import OpenGL.EGL as egl
import ctypes

# we have to define a few missing objects in PyOpenGL implementation
# (as of version 3.1.0)

def define_egl_ext_function(func_name, res_type, *arg_types):
    if hasattr(egl, func_name):
        return  # function already exists
    addr = egl.eglGetProcAddress(func_name)
    if addr is None:
        return  # function is not available
    else:
        proto = ctypes.CFUNCTYPE(res_type)
        proto.argtypes = arg_types
        globals()['proto__' + func_name] = proto    # avoid garbage collection
        func = proto(addr)
        setattr(egl, func_name, func)

def define_egl_ext_structure(struct_name):
    if hasattr(egl, struct_name):
        return  # structure already exists
    from OpenGL._opaque import opaque_pointer_cls
    setattr(egl, struct_name, opaque_pointer_cls(struct_name))

define_egl_ext_structure('EGLDeviceEXT')
define_egl_ext_function('eglGetPlatformDisplayEXT', egl.EGLDisplay)
define_egl_ext_function('eglQueryDevicesEXT', egl.EGLBoolean)
define_egl_ext_function('eglQueryDeviceStringEXT', ctypes.c_char_p)

EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_DRM_DEVICE_FILE_EXT = 0x3233

# utility function for egl attributes management
def egl_convert_to_int_array(dict_attrs):
    # convert to EGL_NONE terminated list
    attrs = sum(([ k, v] for k, v in dict_attrs.items()), []) + [ egl.EGL_NONE ]
    # convert to ctypes array
    return (egl.EGLint * len(attrs))(*attrs)

# expose objects at this level
from .context import EGLContext
